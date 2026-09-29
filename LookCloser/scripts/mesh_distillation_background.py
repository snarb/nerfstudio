"""Audited, quiet background pilot for the opt-in DEC5 distillation campaign.

Only frozen train RGB is used for fitting. The six validation views are a subset
of train, and the actual held-out view is never opened by the pilot.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from nerfstudio.model_components.mesh_distillation import SeparateBackground, weighted_charbonnier

BUNDLE = Path('/mnt/data/dec5_000973_mesh_distillation_v1')
OUTPUT = Path('/mnt/data/dec5_lookcloser_mesh_distillation_v1')
VALIDATION = (0, 15, 30, 33, 45, 58)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def publish_hardlink(source, destination):
    """Atomically publish immutable bytes without a duplicate checkpoint copy."""
    source,destination=Path(source),Path(destination)
    temporary=destination.with_name(destination.name+'.link');temporary.unlink(missing_ok=True)
    # rename(a,b) is a no-op if a and b are already links to the same inode;
    # avoid leaving the temporary link behind when the selected step is stable.
    if destination.exists() and os.path.samefile(source,destination):return
    os.link(source,temporary);temporary.replace(destination)


def train_rows(bundle):
    meta = read(bundle/'real/transforms.json'); names = set(meta['train_filenames'])
    return [r for r in meta['frames'] if r['file_path'] in names]


def rays(row, yy, xx, device='cuda'):
    yy = torch.as_tensor(yy, device=device).float(); xx = torch.as_tensor(xx, device=device).float()
    pose = torch.tensor(row['transform_matrix'], device=device, dtype=torch.float32)
    local = torch.stack([(xx + .5 - row['cx'])/row['fl_x'],
                         -(yy + .5 - row['cy'])/row['fl_y'], -torch.ones_like(xx)], -1)
    norm = local.norm(dim=-1, keepdim=True)
    direction = (local/norm) @ pose[:3, :3].T
    return pose[:3, 3].expand_as(direction), direction, norm


def check(output, stage, step, start, **extra):
    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # kill(pid, 0) returning EPERM means the process exists but is
            # owned by another user (for example after reparenting). A
            # liveness diagnostic must not abort an otherwise healthy fit.
            return True
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True)
    row = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(),
               ppid=os.getppid(), controller_alive=alive(os.getppid()), worker_alive=alive(os.getpid()),
               supervision='synchronous controller; worker heartbeat', stage=stage, step=step,
               elapsed_seconds=time.monotonic()-start, gpu=gpu.stdout.strip(),
               gpu_query_returncode=gpu.returncode,
               cuda_oom_count=torch.cuda.memory_stats().get('num_ooms',0) if torch.cuda.is_initialized() else 0,
               cuda_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
               **extra)
    with (output/'campaign_checks.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    write(output/'progress.json', row)
    print(f'stage={stage} step={step} seconds={row["elapsed_seconds"]:.0f} gpu={row["gpu"]}', flush=True)


def masks(bundle):
    root=bundle/'config/source_masks'; a=np.load(root/'masks.npz')['masks']
    return dict(zip(read(root/'cameras.json'), a))


def prepare(bundle, output):
    import trimesh
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'background_request.json'
    identity=dict(bundle=str(bundle), input_hashes=sha(bundle/'dataset_hashes.json'),
                  script_sha256=sha(__file__), module_sha256=sha(Path(__file__).resolve().parents[2]/'nerfstudio/model_components/mesh_distillation.py'),
                  validation_train_indices=list(VALIDATION), actual_eval_used=False, seed=42,
                  boundary_pixels=16, samples=48, texture_resolution=1024)
    if manifest.exists():
        if read(manifest)!=identity: raise ValueError('Background request mismatch; use a new pilot directory')
        return
    rows=train_rows(bundle); mask_by_name=masks(bundle)
    norm=read(bundle/'mesh/normalization.json'); source=Path(norm['data'])
    depth_rows={r['physical_camera']:r for r in read(source/'transforms.json')['frames']}
    bounds=trimesh.load(bundle/'mesh/teacher.ply',process=False).bounds
    pad=np.maximum((bounds[1]-bounds[0])*.1,.005); actor=np.array([bounds[0]-pad,bounds[1]+pad])
    rng=np.random.default_rng(42); points=[]; source_ids=[]; audits=[]
    review=output/'mask_review';review.mkdir(exist_ok=True)
    for i,row in enumerate(rows):
        mask=mask_by_name[row['physical_camera']]>0
        fg=cv2.erode(mask.astype('uint8'),np.ones((33,33),np.uint8))>0
        bg=cv2.dilate(mask.astype('uint8'),np.ones((33,33),np.uint8))==0
        dr=depth_rows[row['physical_camera']]
        for k in ('fl_x','fl_y','cx','cy','w','h'):
            if not np.isclose(row[k],dr[k]): raise ValueError('Depth calibration mismatch')
        with gzip.open(source/dr['depth_file_path'],'rb') as f: depth=np.load(f)*norm['dataparser_scale']
        yy,xx=np.where(bg & (depth>0) & np.isfinite(depth)); ix=rng.choice(len(xx),min(len(xx),2048),False)
        yy,xx=yy[ix],xx[ix]; o,d,raynorm=rays(row,yy,xx,'cpu')
        p=(o+d*torch.from_numpy(depth[yy,xx])[:,None]*raynorm).numpy()
        keep=~((p>=actor[0]-.02)&(p<=actor[1]+.02)).all(-1)
        # Internal validation cameras are excluded from geometry fitting as well.
        if i not in VALIDATION: points.append(p[keep]);source_ids.append(np.full(keep.sum(),i))
        fy,fx=np.where(fg & (depth>0));fy,fx=fy[::32],fx[::32]
        oo,dd,nn=rays(row,fy,fx,'cpu');fp=(oo+dd*torch.from_numpy(depth[fy,fx])[:,None]*nn).numpy()
        fraction=float(((fp>=actor[0])&(fp<=actor[1])).all(-1).mean())
        audits.append(dict(image=row['file_path'],foreground_points_inside=fraction,
                           background_depth_coverage=float(((depth>0)&bg).sum()/max(bg.sum(),1))))
        rgb=np.array(Image.open(bundle/'real'/row['file_path']))
        thumb=rgb.copy();edge=~fg&~bg;thumb[edge]=(thumb[edge]*.45+np.array([0,255,0])*.55).astype('uint8')
        Image.fromarray(thumb).resize((480,270)).save(review/f'{i:04d}.jpg')
    p=np.concatenate(points);sources=np.concatenate(source_ids)
    sample=p[rng.choice(len(p),min(len(p),10000),False)]; best_score=-1;best=None
    for _ in range(1500):
        ix=rng.choice(len(p),3,False)
        if len(np.unique(sources[ix]))<3:continue
        a,b,c=p[ix];n=np.cross(b-a,c-a);length=np.linalg.norm(n)
        if length<1e-8:continue
        n/=length;offset=-a@n;score=(abs(sample@n+offset)<.005).sum()
        if score>best_score:best_score=score;best=(n,offset)
    n,offset=best
    if actor.mean(0)@n+offset<0:n,offset=-n,-offset
    corners=np.array(np.meshgrid(*zip(actor[0],actor[1]))).T.reshape(-1,3)
    separation=float((corners@n+offset).min())
    if separation<.05:raise ValueError('Candidate background is not clearly behind actor')
    # UV basis and extent cover all train rays, but fitting uses no validation RGB.
    axis=np.eye(3)[np.argmin(abs(n))];u=np.cross(axis,n);u/=np.linalg.norm(u);v=np.cross(n,u)
    basis=np.stack([u,v]);uv=[]
    for row in rows:
        yy,xx=np.meshgrid(np.linspace(0,1079,8),np.linspace(0,1919,12),indexing='ij')
        o,d,_=rays(row,yy.ravel(),xx.ravel(),'cpu');o,d=o.numpy(),d.numpy()
        t=-(o@n+offset)/(d@n);uv.append((o+d*t[:,None])@basis.T)
    uv=np.concatenate(uv);uv_bounds=np.array([uv.min(0)-.03,uv.max(0)+.03])
    bg_points=p[(p@n+offset)<separation-.03]
    bg_bounds=np.quantile(bg_points,[.01,.99],axis=0)+np.array([[-.08],[.08]])
    geometry=dict(plane=[*n.tolist(),float(offset)],basis=basis.tolist(),uv_bounds=uv_bounds.tolist(),
                  bounds=bg_bounds.tolist(),actor_bounds=actor.tolist(),behind_limit=separation-.03)
    write(output/'geometry.json',geometry)
    write(output/'geometry_audit.json',dict(views=audits,plane_inlier_fraction=float((abs(p@n+offset)<.005).mean()),
        train_only_plane_fit=True, actor_box_source='teacher bounds + max(10 percent extent, .005)',
        raw_depth_hashes={r['physical_camera']:sha(source/depth_rows[r['physical_camera']]['depth_file_path']) for r in rows}))
    write(manifest,identity)


def load_data(bundle, scale=2, with_depth=False):
    rows=train_rows(bundle);ms=masks(bundle);data=[]
    if with_depth:
        normalization=read(bundle/'mesh/normalization.json');depth_root=Path(normalization['data'])
        depth_rows={r['physical_camera']:r for r in read(depth_root/'transforms.json')['frames']}
    for row in rows:
        rgb=np.array(Image.open(bundle/'real'/row['file_path']))
        bg=cv2.dilate(ms[row['physical_camera']],np.ones((33,33),np.uint8))==0
        data.append(dict(row=row, rgb=torch.from_numpy(rgb[::scale,::scale].copy()).cuda().float()/255,
                         mask=torch.from_numpy(bg[::scale,::scale].copy()).cuda()))
        if with_depth:
            path=depth_root/depth_rows[row['physical_camera']]['depth_file_path']
            with gzip.open(path,'rb') as f:depth=np.load(f)*normalization['dataparser_scale']
            data[-1]['depth']=torch.from_numpy(depth[::scale,::scale].copy()).cuda()
    return data


@torch.no_grad()
def evaluate(model,data,indices,out,step,lpips,scale=2):
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    out.mkdir(parents=True,exist_ok=True);results=[]
    for i in indices:
        row=data[i]['row'];gt=data[i]['rgb'];mask=data[i]['mask'];h,w=gt.shape[:2]
        yy,xx=torch.meshgrid(torch.arange(h,device='cuda')*scale,torch.arange(w,device='cuda')*scale,indexing='ij')
        origins,dirs,_=rays(row,yy.flatten(),xx.flatten());parts=[];valid_parts=[]
        for j in range(0,len(origins),8192):
            result=model(origins[j:j+8192],dirs[j:j+8192])
            parts.append(result['rgb']);valid_parts.append(result['valid'])
        pred=torch.cat(parts).reshape(h,w,3).clamp(0,1)
        supported=torch.cat(valid_parts).reshape(h,w)
        mse=(pred[mask]-gt[mask]).square().mean()
        a=torch.where(mask[...,None],pred,0).permute(2,0,1)[None];b=torch.where(mask[...,None],gt,0).permute(2,0,1)[None]
        result=dict(camera=i,psnr=float(-10*torch.log10(mse.clamp_min(1e-12))),ssim=float(ssim(a,b,data_range=1.)),lpips=float(lpips(a,b)),
                    background_ray_coverage=float(supported[mask].float().mean()))
        results.append(result)
        panel=torch.cat([gt,pred,torch.where(mask[...,None],(pred-gt).abs()*3,0)],1)
        Image.fromarray((panel.cpu().numpy().clip(0,1)*255).astype('uint8')).save(out/f'{i:04d}.jpg')
        if scale==1:
            for name,(x0,y0,x1,y1) in {'wall_top':(1440,0,1920,540),'wall_bottom':(1440,540,1920,1080)}.items():
                crop=torch.cat([gt[y0:y1,x0:x1],pred[y0:y1,x0:x1]],1)
                Image.fromarray((crop.cpu().numpy().clip(0,1)*255).astype('uint8')).save(out/f'{i:04d}_{name}.png')
    aggregate={k:float(np.mean([r[k] for r in results])) for k in ('psnr','ssim','lpips')}
    receipt=dict(step=step,metrics=aggregate,per_camera=results,domain='internal_train_background',
                 protocol=f'exact mask PSNR; zero-outside-mask SSIM/LPIPS; stride{scale}',scale=scale)
    write(out/'metrics.json',receipt);return receipt


def fit(bundle,output,kind,steps=12000,seconds=2400,refit=False,lr=.003,plane_lr=.00005):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(2)
    stage=f'background_{kind}'+('_refit' if refit else '')
    folder=output/stage;folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();geometry=read(output/'geometry.json')
    data=load_data(bundle,with_depth=bool(geometry.get('depth_guidance',False)))
    if kind=='field' and geometry.get('initialize_from_fitted_plane',False):
        fitted=torch.load(output/'background_plane/best.pt',map_location='cpu',weights_only=False)
        geometry['plane']=fitted['model']['plane'].tolist()
        corners=np.array(np.meshgrid(*zip(*geometry['actor_bounds']))).T.reshape(-1,3)
        n=np.array(geometry['plane'][:3]);off=geometry['plane'][3]
        geometry['behind_limit']=float((corners@n+off).min()-.03)
        geometry['density_plane_bias']=True
    if refit:
        state=torch.load(output/f'background_{kind}'/'best.pt',map_location='cpu',weights_only=False)
        gate_path=output/'visual_gate.json'
        if not gate_path.exists() or read(gate_path).get('accepted_background_checkpoint_sha256')!=sha(output/f'background_{kind}'/'best.pt'):
            raise ValueError('Refit requires a visual gate accepting this exact checkpoint')
        geometry=state['geometry']
    request=dict(kind=kind,geometry=geometry,steps=steps,seconds=seconds,
        lr=lr,plane_lr=plane_lr,refit_lr=.002 if refit else None,
        bundle=str(bundle.resolve()),input_hash_manifest_sha256=sha(bundle/'dataset_hashes.json'),
        script_sha256=sha(__file__),module_sha256=sha(Path(__file__).resolve().parents[2]/'nerfstudio/model_components/mesh_distillation.py'),
        refit=refit)
    if (folder/'complete.json').exists():
        if not (folder/'fit_request.json').exists() or read(folder/'fit_request.json')!=request:
            raise ValueError('Completed pilot belongs to a different request; use a new output directory')
        receipt=read(folder/'complete.json')
        if sha(folder/'best.pt')!=receipt['checkpoint_sha256']:
            raise ValueError('Completed checkpoint hash mismatch')
        return
    if (folder/'fit_request.json').exists():
        raise ValueError('Interrupted pilot detected; preserve it and use a new output directory')
    write(folder/'fit_request.json',request)
    source=folder/'source';source.mkdir(exist_ok=True)
    for path in (Path(__file__),Path(__file__).resolve().parents[2]/'nerfstudio/model_components/mesh_distillation.py'):
        shutil.copyfile(path,source/path.name)
    model=SeparateBackground(kind,geometry).cuda()
    groups=[dict(params=[p for name,p in model.named_parameters() if name!='plane'],lr=lr)]
    if isinstance(model.plane,torch.nn.Parameter):groups.append(dict(params=[model.plane],lr=plane_lr))
    opt=torch.optim.Adam(groups,eps=1e-15)
    if refit:
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        for group in opt.param_groups:group['lr']=.002
    lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    train=[i for i in range(len(data)) if refit or i not in VALIDATION]
    eligible={i:torch.nonzero(data[i]['mask']) for i in train};history=[];candidates=[]
    depth_eligible={i:torch.nonzero(data[i]['mask']&(data[i]['depth']>0)&torch.isfinite(data[i]['depth'])) for i in train} if geometry.get('depth_guidance',False) else {}
    check(output,stage,0,started)
    for step in range(1,steps+1):
        # Eight images per update: background cannot specialize to one camera.
        os_,ds_,cs_,zs_=[] ,[],[],[]
        for i in np.random.choice(train,8,replace=False):
            yx=eligible[i][torch.randint(len(eligible[i]),(512,),device='cuda')]
            if i in depth_eligible and len(depth_eligible[i]):
                yx[-128:]=depth_eligible[i][torch.randint(len(depth_eligible[i]),(128,),device='cuda')]
            o,d,norm=rays(data[i]['row'],yx[:,0]*2,yx[:,1]*2)
            os_.append(o);ds_.append(d);cs_.append(data[i]['rgb'][yx[:,0],yx[:,1]])
            if depth_eligible:zs_.append(data[i]['depth'][yx[:,0],yx[:,1]]*norm[:,0])
        pred=model(torch.cat(os_),torch.cat(ds_));target=torch.cat(cs_)
        loss=weighted_charbonnier(pred['rgb'],target,pred['valid'].float())
        if zs_:
            depth=torch.cat(zs_);valid=(depth>0)&torch.isfinite(depth)&pred['valid'][:,0]
            agreement=torch.exp(-.5*((pred['distances']-depth[:,None])/.01).square())
            mass=(pred['weights']*agreement).sum(-1)
            loss=loss+float(geometry.get('depth_weight',.05))*(-mass[valid].clamp_min(1e-8).log()).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite background objective')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if isinstance(model.plane,torch.nn.Parameter):
            with torch.no_grad():
                initial=torch.tensor(geometry['plane'],device='cuda')
                model.plane[:3].copy_(torch.maximum(torch.minimum(model.plane[:3],initial[:3]+.04),initial[:3]-.04))
                model.plane[3].clamp_(float(initial[3])-.15,float(initial[3])+.15)
        if step%500==0:check(output,stage,step,started)
        stop=step==steps or time.monotonic()-started>=seconds
        if step%2000==0 or stop:
            result=evaluate(model,data,VALIDATION,folder/f'eval_{step:06d}',step,lpips)
            history.append(result);m=result['metrics'];print(stage,step,json.dumps(m),flush=True)
            maximum=max(r['metrics']['psnr'] for r in history)
            if m['psnr']>=maximum-.07 or refit:
                path=folder/f'candidate_{step:06d}.pt'
                torch.save(dict(kind=kind,geometry=geometry,model=model.state_dict(),optimizer=opt.state_dict(),step=step,
                    metrics=m,request_sha256=sha(output/'background_request.json'),
                    fit_request_sha256=sha(folder/'fit_request.json')),path)
                candidates.append(dict(path=path,**m))
            for candidate in candidates[:]:
                if (refit and candidate['path']!=path) or (not refit and candidate['psnr']<maximum-.07):
                    candidate['path'].unlink();candidates.remove(candidate)
            chosen=path if refit else min(candidates,key=lambda r:r['lpips'])['path']
            shutil.copyfile(chosen,folder/'best.pt')
            write(folder/'history.json',history)
        if stop:break
    check(output,stage,step,started,finished=True)
    write(folder/'complete.json',dict(stage=stage,step=step,seconds=time.monotonic()-started,
        checkpoint=str(folder/'best.pt'),checkpoint_sha256=sha(folder/'best.pt'),internal_validation_refitted=refit,
        visual_review_pending=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','plane','field','refit-plane','refit-field'])
    p.add_argument('--bundle',type=Path,default=BUNDLE);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--steps',type=int,default=12000);p.add_argument('--seconds',type=float,default=2400)
    p.add_argument('--lr',type=float,default=.003);p.add_argument('--plane-lr',type=float,default=.00005)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    try:
        if a.action=='prepare':prepare(a.bundle,a.output)
        else:fit(a.bundle,a.output,a.action.removeprefix('refit-'),a.steps,a.seconds,a.action.startswith('refit-'),a.lr,a.plane_lr)
    except BaseException as exc:
        write(a.output/'failure.json',dict(action=a.action,type=type(exc).__name__,error=str(exc),pid=os.getpid()))
        raise


if __name__=='__main__':main()
