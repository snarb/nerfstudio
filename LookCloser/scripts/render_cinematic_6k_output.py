"""True 3456x6144 cinematic output: new rays, native visibility and native RGB.

Only mesh-face graph labels are reused. HD output pixels and HD source-ID maps
are never used to construct the new output. Original source PNGs stay read-only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

BASE=Path('/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free')
OUT=Path('/mnt/data/dec5_cinematic_wide_spiral_6k_output_v1')
REMOTE='ubuntu@dev3'
REMOTE_ROOT='/fsx/tmp/lookcloser_cinematic_6k_output_v1'
REMOTE_PYTHON='/home/ubuntu/anaconda3/envs/nerfstudio/bin/python'
RAW='/fsx/oregon/projects/Dec5Shoots/workspace/DEC5_5A_3/subpix_out/DEC5_5A_3/working/fullres_pq16'
SCALE=np.array([5461/1920,3072/1080])
WIDTH,HEIGHT=6144,3456  # Calibrated landscape lattice; rotate once for delivery.


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(temp,path)
def call(args):subprocess.run([str(x) for x in args],check=True)


def scale_camera(camera,width,height):
    row=deepcopy(camera);sx=width/camera['w'];sy=height/camera['h']
    for key in ['fl_x','cx']:row[key]*=sx
    for key in ['fl_y','cy']:row[key]*=sy
    row.update(w=width,h=height);return row


def remote_sample(job):
    import cv2
    import convert_dec5_5a3_pq16_to_exr as decoder
    cv2.setNumThreads(1);spec=read(job/'job.json')
    with np.load(job/'uv.npz') as z:coordinates={k:z[k] for k in z.files}
    assert spec['coordinate_lattice']=='native_cropped_5461x3072'
    assert spec['frame'].isdigit() and len(spec['frame'])==6
    start=time.monotonic()
    def one(row):
        ci=row['index'];name=row['camera'];assert '/' not in name
        source=Path(RAW)/spec['frame']/'images'/f"{name}_{spec['frame']}.png"
        data=source.read_bytes();digest=hashlib.sha256(data).hexdigest()
        bgr=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
        assert bgr.shape==(3072,6144,3) and bgr.dtype==np.uint16
        coordinates_ci=coordinates[str(ci)];sampled=np.empty((len(coordinates_ci),3),np.float32)
        for begin in range(0,len(sampled),500000):
            end=min(begin+500000,len(sampled));uv=np.clip(coordinates_ci[begin:end].astype(np.float64),[0,0],[5460,3071])
            ix=np.floor(uv).astype(np.int64);frac=uv-ix;value=np.zeros((len(uv),3),np.float32)
            for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
                raw=bgr[np.minimum(ix[:,1]+dy,3071),341+np.minimum(ix[:,0]+dx,5460),::-1]
                linear=decoder.pq_decode_array(raw.astype(np.float32)/np.float32(65535))/np.float32(decoder.GAIN_TO_NITS)
                linear[raw==decoder.FLOOR_U16]=0;linear=linear@decoder.AP1_TO_REC709.astype(np.float32).T
                weight=(frac[:,0] if dx else 1-frac[:,0])*(frac[:,1] if dy else 1-frac[:,1])
                value+=linear*weight[:,None]
            sampled[begin:end]=value
        assert np.isfinite(sampled).all()
        return ci,sampled,dict(index=ci,camera=name,path=str(source),sha256=digest,
            bytes=len(data),dimensions=[6144,3072],selected_samples=len(sampled))
    result={};provenance=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ci,rgb,source in pool.map(one,spec['sources']):result[str(ci)]=rgb;provenance.append(source)
    np.savez(job/'rgb.npz',**result)
    write(job/'result.json',dict(frame=spec['frame'],sources=provenance,uv_sha256=sha(job/'uv.npz'),
        rgb_sha256=sha(job/'rgb.npz'),worker_sha256=sha(__file__),
        decoder_sha256=sha(Path(__file__).with_name('convert_dec5_5a3_pq16_to_exr.py')),
        seconds=time.monotonic()-start,source_crop=[341,0,5802,3072],resize=False,
        coordinate_lattice=spec['coordinate_lattice']))


def initialize():
    from joint_temporal_texture import ROOT as COLOR,CALIBRATION
    q=read(BASE/'request.json');assert not q['recipe']['static_registration']
    assert sha(COLOR/'parameters.npz')==q['profiles_sha256']
    assert sha(COLOR/'exposure.json')==q['exposure_sha256'] and sha(CALIBRATION)==q['calibration_sha256']
    dependencies=['joint_temporal_texture.py','native_texture_footprint.py','bake_joint_temporal_mesh.py',
        'diffusion_mesh_repair.py','render_patchmatch_camera_path.py','temporal_texture_view_prior.py']
    hashes={name:sha(Path(__file__).with_name(name)) for name in dependencies}
    for name,digest in hashes.items():assert digest==q['script_hashes'][name]
    request=dict(parent=str(BASE),parent_request_sha256=sha(BASE/'request.json'),
        script_sha256=sha(__file__),decoder_sha256=sha(Path(__file__).with_name('convert_dec5_5a3_pq16_to_exr.py')),
        dependencies=hashes,target_landscape_dimensions=[WIDTH,HEIGHT],output_dimensions=[HEIGHT,WIDTH],
        target_intrinsics_scale=[WIDTH/1920,HEIGHT/1080],source_visibility_dimensions=[5461,3072],
        source_scale_xy=SCALE.tolist(),source_root=RAW,fps=24,frame_count=150,
        fixed_profiles_sha256=q['profiles_sha256'],fixed_exposure_sha256=q['exposure_sha256'],
        calibration_sha256=q['calibration_sha256'],mesh_face_labels_reused=True,
        output_pixel_source_ids_recomputed=True,hd_pixels_upsampled=False,geometry_changed=False,
        hd_foreground_masks_mapped_to_native_pixel_centers=True,heldout_rgb_used=False,
        native_pixel_batch_size=60000,remote_sample_batch_size=500000,
        inventory=[dict(frame_id=r['frame_id'],index=r['index'],camera=scale_camera(r['camera'],WIDTH,HEIGHT),
            mesh_sha256=r['mesh_sha256']) for r in q['inventory']])
    OUT.mkdir(exist_ok=True)
    if (OUT/'request.json').exists():assert read(OUT/'request.json')==request
    else:write(OUT/'request.json',request)
    call(['ssh',REMOTE,'mkdir','-p',REMOTE_ROOT])
    call(['scp','-q',__file__,Path(__file__).with_name('convert_dec5_5a3_pq16_to_exr.py'),f'{REMOTE}:{REMOTE_ROOT}/'])
    return q


def native_samples(frame,rows,coordinates,folder):
    from joint_temporal_texture import HELD_CAMERAS
    job=OUT/'scratch'/frame;job.mkdir(parents=True,exist_ok=False)
    sources=[dict(index=int(ci),camera=rows[int(ci)]['physical_camera']) for ci in coordinates]
    assert not set(r['camera'] for r in sources)&HELD_CAMERAS
    write(job/'job.json',dict(frame=frame,sources=sources,coordinate_lattice='native_cropped_5461x3072'))
    np.savez(job/'uv.npz',**{str(k):v for k,v in coordinates.items()})
    remote=f'{REMOTE_ROOT}/{frame}';call(['ssh',REMOTE,'mkdir','-p',remote])
    call(['scp','-q',job/'job.json',job/'uv.npz',f'{REMOTE}:{remote}/'])
    call(['ssh',REMOTE,f'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 {REMOTE_PYTHON} {REMOTE_ROOT}/render_cinematic_6k_output.py remote --job {remote}'])
    call(['scp','-q',f'{REMOTE}:{remote}/rgb.npz',f'{REMOTE}:{remote}/result.json',str(job)])
    receipt=read(job/'result.json');config=read(OUT/'request.json')
    assert receipt['uv_sha256']==sha(job/'uv.npz') and receipt['rgb_sha256']==sha(job/'rgb.npz')
    assert receipt['worker_sha256']==config['script_sha256'] and receipt['decoder_sha256']==config['decoder_sha256']
    with np.load(job/'rgb.npz') as z:result={int(k):z[k] for k in z.files}
    write(folder/'source_provenance.json',receipt)
    call(['ssh',REMOTE,'rm','-f',f'{remote}/uv.npz',f'{remote}/rgb.npz',f'{remote}/job.json',f'{remote}/result.json'])
    call(['ssh',REMOTE,'rmdir',remote]);shutil.rmtree(job)
    return result


def source_choices(record,rows,folder,status):
    import open3d as o3d
    import torch
    from PIL import Image
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    from temporal_texture_view_prior import angle_weights
    frame=record['frame_id'];baseline=BASE/'frames'/frame;receipt=read(baseline/'complete.json')
    assert receipt['request_sha256']==sha(BASE/'request.json')
    assert sha(baseline/'face_source_labels.npy')==receipt['hashes']['face_source_labels.npy']
    for key in ['mesh','metadata']:assert sha(record[key])==record[key+'_sha256']
    mesh=o3d.io.read_triangle_mesh(record['mesh']);vertices=np.asarray(mesh.vertices,np.float32)
    triangles=np.asarray(mesh.triangles,np.uint32);tv=vertices[triangles]
    normals=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None].clip(1e-12)
    labels=np.load(baseline/'face_source_labels.npy');assert len(labels)==len(triangles)
    scene=scene_for(vertices,triangles);native_rows=[scale_camera(r,5461,3072) for r in rows]
    maskspec=record['source_masks'];maskroot=Path(maskspec['root'])
    for name,key in [('masks.npz','masks_sha256'),('cameras.json','cameras_sha256'),('complete.json','complete_sha256')]:
        assert sha(maskroot/name)==maskspec[key]
    masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json');lookup=dict(zip(names,masks))
    xhd=np.minimum(((np.arange(5461)+.5)/SCALE[0]).astype(int),1919)
    yhd=np.minimum(((np.arange(3072)+.5)/SCALE[1]).astype(int),1079)
    depth=torch.empty((len(rows),3072,5461),device='cuda',dtype=torch.float32)
    status('native_source_visibility')
    def cast(ci):
        row=native_rows[ci];d,_,_=camera_depth(scene,row)
        valid=lookup[row['physical_camera']][yhd[:,None],xhd[None,:]]>0
        return ci,np.where(np.isfinite(d)&valid,d,0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ci,d in pool.map(cast,range(len(rows))):depth[ci]=torch.from_numpy(d).cuda()
    status('full_6k_target_raycast');camera=scale_camera(record['camera'],WIDTH,HEIGHT)
    d,face,bary=camera_depth(scene,camera);hit=np.isfinite(d)
    assert hit.mean()>.01;np.savez_compressed(folder/'target_depth.npz',depth=np.where(hit,d,0))
    pixels=np.flatnonzero(hit);faces=face.ravel()[pixels];bary=bary.reshape(-1,2)[pixels]
    selected=np.full(WIDTH*HEIGHT,255,np.uint8);chosen_uv=np.empty((WIDTH*HEIGHT,2),np.float32)
    poses=torch.tensor(np.array([r['transform_matrix'] for r in rows],np.float32),device='cuda')
    centers=poses[:,:3,3];rotation=poses[:,:3,:3]
    intr=torch.tensor([[r[k] for k in ['fl_x','fl_y','cx','cy']] for r in native_rows],device='cuda',dtype=torch.float32)
    angular=torch.tensor(angle_weights(rows,record['camera'],4.)[0],device='cuda')
    flat=depth.flatten(1);source_indices=torch.arange(len(rows),device='cuda')[:,None]
    scalet=torch.tensor(SCALE,device='cuda',dtype=torch.float32)
    fallback_count=0;status('full_6k_pixel_source_selection')
    with torch.inference_mode():
        for start in range(0,len(pixels),60000):
            end=min(start+60000,len(pixels));f=faces[start:end];b=bary[start:end]
            weights=np.column_stack((1-b.sum(1),b));points=torch.tensor((tv[f]*weights[:,:,None]).sum(1),device='cuda')
            local=torch.einsum('cnk,ckj->cnj',points[None]-centers[:,None],rotation)
            z=-local[...,2];safe=torch.where(z.abs()>1e-8,z,torch.ones_like(z))
            uv=torch.stack((intr[:,0,None]*local[...,0]/safe+intr[:,2,None]-.5,
                -intr[:,1,None]*local[...,1]/safe+intr[:,3,None]-.5),-1)
            nearest=uv.round();uv=torch.where((uv-nearest).abs()<=.001,nearest,uv)
            base=uv.floor();frac=uv-base;center_depth=torch.zeros_like(z);tap_valid=torch.ones_like(z,dtype=torch.bool)
            for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
                x=(base[...,0].long()+dx).clamp(0,5460);y=(base[...,1].long()+dy).clamp(0,3071)
                tap=torch.gather(flat,1,y*5461+x)
                weight=(frac[...,0] if dx else 1-frac[...,0])*(frac[...,1] if dy else 1-frac[...,1])
                center_depth+=tap*weight
                tap_valid&=((tap>0)&((tap-z).abs()<.003*z))|(weight==0)
            hd=(uv+.5)/scalet-.5
            valid=(z>0)&(center_depth>0)&((center_depth-z).abs()<.0015*z)&tap_valid
            valid&=(hd[...,0]>2)&(hd[...,0]<1917)&(hd[...,1]>2)&(hd[...,1]<1077)
            direction=centers[:,None]-points;distance=torch.linalg.vector_norm(direction,dim=-1)
            direction/=distance[:,:,None]
            normal=torch.tensor(normals[f],device='cuda')
            quality=(direction*normal).sum(-1).abs().square()/distance.clamp_min(.01).square()*angular[:,None]*valid
            preferred=torch.tensor(labels[f].astype(np.int64),device='cuda');index=torch.arange(len(f),device='cuda')
            clipped=preferred.clamp(0,len(rows)-1)
            good=(preferred>=0)&(preferred<len(rows))&(quality[clipped,index]>0)
            best=quality.argmax(0);choice=torch.where(good,preferred,best);supported=quality.max(0).values>0
            coords=uv[choice,index].cpu().numpy();ids=torch.where(supported,choice,255).cpu().numpy().astype(np.uint8)
            selected[pixels[start:end]]=ids;chosen_uv[pixels[start:end]]=coords
            fallback_count+=int((supported&~good).sum())
    Image.fromarray(selected.reshape(HEIGHT,WIDTH)).save(folder/'source_ids.png')
    shutil.copyfile(baseline/'face_source_labels.npy',folder/'face_source_labels.npy')
    coordinates={};locations={}
    for ci in np.unique(selected[selected!=255]):
        where=np.flatnonzero(selected==ci);locations[int(ci)]=where;coordinates[int(ci)]=chosen_uv[where]
    del depth,flat;torch.cuda.empty_cache()
    return coordinates,locations,dict(mesh_hit_fraction=float(hit.mean()),source_supported_fraction=float((selected!=255).mean()),
        fallback_pixels=fallback_count,source_ids_recomputed_at_6k=True,face_labels_sha256=sha(folder/'face_source_labels.npy'))


def render_frame(q,record):
    import torch
    from joint_temporal_texture import cameras,ROOT as COLOR,display
    from compose_cinematic_train_ending import dissolve_alpha,blend_display,write_png
    torch.set_num_threads(2);start=time.monotonic();frame=record['frame_id'];index=record['index']
    folder=OUT/'frames'/frame;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        receipt=read(folder/'complete.json');assert receipt['request_sha256']==sha(OUT/'request.json')
        for name,h in receipt['hashes'].items():assert sha(folder/name)==h
        return
    def status(stage):write(OUT/'progress.json',dict(frame=frame,index=index,pid=os.getpid(),stage=stage,seconds=time.monotonic()-start))
    rows,_,_=cameras(frame);camera=scale_camera(record['camera'],WIDTH,HEIGHT);alpha=dissolve_alpha(index)
    profile=np.load(COLOR/'parameters.npz')['log_gain'];gains=np.exp(profile-profile.mean(0,keepdims=True))
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain'];coordinates={};locations={};checks={}
    if alpha<1:coordinates,locations,checks=source_choices(record,rows,folder,status)
    ending_ci=None;offset=0
    if alpha>0:
        ending_ci=next(i for i,r in enumerate(rows) if r['physical_camera']==q['camera_path_report']['endpoint_train_camera'])
        source=scale_camera(rows[ending_ci],5461,3072);uv=np.empty((HEIGHT,WIDTH,2),np.float32)
        uv[...,0]=(np.arange(WIDTH)[None]+.5-camera['cx'])/camera['fl_x']*source['fl_x']+source['cx']-.5
        uv[...,1]=(np.arange(HEIGHT)[:,None]+.5-camera['cy'])/camera['fl_y']*source['fl_y']+source['cy']-.5
        offset=len(coordinates.get(ending_ci,[]));coordinates[ending_ci]=np.concatenate((coordinates.get(ending_ci,np.empty((0,2),np.float32)),uv.reshape(-1,2)))
    status('native_6k_rgb_sampling');samples=native_samples(frame,rows,coordinates,folder)
    raw=None;train=None
    if alpha<1:
        raw=np.zeros((WIDTH*HEIGHT,3),np.uint8)
        for ci,where in locations.items():
            for begin in range(0,len(where),500000):
                end=min(begin+500000,len(where));value=display(samples[ci][begin:end]*gains[ci],exposure)
                raw[where[begin:end]]=np.rint(value*255).clip(0,255).astype(np.uint8)
        raw=np.rot90(raw.reshape(HEIGHT,WIDTH,3));write_png(folder/'render.png',raw)
    if alpha>0:
        train=np.empty((WIDTH*HEIGHT,3),np.uint8);value=samples[ending_ci][offset:]
        for begin in range(0,len(value),500000):
            end=min(begin+500000,len(value));train[begin:end]=np.rint(display(value[begin:end]*gains[ending_ci],exposure)*255).clip(0,255).astype(np.uint8)
        train=np.rot90(train.reshape(HEIGHT,WIDTH,3));write_png(folder/'train.png',train)
    image=raw if alpha==0 else train if alpha==1 else blend_display(raw,train,alpha)
    assert image.shape==(6144,3456,3);write_png(folder/'frame.png',image)
    write(folder/'result.json',dict(frame_id=frame,index=index,camera=camera,original_camera=record['camera'],
        output_dimensions=[3456,6144],train_alpha=alpha,kind='3d_render' if alpha==0 else 'real_train_rgb' if alpha==1 else 'explicit_3d_to_train_dissolve',
        checks=checks,fixed_exposure=exposure,seconds=time.monotonic()-start,hd_pixels_upsampled=False,visual_status='pending'))
    write(folder/'complete.json',dict(request_sha256=sha(OUT/'request.json'),hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='complete.json'}))
    print(f'frame={frame} index={index} seconds={time.monotonic()-start:.1f} output=3456x6144 sources={len(samples)}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['render','remote']);p.add_argument('--job',type=Path);p.add_argument('--frames',nargs='+');a=p.parse_args()
    if a.action=='remote':remote_sample(a.job)
    else:
        q=initialize()
        for record in q['inventory']:
            if a.frames is None or record['frame_id'] in a.frames:render_frame(q,record)
        write(OUT/'progress.json',dict(stage='requested_frames_finished',complete=len(list((OUT/'frames').glob('*/complete.json')))))
