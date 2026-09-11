"""Fixed-geometry, multi-time camera response and native texture registration.

One RGB response per physical camera is shared across times. Mesh-space patches
define common correspondence coordinates; bounded native-image warp grids are
optimized against robust common patches. No eval image enters these fits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HELD_CAMERAS = {'F004_B005_1210O9', 'J004_D005_1210TA', 'L004_B005_12106A'}
CAMERA_FIELDS = ('transform_matrix','fl_x','fl_y','cx','cy','w','h','k1','k2','p1','p2','camera_model')
ROOT = Path('/mnt/data/lookcloser_dec5_5a3_joint_texture')
SOURCE = Path('/mnt/data/dec5_5a3_nerfstudio_exr_1920x1080')
GEOMETRY = Path('/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/frames')
CALIBRATION = Path('/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/config/calibration/transforms.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20), b''): h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temp,path)


def read(path):return json.loads(Path(path).read_text())


def display(x, gain):
    x=np.maximum(x,0)*gain;x=x/(1+x)
    return np.where(x<=.0031308,12.92*x,1.055*np.power(x,1/2.4)-.055)


def save_png(path, x):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8)).save(path)


def exr(path):
    x=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if x is None or x.shape!=(1080,1920,3) or not np.isfinite(x).all():
        raise ValueError(f'Invalid native EXR: {path}')
    return np.ascontiguousarray(x[...,::-1])


def geometry_paths(frame):
    # The earlier full-block correction removed a diagnosed discretization shard.
    if frame=='000973':
        base=Path('/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/full_block_tsdf')
        return base/'mesh.ply',base/'mesh.json'
    base=GEOMETRY/frame/'mesh'
    return base/'colmap_patchmatch_tsdf.ply',base/'colmap_patchmatch_tsdf.json'


def cameras(frame, calibration=CALIBRATION):
    from render_patchmatch_camera_path import normalize_frame
    original=read(SOURCE/frame/'transforms.json');cal=read(calibration)
    by_name={f['physical_camera']:f for f in cal['frames']}
    if len(by_name)!=len(cal['frames']):raise ValueError('Duplicate calibration cameras')
    mesh,meta=geometry_paths(frame);metadata=read(meta)
    rows=[]
    for source in original['frames']:
        if 'frame_train_' not in Path(source['file_path']).name:continue
        name=source['physical_camera']
        if name in HELD_CAMERAS:raise ValueError('Held-out source leak')
        row=deepcopy(source)
        for key in CAMERA_FIELDS:row[key]=deepcopy(by_name[name][key])
        row=normalize_frame(row,cal,metadata)
        row['file_path']=str((SOURCE/frame/source['file_path']).resolve(strict=True))
        rows.append(row)
    rows.sort(key=lambda f:f['physical_camera'])
    if len(rows)!=62 or len({f['physical_camera'] for f in rows})!=62:raise ValueError('Expected 62 unique train cameras')
    return rows,mesh,meta


def project(points, rows):
    poses=np.array([r['transform_matrix'] for r in rows],np.float32)
    q=np.einsum('cnk,ckj->cnj',points[None]-poses[:,None,:3,3],poses[:,:3,:3])
    z=-q[...,2];fx=np.array([r['fl_x'] for r in rows])[:,None];fy=np.array([r['fl_y'] for r in rows])[:,None]
    cx=np.array([r['cx'] for r in rows])[:,None];cy=np.array([r['cy'] for r in rows])[:,None]
    safe=np.where(np.abs(z)>1e-8,z,1)
    uv=np.stack((fx*q[...,0]/safe+cx-.5,-fy*q[...,1]/safe+cy-.5),-1)
    return uv.astype(np.float32),z.astype(np.float32)


def normalized_uv(uv,w=1920,h=1080):
    return (uv+.5)*uv.new_tensor([2/w,2/h])-1


def sample(images, uv):
    return F.grid_sample(images,normalized_uv(uv),align_corners=False,padding_mode='border')


def apply_response(rgb, log_gain):
    centered=log_gain-log_gain.mean(0,keepdim=True)
    return rgb*centered.exp()[:,:,None,None]


def bounded_warp(static, residual, uv, limit=2.):
    """Native displacement bounded by ``limit`` PER AXIS (norm <= sqrt(2)*limit)."""
    grid=limit*torch.tanh(static+residual)
    return F.grid_sample(grid,normalized_uv(uv),align_corners=False,padding_mode='border').permute(0,2,3,1)


def robust_center(colors, weight, iterations=3):
    """One location per corresponding point; no smoothing between patch pixels."""
    w=weight[:,None]
    center=(colors*w).sum(0)/w.sum(0).clamp_min(1e-8)
    for _ in range(iterations):
        error=(colors-center[None]).square().mean(1).sqrt()
        rw=w/(1+(error[:,None]/.08).square())
        center=(colors*rw).sum(0)/rw.sum(0).clamp_min(1e-8)
    return center


def patch_inventory(mesh, count=6000, seed=17):
    v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles)
    a,b,c=v[t[:,0]],v[t[:,1]],v[t[:,2]]
    n=np.cross(b-a,c-a);area=np.linalg.norm(n,axis=1)
    rng=np.random.default_rng(seed);ids=rng.choice(len(t),count,replace=True,p=area/area.sum())
    q=rng.uniform(size=(count,2));q[q.sum(1)>1]=1-q[q.sum(1)>1]
    center=a[ids]+q[:,:1]*(b-a)[ids]+q[:,1:]*(c-a)[ids]
    normal=n[ids]/area[ids,None].clip(1e-12)
    tangent=(b-a)[ids];tangent/=np.linalg.norm(tangent,axis=1)[:,None].clip(1e-12)
    other=np.cross(normal,tangent)
    offsets=np.array([(x,y) for y in (-1,0,1) for x in (-1,0,1)],np.float32)
    points=center[:,None]+.00012*(tangent[:,None]*offsets[None,:,0:1]+other[:,None]*offsets[None,:,1:2])
    blocks=np.floor(center/.008).astype(np.int64)
    held=((blocks[:,0]*73856093)^(blocks[:,1]*19349663)^(blocks[:,2]*83492791))%5==0
    return points,normal,held,ids


def prepare_frame(root, frame, count):
    import open3d as o3d
    rows,meshpath,metapath=cameras(frame)
    dest=root/'cache'/frame;dest.mkdir(parents=True,exist_ok=True)
    request={'frame':frame,'mesh':str(meshpath),'mesh_sha256':sha(meshpath),'metadata_sha256':sha(metapath),
             'source_transforms_sha256':sha(SOURCE/frame/'transforms.json'),
             'calibration_sha256':sha(CALIBRATION),'patches':count,'cameras':rows,
             'script_sha256':sha(__file__)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        request['source_hashes']=dict(zip([r['physical_camera'] for r in rows],pool.map(sha,[r['file_path'] for r in rows])))
    if (dest/'request.json').exists():
        if read(dest/'request.json')!=request:raise ValueError('Prepared input changed; use a new output root')
        if (dest/'complete.json').exists():
            for f,h in read(dest/'complete.json')['hashes'].items():
                if sha(dest/f)!=h:raise ValueError('Cache hash mismatch')
            print(f'prepare frame={frame} reused',flush=True);return
    atomic_json(dest/'request.json',request)
    mesh=o3d.io.read_triangle_mesh(str(meshpath));scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    points,normals,held,ids=patch_inventory(mesh,count)
    uv,z=project(points.reshape(-1,3),rows);uv=uv.reshape(62,count,9,2);z=z.reshape(62,count,9)
    valid=[];angles=[];medians=[]
    for i,row in enumerate(rows):
        rgb=exr(row['file_path']);np.save(dest/f'rgb_{i:02d}.npy',rgb)
        medians.append(float(np.median(np.maximum(rgb[::16,::16],0)@np.array([.2126,.7152,.0722]))))
        pose=np.asarray(row['transform_matrix']);center=pose[:3,3].astype(np.float32)
        directions=points.reshape(-1,3)-center
        rays=np.concatenate([np.broadcast_to(center,directions.shape),directions],1).astype(np.float32)
        hit=scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        distance=np.abs(hit-1)*np.linalg.norm(directions,axis=1)
        good=(np.isfinite(hit)&(distance<.00007)).reshape(count,9)
        good&=(uv[i,...,0]>6)&(uv[i,...,0]<1913)&(uv[i,...,1]>6)&(uv[i,...,1]<1073)&(z[i]>0)
        good=good.all(1)
        view=center-points[:,4];view/=np.linalg.norm(view,axis=1)[:,None]
        angle=np.abs((view*normals).sum(1));good&=angle>.2
        valid.append(good);angles.append(angle)
        if i%16==0:print(f'prepare frame={frame} camera={i+1}/62 visible_patches={int(good.sum())}',flush=True)
    np.savez_compressed(dest/'observations.npz',uv=uv,z=z,valid=np.array(valid),angle=np.array(angles),
                        points=points,normals=normals,held=held,triangle_ids=ids)
    atomic_json(dest/'statistics.json',{'camera_median_luminance':medians,'held_patches':int(held.sum())})
    atomic_json(dest/'complete.json',{'hashes':{p.name:sha(p) for p in dest.iterdir() if p.name!='complete.json'}})
    print(f'prepare frame={frame} complete',flush=True)


def load_frame(root,frame,device='cuda'):
    dest=root/'cache'/frame
    validate_cache(dest)
    arrays=np.load(dest/'observations.npz')
    obs={k:torch.as_tensor(arrays[k],device=device) for k in arrays.files}
    # CPU file reads overlap and all native RGB is retained in linear float32.
    with ThreadPoolExecutor(max_workers=8) as pool:
        rgb=list(pool.map(np.load,[dest/f'rgb_{i:02d}.npy' for i in range(62)]))
    obs['images']=torch.as_tensor(np.stack(rgb).transpose(0,3,1,2),device=device)
    obs['rows']=read(dest/'request.json')['cameras']
    return obs


def validate_cache(dest):
    complete=read(dest/'complete.json')
    for name,digest in complete['hashes'].items():
        if Path(name).name!=name:raise ValueError('Invalid cache member')
        if sha(dest/name)!=digest:raise ValueError(f'Cache checksum mismatch: {name}')
    expected={'request.json','statistics.json','observations.npz',*[f'rgb_{i:02d}.npy' for i in range(62)]}
    if set(complete['hashes'])!=expected:raise ValueError('Incomplete prepared frame')
    request=read(dest/'request.json')
    names=[r['physical_camera'] for r in request['cameras']]
    if len(set(names))!=62 or len(names)!=62 or names!=sorted(names) or set(names)&HELD_CAMERAS:
        raise ValueError('Invalid camera inventory')


def load_parameters(root,frame,device='cuda'):
    """Frozen profiles with either fitted-time or separately adapted UV residuals."""
    result=read(root/'fit_result.json')
    if sha(root/'parameters.npz')!=result['parameters_sha256']:raise ValueError('Model checksum mismatch')
    if sha(root/'fit_request.json')!=result['request_sha256']:raise ValueError('Fit request checksum mismatch')
    profiles=read(root/'camera_profiles.json')
    if sha(root/'exposure.json')!=profiles['exposure_sha256']:raise ValueError('Frozen exposure checksum mismatch')
    pars=np.load(root/'parameters.npz');key=f'residual_{frame}'
    gains=np.exp(pars['log_gain']-pars['log_gain'].mean(0))
    if not np.allclose(gains,profiles['rgb_gain'],rtol=1e-6,atol=1e-7):raise ValueError('Camera profile mismatch')
    if key in pars:
        residual=pars[key]
    else:
        dest=root/'adaptations'/frame;record=read(dest/'result.json')
        request=read(dest/'request.json')
        if request['parameters_sha256']!=sha(root/'parameters.npz') or request['exposure_sha256']!=sha(root/'exposure.json'):
            raise ValueError('Adaptation belongs to another frozen calibration')
        if sha(dest/'residual.npz')!=record['residual_sha256']:raise ValueError('Adaptation checksum mismatch')
        residual=np.load(dest/'residual.npz')['residual']
    return tuple(torch.tensor(x,device=device) for x in [pars['log_gain'],pars['static_warp'],residual])


def adapt(root,frame,iterations=60):
    """Add a new time without re-estimating any camera color/exposure parameter."""
    load_parameters(root,read(root/'fit_request.json')['fit_frames'][0],device='cpu')
    dest=root/'adaptations'/frame;dest.mkdir(parents=True,exist_ok=True)
    request={'frame':frame,'iterations':iterations,'parameters_sha256':sha(root/'parameters.npz'),
             'exposure_sha256':sha(root/'exposure.json'),'input_sha256':sha(root/'cache'/frame/'request.json'),
             'script_sha256':sha(__file__),'uses_eval_rgb':False,'maximum_native_shift_per_axis_px':2.}
    if (dest/'request.json').exists() and read(dest/'request.json')!=request:raise ValueError('Adaptation request changed')
    atomic_json(dest/'request.json',request)
    if (dest/'result.json').exists():
        load_parameters(root,frame);return
    data=load_frame(root,frame);pars=np.load(root/'parameters.npz')
    profile=torch.tensor(pars['log_gain'],device='cuda');static=torch.tensor(pars['static_warp'],device='cuda')
    residual=torch.nn.Parameter(torch.zeros_like(static));opt=torch.optim.Adam([residual],lr=.012)
    torch.manual_seed(13)
    for step in range(iterations):
        opt.zero_grad();eligible=torch.where(~data['held'])[0]
        ids=eligible[torch.randperm(len(eligible),device='cuda')[:2500]]
        rgb,w,_=observe(data,profile,static,residual,ids)
        with torch.no_grad():target=robust_center(rgb,w)
        err=rgb-target[None]
        loss=(torch.sqrt(err.square()+.03**2)*w[:,None]).sum()/(w.sum()*3).clamp_min(1)+residual.square().mean()*.02
        loss.backward();opt.step()
    np.savez_compressed(dest/'residual.npz',residual=residual.detach().cpu().numpy())
    for path,key in [('parameters.npz','parameters_sha256'),('exposure.json','exposure_sha256')]:
        if sha(root/path)!=request[key]:raise ValueError('Frozen calibration changed during adaptation')
    atomic_json(dest/'result.json',{'status':'adapted','residual_sha256':sha(dest/'residual.npz'),
                'request_sha256':sha(dest/'request.json'),'camera_profiles_changed':False,'geometry_changed':False})


def observe(data, log_gain, static, residual, indices=None, correct=True, warp=True):
    uv=data['uv'] if indices is None else data['uv'][:,indices]
    shift=bounded_warp(static,residual,uv) if warp else torch.zeros_like(uv)
    rgb=sample(data['images'],uv+shift).clamp_min(1e-6)
    if correct:rgb=apply_response(rgb,log_gain)
    # Log-linear radiance is robust to the wide HDR range and fixes a gain gauge.
    color=rgb.log()
    valid=data['valid'] if indices is None else data['valid'][:,indices]
    angle=data['angle'] if indices is None else data['angle'][:,indices]
    # Eligibility independent of the fitted response/warp.
    base=sample(data['images'],uv).clamp_min(0)
    eligible=(base>.0002).all(1).all(-1)&(base<10).all(1).all(-1)
    weight=(valid&eligible).float()*angle.square()
    return color,weight[...,None].expand(-1,-1,9),shift


def fit(root,fit_frames,held_frames,iterations=160):
    torch.manual_seed(13);torch.set_num_threads(8)
    params_path=root/'fit_request.json'
    request={'fit_frames':fit_frames,'held_frames':held_frames,'iterations':iterations,'grid':[12,20],
             'maximum_native_shift_per_axis_px':2.,'profile_shared_across_time':True,
             'geometry_fixed':True,'poses_fixed':True,'uses_eval_rgb':False,'script_sha256':sha(__file__),
             'input_hashes':{f:sha(root/'cache'/f/'request.json') for f in fit_frames+held_frames}}
    if params_path.exists():
        if read(params_path)!=request:raise ValueError('Fit configuration differs; use a new root')
        if (root/'fit_result.json').exists():
            load_parameters(root,fit_frames[0]);return
    atomic_json(params_path,request)
    all_data={f:load_frame(root,f) for f in fit_frames+held_frames}
    names=[r['physical_camera'] for r in all_data[fit_frames[0]]['rows']]
    if any([r['physical_camera'] for r in d['rows']]!=names for d in all_data.values()):raise ValueError('Camera identity mismatch')
    # Freeze one temporal-global display exposure, estimated ONCE from fit inputs only.
    lum=np.array([read(root/'cache'/f/'statistics.json')['camera_median_luminance'] for f in fit_frames])
    gain=float(.18/(np.exp(np.log(lum.clip(1e-8)).mean())*.82))
    atomic_json(root/'exposure.json',{'exposure_mode':'fixed','fixed_exposure_gain':gain,
             'fit_frames':fit_frames,'time_varying_gain':False,'curve':'linear_exposure_reinhard_srgb'})
    profile=torch.nn.Parameter(torch.zeros(62,3,device='cuda'))
    static=torch.nn.Parameter(torch.zeros(62,2,12,20,device='cuda'))
    residuals={f:torch.nn.Parameter(torch.zeros_like(static)) for f in all_data}
    opt=torch.optim.Adam([{'params':[profile],'lr':.008},{'params':[static,*[residuals[f] for f in fit_frames]],'lr':.012}])
    trace=[];start=time.monotonic()
    for step in range(iterations):
        opt.zero_grad();values=[]
        for f in fit_frames:
            d=all_data[f];eligible=torch.where(~d['held'])[0]
            ids=eligible[torch.randperm(len(eligible),device='cuda')[:2500]]
            rgb,w,shift=observe(d,profile,static,residuals[f],ids,warp=step>=20)
            with torch.no_grad():target=robust_center(rgb,w)
            error=rgb-target[None]
            photo=(torch.sqrt(error.square()+.03**2)*w[:,None]).sum()/(w.sum()*3).clamp_min(1)
            # Local tangent-patch contrast constrains shifts separately from gains.
            contrast=error-error[:,:,:,4:5]
            detail=(torch.sqrt(contrast.square()+.02**2)*w[:,None]).sum()/(w.sum()*3).clamp_min(1)
            reg=(static+residuals[f]).square().mean()*.008+residuals[f].square().mean()*.02
            grid=static+residuals[f]
            smooth=((grid[:,:,1:]-grid[:,:,:-1]).square().mean()+(grid[:,:,:,1:]-grid[:,:,:,:-1]).square().mean())*.08
            objective=(photo+detail*.8+reg+smooth)/len(fit_frames)
            objective.backward();values.append(float(photo.detach()))
        (profile.square().mean()*.02).backward();opt.step()
        with torch.no_grad():profile.sub_(profile.mean(0));profile.clamp_(-.4,.4)
        if step%10==0 or step==iterations-1:
            row={'iteration':step,'fit_patch_log_residual':values,'seconds':time.monotonic()-start}
            trace.append(row);atomic_json(root/'fit_progress.json',row)
            print(f'fit step={step}/{iterations} log_residual={np.mean(values):.6f} seconds={row["seconds"]:.1f}',flush=True)
    # Whole-time holdout: no response/static-grid updates. Only its own local UVs
    # can adapt using non-held spatial blocks, which are evaluated separately.
    for f in held_frames:
        opt_hold=torch.optim.Adam([residuals[f]],lr=.012)
        for step in range(60):
            opt_hold.zero_grad();d=all_data[f];eligible=torch.where(~d['held'])[0]
            ids=eligible[torch.randperm(len(eligible),device='cuda')[:2500]]
            rgb,w,_=observe(d,profile.detach(),static.detach(),residuals[f],ids)
            with torch.no_grad():target=robust_center(rgb,w)
            err=rgb-target[None];obj=(torch.sqrt(err.square()+.03**2)*w[:,None]).sum()/(w.sum()*3).clamp_min(1)
            obj=obj+residuals[f].square().mean()*.02
            obj.backward();opt_hold.step()
    with torch.no_grad():
        gains=(profile-profile.mean(0)).exp().cpu().numpy()
        arrays={'log_gain':profile.cpu().numpy(),'static_warp':static.cpu().numpy()}
        arrays.update({f'residual_{f}':r.cpu().numpy() for f,r in residuals.items()})
        np.savez_compressed(root/'parameters.npz',**arrays)
        validation={}
        for f,d in all_data.items():
            ids=torch.where(d['held'])[0];rows={}
            for variant,corr,warp in [('fixed_exposure',False,False),('camera_profile',True,False),('joint',True,True)]:
                rgb,w,shift=observe(d,profile,static,residuals[f],ids,correct=corr,warp=warp)
                target=robust_center(rgb,w)
                valid=w>0
                err=(rgb-target[None]).abs().mean(1)[valid]
                rows[variant]={'log_rgb_median':float(err.median()),'log_rgb_p90':float(torch.quantile(err,.9)),
                    'samples':int(valid.sum()),'max_uv_shift_px':float(shift.norm(dim=-1).max())}
            validation[f]=rows
    atomic_json(root/'camera_profiles.json',{'physical_cameras':names,'rgb_gain':gains.tolist(),
                'fixed_across_time':True,'fit_frames':fit_frames,'held_time_frames':held_frames,
                'uses_eval_rgb':False,'exposure_sha256':sha(root/'exposure.json')})
    atomic_json(root/'fit_trace.json',trace)
    atomic_json(root/'fit_result.json',{'status':'fitted_pending_native_review','validation':validation,
                'parameters_sha256':sha(root/'parameters.npz'),'request_sha256':sha(params_path),
                'camera_gain_min':gains.min(0).tolist(),'camera_gain_max':gains.max(0).tolist()})
    print(json.dumps(validation),flush=True)


def calibrate(root, fit_frames, held_frames, *, patches=6000, iterations=160, dry_run=False):
    """Prepare multiple head poses and fit one temporally frozen camera profile.

    Report: LookCloser/experiments/dec5_joint_temporal_texture.md
    The held times do not fit camera response/shared UVs; only their own local
    UV residuals may adapt. This helper does not bake, score or approve a mesh.
    Existing hash-pinned runs are never silently upgraded to different code.
    """
    root = Path(root).expanduser().resolve()
    fit_frames, held_frames = list(fit_frames), list(held_frames)
    frames = fit_frames + held_frames
    if len(fit_frames) < 2 or not held_frames:
        raise ValueError('Calibration needs at least two fit times and one held-out time')
    if any(not isinstance(f, str) or len(f) != 6 or not f.isascii() or not f.isdigit() for f in frames):
        raise ValueError('Frame IDs must be six-digit ASCII strings')
    if len(frames) != len(set(frames)):
        raise ValueError('Calibration times must be unique with no train/holdout overlap')
    if patches < 1 or iterations < 1:
        raise ValueError('Patches and iterations must be positive')
    if root.is_relative_to(SOURCE.resolve()):
        raise ValueError('Calibration output must not be inside the immutable source dataset')
    plan = {
        'status': 'planned', 'output': str(root), 'source': str(SOURCE),
        'calibration_template': str(CALIBRATION), 'fit_frames': fit_frames,
        'held_frames': held_frames, 'patches': patches, 'iterations': iterations,
        'stages': [{'stage': 'prepare', 'frame': f} for f in frames] + [{'stage': 'fit'}],
        'report': str(Path(__file__).resolve().parents[1] / 'experiments/dec5_joint_temporal_texture.md'),
        'exposure_fixed_across_time': True, 'camera_profiles_fixed_across_time': True,
        'uses_eval_rgb': False, 'changes_geometry': False,
        'artifacts': ['exposure.json', 'camera_profiles.json', 'parameters.npz', 'fit_result.json'],
    }
    # Preflight cheap request checks before touching any cache or loading images.
    request_path = root / 'fit_request.json'
    if request_path.exists():
        previous = read(request_path)
        expected = {'fit_frames': fit_frames, 'held_frames': held_frames,
                    'iterations': iterations, 'script_sha256': sha(__file__)}
        if any(previous.get(k) != value for k, value in expected.items()):
            raise ValueError('Existing calibration request differs; use a new output root or its recorded source snapshot')
    for frame in frames:
        cached = root / 'cache' / frame / 'request.json'
        if cached.exists():
            previous = read(cached)
            if previous.get('patches') != patches or previous.get('script_sha256') != sha(__file__):
                raise ValueError(f'Prepared frame {frame} has a different request; use a new output root')
    if dry_run:
        return plan
    root.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        prepare_frame(root, frame, patches)
    fit(root, fit_frames, held_frames, iterations)
    return {**plan, 'status': 'calibrated_not_visually_approved'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','fit','adapt','calibrate'])
    p.add_argument('--dry-run',action='store_true',help='Print the calibrate plan without writing files or loading images/GPU tensors')
    p.add_argument('--frames',nargs='+',help='Explicit new times for prepare/adapt; never changes fit-frame selection')
    p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--fit-frames',nargs='+',default=['000899','000973','001139','001197'])
    p.add_argument('--held-frames',nargs='+',default=['001059'])
    p.add_argument('--patches',type=int,default=6000)
    p.add_argument('--iterations',type=int,default=160)
    a=p.parse_args(argv)
    if a.dry_run and a.action!='calibrate':p.error('--dry-run is only valid with calibrate')
    if a.action=='calibrate':
        if a.frames:p.error('calibrate uses --fit-frames and --held-frames, not --frames')
        print(json.dumps(calibrate(a.output,a.fit_frames,a.held_frames,patches=a.patches,
                                  iterations=a.iterations,dry_run=a.dry_run),indent=2))
        return
    a.output.mkdir(parents=True,exist_ok=True)
    if set(a.fit_frames)&set(a.held_frames):raise ValueError('Temporal train/holdout overlap')
    if a.action=='prepare':
        for f in a.frames or a.fit_frames+a.held_frames:prepare_frame(a.output,f,a.patches)
    elif a.action=='adapt':
        if not a.frames:p.error('adapt requires --frames')
        for f in a.frames:adapt(a.output,f)
    else:
        if a.frames:p.error('--frames is only valid with prepare/adapt')
        fit(a.output,a.fit_frames,a.held_frames,a.iterations)


if __name__=='__main__':main()
