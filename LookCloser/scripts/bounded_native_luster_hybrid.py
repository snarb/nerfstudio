"""Native frozen-field inference, skipping rays outside the projected saved AABB.

Only a conservative projection bound is used, never GT or a person mask.
Empty exterior receives the same measured background; valid geometry is not
resized, fitted, filled or dilated.
"""
import itertools
from pathlib import Path
import time
import numpy as np
import torch
from luster_hybrid_common import (DIGEST,TARGETS,image,log,native_rows,project,read,sha,write)
from render_luster_hybrid import load,packed_quantiles,state_hashes


def projected_box(bounds,row):
    corners=np.array(list(itertools.product(*zip(bounds[0],bounds[1]))),np.float32)
    uv,z=project(corners,row)
    if not np.isfinite(uv).all() or np.any(z<=0):
        return [0,0,row['w'],row['h']]
    lo=np.floor(uv.min(0)).astype(int)-2
    hi=np.ceil(uv.max(0)).astype(int)+3
    return [int(np.clip(lo[0],0,row['w'])),int(np.clip(lo[1],0,row['h'])),
            int(np.clip(hi[0],0,row['w'])),int(np.clip(hi[1],0,row['h']))]


def run(args):
    root=args.root;cfg,model=load(root,args.chunk)
    from render_luster_quality import camera_for,measured_plate
    before=state_hashes(model);rows=native_rows()
    bounds=model.scene_box.aabb.detach().cpu().numpy()
    for cid in args.cameras:
        row=rows[cid];out=root/'geometry'/f'cam{cid:03d}'
        if (out/'receipt.json').exists():
            r=read(out/'receipt.json')
            assert r['checkpoint_sha256']==DIGEST and r['state_hash']==before
            assert all(sha(out/f'{k}.npy')==r['hashes'][k] for k in r['arrays'])
            log(root,'bounded_render_cached',camera=cid);continue
        out.mkdir(parents=True,exist_ok=True)
        h,w=row['h'],row['w'];box=projected_box(bounds,row);x0,y0,x1,y1=box
        arrays={k:np.lib.format.open_memmap(out/f'{k}.npy',mode='w+',dtype=np.float32,
                shape=(h,w,3) if k=='rgb' else (h,w)) for k in
                ['rgb','alpha','ray_distance_mean','depth_z_q10','depth_z_q50','depth_z_q90']}
        camera=camera_for(row,model.device);plate,pr=measured_plate(cfg,row)
        arrays['rgb'][:]=plate if plate is not None else 0.
        for k in ['alpha','ray_distance_mean']:arrays[k][:]=0.
        for k in ['depth_z_q10','depth_z_q50','depth_z_q90']:arrays[k][:]=np.nan
        bw=x1-x0;total=bw*(y1-y0);started=last=time.monotonic()
        with torch.inference_mode():
            for start in range(0,total,args.chunk):
                ids=torch.arange(start,min(start+args.chunk,total),device=model.device)
                yy=ids//bw+y0;xx=ids%bw+x0
                coords=torch.stack([yy,xx],-1).float()+.5
                rays=model.collider(camera.generate_rays(camera_indices=0,coords=coords))
                outputs=model.get_outputs(rays);q=packed_quantiles(outputs,rays)
                q*=(-(rays.directions@camera.camera_to_worlds[0,:3,:3])[:,2])[:,None]
                alpha=outputs['accumulation'].float().cpu().numpy().reshape(-1)
                rgb=outputs['rgb'].float().cpu().numpy()
                y=yy.cpu().numpy();x=xx.cpu().numpy()
                if plate is not None:rgb+=(1-alpha[:,None])*plate[y,x]
                arrays['rgb'][y,x]=rgb;arrays['alpha'][y,x]=alpha
                arrays['ray_distance_mean'][y,x]=outputs['depth'].float().cpu().numpy().reshape(-1)
                for i,k in enumerate((10,50,90)):arrays[f'depth_z_q{k}'][y,x]=q[:,i].cpu().numpy()
                if time.monotonic()-last>45:
                    log(root,'bounded_render_progress',camera=cid,pixels=start+len(ids),total=total,
                        elapsed=time.monotonic()-started,gpu_peak_gb=torch.cuda.max_memory_allocated()/1e9)
                    last=time.monotonic()
        for value in arrays.values():value.flush()
        assert state_hashes(model)==before
        if cid in TARGETS:image(root/'baseline'/f'cam{cid:03d}.png',arrays['rgb'])
        receipt=dict(state='complete',camera=cid,row=row,arrays=list(arrays),checkpoint_sha256=DIGEST,
            state_hash=before,code_sha256=sha(Path(__file__)),hashes={k:sha(out/f'{k}.npy') for k in arrays},
            background=pr,depth='conditional termination quantiles in camera-z; separate mean ray distance',
            chunk=args.chunk,seconds=time.monotonic()-started,render_box=box,
            projection_bound='saved model AABB; no masks or target RGB',queried_fraction=total/(h*w),
            empty_exterior_mean_ray_distance=0.)
        write(out/'receipt.json',receipt)
        log(root,'bounded_render_complete',camera=cid,seconds=receipt['seconds'],queried_fraction=total/(h*w))
