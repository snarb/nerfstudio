"""Skip only rays provably outside every saved occupied cell projection.

This is a conservative CPU rasterization of cell bounding rectangles, not a
new occupancy grid or geometry reconstruction. Saved model buffers are read-only.
"""
import itertools
from pathlib import Path
import time
import numpy as np
import torch
from luster_hybrid_common import DIGEST,TARGETS,image,log,native_rows,project,read,sha,write
from render_luster_hybrid import load,packed_quantiles,state_hashes


def project_cell_union(cell_min,cell_max,row):
    h,w=row['h'],row['w']
    difference=np.zeros((h+1,w+1),np.int32)
    offsets=np.array(list(itertools.product([0,1],repeat=3)),np.float32)
    for start in range(0,len(cell_min),16384):
        lo=cell_min[start:start+16384];hi=cell_max[start:start+16384]
        corners=lo[:,None]+offsets[None]*(hi-lo)[:,None]
        uv,z=project(corners,row)
        if not np.isfinite(uv).all() or np.any(z<=0):
            return np.ones((h,w),bool)
        low=np.floor(uv.min(1)).astype(np.int64)-2
        high=np.ceil(uv.max(1)).astype(np.int64)+3
        x0=low[:,0].clip(0,w);y0=low[:,1].clip(0,h)
        x1=high[:,0].clip(0,w);y1=high[:,1].clip(0,h)
        np.add.at(difference,(y0,x0),1);np.add.at(difference,(y0,x1),-1)
        np.add.at(difference,(y1,x0),-1);np.add.at(difference,(y1,x1),1)
    np.cumsum(difference,axis=0,out=difference)
    np.cumsum(difference,axis=1,out=difference)
    return difference[:-1,:-1]>0


def saved_cells(model):
    assert model.config.adaptive_fixed_fallback_samples_per_ray==0
    assert model.config.occupancy_fixed_fallback_samples_per_ray==0
    assert model.config.background_color=='black'
    assert model.config.occupancy_prior_grid_path is None
    assert not getattr(model.config,'eval_occupancy_dilation_radius',0)
    grid=model.occupancy_grid
    coords=grid.grid_coords.detach().cpu().numpy()
    res=grid.resolution.detach().cpu().numpy()
    lows=[];highs=[]
    for binary,box in zip(grid.binaries.detach().cpu().numpy(),grid.aabbs.detach().cpu().numpy()):
        ids=np.flatnonzero(binary.ravel());ijk=coords[ids]
        width=(box[3:]-box[:3])/res
        lows.append(box[:3]+ijk*width);highs.append(box[:3]+(ijk+1)*width)
    return np.concatenate(lows).astype(np.float32),np.concatenate(highs).astype(np.float32)


def run(args):
    root=args.root;cfg,model=load(root,args.chunk)
    from render_luster_quality import camera_for,measured_plate
    before=state_hashes(model);rows=native_rows();cell_min,cell_max=saved_cells(model)
    for cid in args.cameras:
        row=rows[cid];out=root/'geometry'/f'cam{cid:03d}'
        if (out/'receipt.json').exists():
            r=read(out/'receipt.json')
            assert r['checkpoint_sha256']==DIGEST and r['state_hash']==before
            assert all(sha(out/f'{k}.npy')==r['hashes'][k] for k in r['arrays'])
            log(root,'occupancy_render_cached',camera=cid);continue
        out.mkdir(parents=True,exist_ok=True)
        h,w=row['h'],row['w'];started=last=time.monotonic()
        query=project_cell_union(cell_min,cell_max,row)
        ids_cpu=np.flatnonzero(query)
        arrays={k:np.lib.format.open_memmap(out/f'{k}.npy',mode='w+',dtype=np.float32,
                shape=(h,w,3) if k=='rgb' else (h,w)) for k in
                ['rgb','alpha','ray_distance_mean','depth_z_q10','depth_z_q50','depth_z_q90']}
        camera=camera_for(row,model.device);plate,pr=measured_plate(cfg,row)
        arrays['rgb'][:]=plate if plate is not None else 0.
        for k in ['alpha','ray_distance_mean']:arrays[k][:]=0.
        for k in ['depth_z_q10','depth_z_q50','depth_z_q90']:arrays[k][:]=np.nan
        with torch.inference_mode():
            for start in range(0,len(ids_cpu),args.chunk):
                ids=torch.from_numpy(ids_cpu[start:start+args.chunk]).to(model.device)
                yy=ids//w;xx=ids%w
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
                    log(root,'occupancy_render_progress',camera=cid,pixels=start+len(ids),total=len(ids_cpu),
                        elapsed=time.monotonic()-started,gpu_peak_gb=torch.cuda.max_memory_allocated()/1e9)
                    last=time.monotonic()
        for value in arrays.values():value.flush()
        assert state_hashes(model)==before
        if cid in TARGETS:image(root/'baseline'/f'cam{cid:03d}.png',arrays['rgb'])
        np.save(out/'query_mask.npy',query)
        receipt=dict(state='complete',camera=cid,row=row,arrays=list(arrays),checkpoint_sha256=DIGEST,
            state_hash=before,code_sha256=sha(Path(__file__)),hashes={k:sha(out/f'{k}.npy') for k in arrays},
            background=pr,depth='conditional termination quantiles in camera-z; separate mean ray distance',
            chunk=args.chunk,seconds=time.monotonic()-started,
            query_mask_sha256=sha(out/'query_mask.npy'),
            projection_bound='union of conservative projected saved occupied-cell rectangles; no masks/GT',
            queried_fraction=float(query.mean()),empty_exterior_mean_ray_distance=0.)
        write(out/'receipt.json',receipt)
        log(root,'occupancy_render_complete',camera=cid,seconds=receipt['seconds'],queried_fraction=float(query.mean()))
