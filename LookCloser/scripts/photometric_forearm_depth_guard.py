"""Opt-in free-space veto requiring three depth AND color-compatible witnesses.

This does not alter input depth maps. A geometrically supported far observation
with incompatible RGB is insufficient to carve an inferred foreground patch.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from joint_temporal_texture import read,sha,exr,display,cameras,ROOT
from study_confidence_depth_prior import unproject
from diagnose_forearm_color_witnesses import witness_errors
from bake_joint_temporal_mesh import camera_depth


def make_guard(frame,rows,depths,limit=.04):
    actual,_,_=cameras(frame);profiles=np.load(ROOT/'parameters.npz')['log_gain'];gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    def load(pair):
        i,row=pair;rgb=display(exr(row['file_path'])*np.exp(profiles[i]),gain)
        return row['physical_camera'],np.rint(rgb*255).clip(0,255).astype(np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:images=dict(pool.map(load,enumerate(actual)))
    # Per-observed-pixel evidence is invariant to the proposed mesh and pruning pass.
    cache={r['physical_camera']:np.full((1080,1920,2),255,np.uint8) for r in rows}
    calls=[]
    def guard(scene,camera,observed,unused_rows,unused_depths,original_count,total_count,offset):
        actual=dict(camera)
        if offset==0:actual['cx']+=.5;actual['cy']+=.5
        d,ids,_=camera_depth(scene,actual)
        y,x=np.nonzero(np.isfinite(d)&(ids>=original_count)&(ids<total_count))
        qx=np.rint(x+offset).astype(int);qy=np.rint(y+offset).astype(int)
        valid=(qx<1920)&(qy<1080);x,y,qx,qy=x[valid],y[valid],qx[valid],qy[valid]
        obs=observed[qy,qx];far=np.isfinite(obs)&(obs>0)&(obs>d[y,x]+.003)
        x,y,qx,qy,obs=x[far],y[far],qx[far],qy[far],obs[far]
        stored=cache[camera['physical_camera']];unknown=stored[qy,qx,0]==255
        if unknown.any():
            coords=np.unique(np.column_stack([qx[unknown],qy[unknown]]),axis=0)
            points=unproject(camera,coords[:,0],coords[:,1],observed[coords[:,1],coords[:,0]])
            errors=witness_errors(points,camera,rows,depths,images)
            stored[coords[:,1],coords[:,0],0]=np.isfinite(errors).sum(0)
            stored[coords[:,1],coords[:,0],1]=(errors<=limit).sum(0)
        geometric=stored[qy,qx,0]>=3;qualified=stored[qy,qx,1]>=3
        calls.append(dict(camera=camera['physical_camera'],offset=offset,raw_far_pixels=len(x),
            geometric_veto_pixels=int(geometric.sum()),color_qualified_veto_pixels=int(qualified.sum()),
            color_disqualified_pixels=int((geometric&~qualified).sum())))
        return np.unique(ids[y[qualified],x[qualified]]).astype(int),int(qualified.sum()),len(x)
    provenance=dict(source_rgb_hashes={r['file_path']:sha(r['file_path']) for r in cameras(frame)[0]},
        profiles_sha256=sha(ROOT/'parameters.npz'),exposure_sha256=sha(ROOT/'exposure.json'),
        chroma_mean_abs_limit=limit,min_depth_and_color_witnesses=3,patch_size=5,
        input_depth_maps_changed=False,original_depth_only_guard_not_equivalent=True)
    return guard,calls,provenance
