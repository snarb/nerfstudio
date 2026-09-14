"""Opt-in free-space carving only with three depth-discriminating RGB witnesses."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from joint_temporal_texture import read,sha,exr,display,cameras,ROOT
from study_confidence_depth_prior import unproject
from contrastive_forearm_witnesses import comparison_votes,carving_decision
from bake_joint_temporal_mesh import camera_depth


def make_guard(frame,rows,depths,margin=.01):
    if margin<=0:raise ValueError('Positive comparison margin required')
    actual,_,_=cameras(frame);profiles=np.load(ROOT/'parameters.npz')['log_gain'];gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    def load(pair):
        i,row=pair
        return row['physical_camera'],np.rint(display(exr(row['file_path'])*np.exp(profiles[i]),gain)*255).clip(0,255).astype(np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:images=dict(pool.map(load,enumerate(actual)))
    calls=[]
    def guard(scene,camera,observed,unused_rows,unused_depths,original_count,total_count,offset):
        ray=dict(camera)
        if offset==0:ray['cx']+=.5;ray['cy']+=.5
        d,ids,_=camera_depth(scene,ray)
        y,x=np.nonzero(np.isfinite(d)&(ids>=original_count)&(ids<total_count))
        qx=np.rint(x+offset).astype(int);qy=np.rint(y+offset).astype(int)
        valid=(qx<1920)&(qy<1080);x,y,qx,qy=x[valid],y[valid],qx[valid],qy[valid]
        obs=observed[qy,qx];far=np.isfinite(obs)&(obs>0)&(obs>d[y,x]+.003)
        x,y,qx,qy,obs=x[far],y[far],qx[far],qy[far],obs[far]
        points=unproject(camera,qx,qy,obs);candidate=unproject(camera,x+offset,y+offset,d[y,x])
        votes=comparison_votes(points,candidate,camera,rows,depths,images,margin)
        qualified=carving_decision(votes)
        calls.append(dict(camera=camera['physical_camera'],offset=offset,raw_far_pixels=len(x),
            geometric_veto_pixels=int((votes['geometric']>=3).sum()),rgb_only_veto_pixels=int((votes['rgb_qualified']>=3).sum()),
            color_qualified_veto_pixels=int(qualified.sum()),comparable_pixels=int((votes['comparable']>=3).sum()),
            ambiguous_pixels=int(((votes['rgb_qualified']>=3)&~qualified).sum()),
            unavailable_comparison_fallback_pixels=int(((votes['rgb_qualified']>=3)&(votes['comparable']<3)).sum())))
        return np.unique(ids[y[qualified],x[qualified]]).astype(int),int(qualified.sum()),len(x)
    provenance=dict(source_rgb_hashes={r['file_path']:sha(r['file_path']) for r in actual},
        profiles_sha256=sha(ROOT/'parameters.npz'),exposure_sha256=sha(ROOT/'exposure.json'),
        chroma_mean_abs_limit=.04,rgb_mean_abs_limit=.12,comparison_margin=margin,min_depth_and_color_witnesses=3,patch_size=5,
        input_depth_maps_changed=False,original_depth_only_guard_not_equivalent=True,rgb_only_guard_not_equivalent=True,
        unavailable_comparison_uses_original_rgb_rule=True)
    return guard,calls,provenance
