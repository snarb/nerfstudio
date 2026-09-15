"""Four-view wrong-object qualification of the existing measured-depth guard.

Same lower-row proposal, masks, stereo, calibration and edge thresholds. A common
rule for all 62 query cameras; never special-case B/E or any actor time.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import minimum_filter,distance_transform_edt,map_coordinates
from joint_temporal_texture import read,sha,atomic_json,project
from calibrated_depth_witness import load_images
from study_confidence_depth_prior import support,unproject
from bake_joint_temporal_mesh import camera_depth
from foreground_layer_evidence import rejects_far_layer
from study_foundation_lower_forearm import ROOT as LOWER,FRAME

ROOT=Path('/mnt/data/dec5_forearm_layer_qualified_guard')


def make_guard(images,stage,stats):
    names=[];rectified=[];warm_cache={}
    for pair in stage['pairs']:
        cal=np.load(Path(pair['directory'])/'calibration.npz')
        for side in ['left','right']:
            name=pair[side];names.append(name)
            if side=='left':e=cal['rectified_extrinsic'];k=cal['cropped_intrinsic']
            else:
                e=np.eye(4);e[:3,:3]=cal['R2']@cal['E2'][:3,:3];e[:3,3]=cal['R2']@cal['E2'][:3,3];k=cal['P2'][:,:3]
            rectified.append((e,k,distance_transform_edt(cal[side+'_mask'].astype(bool))))
    assert len(names)==len(set(names))==4

    def color_difference(points,witnesses):
        uv,z=project(points,witnesses);values=[];available=[]
        for row,xy,depth in zip(witnesses,uv,z):
            delta=images[row['physical_camera']][...,0].astype(float)-images[row['physical_camera']][...,2]
            values.append(map_coordinates(delta,xy.T[::-1],order=1,mode='constant',cval=0))
            available.append((depth>0)&(xy[:,0]>3)&(xy[:,0]<1916)&(xy[:,1]>3)&(xy[:,1]<1076))
        return np.array(values).T,np.array(available).T

    def guard(scene,camera,observed,rows,depths,original_count,total_count,offset):
        actual=dict(camera)
        if offset==0:actual['cx']+=.5;actual['cy']+=.5
        d,ids,_=camera_depth(scene,actual)
        y,x=np.nonzero(np.isfinite(d)&(ids>=original_count)&(ids<total_count))
        qx=np.rint(x+offset).astype(int);qy=np.rint(y+offset).astype(int)
        valid=(qx<1920)&(qy<1080);x,y,qx,qy=x[valid],y[valid],qx[valid],qy[valid]
        obs=observed[qy,qx];far=np.isfinite(obs)&(obs>0)&(obs>d[y,x]+.003)
        x,y,qx,qy,obs=x[far],y[far],qx[far],qy[far],obs[far];raw=len(x)
        old_points=unproject(camera,qx,qy,obs);votes,_=support(old_points,camera,rows,depths);trusted=votes>=3
        x,y,qx,qy,old_points=x[trusted],y[trusted],qx[trusted],qy[trusted],old_points[trusted]
        if not len(x):
            stats.append(dict(camera=camera['physical_camera'],offset=offset,raw_far=raw,original_trusted=0,disqualified=0,remaining=0))
            return np.empty(0,int),0,raw
        near=unproject(camera,x,y,d[y,x],offset=offset)
        name=camera['physical_camera']
        if name not in warm_cache:
            im=images[name];warm_cache[name]=minimum_filter(im[...,0].astype(float)-im[...,2],size=3)>8
        query_warm=warm_cache[name][qy,qx]
        lookup={r['physical_camera']:r for r in rows};witnesses=[lookup[n] for n in names]
        near_color,near_known=color_difference(near,witnesses)
        far_color,far_known=color_difference(old_points,witnesses)
        inside=[]
        for e,k,mask_distance in rectified:
            p=near@e[:3,:3].T+e[:3,3];uv=p@k.T;uv=uv[:,:2]/uv[:,2:]
            inside.append((p[:,2]>0)&(map_coordinates(mask_distance,uv.T[::-1],order=1,mode='constant',cval=0)>=3))
        disqualified=rejects_far_layer(query_warm,np.array(inside).T,near_color>8,far_color < -8,near_known&far_known)
        keep=~disqualified
        stats.append(dict(camera=name,offset=offset,raw_far=raw,original_trusted=len(x),
            disqualified=int(disqualified.sum()),remaining=int(keep.sum())))
        return np.unique(ids[y[keep],x[keep]]).astype(int),int(keep.sum()),raw
    return guard


def build():
    import build_foundation_foreground_patch as engine
    ROOT.mkdir(exist_ok=False);stage=read(LOWER/FRAME/'request.json');images,_,rgb=load_images(FRAME)
    assert rgb==stage['rgb_receipt']
    request=dict(frame=FRAME,lower_stage_request_sha256=sha(LOWER/FRAME/'request.json'),
        original_guard_request_sha256=sha(LOWER/'foreground/request.json'),
        raw_proposal_sha256=sha(LOWER/'foreground/proposal.npz'),
        source_rgb_receipt=rgb,physical_witnesses=[n for p in stage['pairs'] for n in [p['left'],p['right']]],
        rule=dict(query_warm_minimum_3x3=8,near_warm_minimum=8,far_blue_minimum=8,
            near_region_interior_pixels=3,near_region_views=4,near_warm_views=4,far_blue_views=3,all_four_views_available=True),
        skin_to_skin_override=False,blanket_guard_disable=False,per_camera_exceptions=False,
        script_hashes={str(Path(__file__).with_name(n).resolve()):sha(Path(__file__).with_name(n)) for n in
            [Path(__file__).name,'foreground_layer_evidence.py','build_foundation_foreground_patch.py','guard_jaw_measured_depth.py']},
        effective_guard='new color/region qualifier over original measured-depth witness; base request guard SHA is provenance, not unchanged behavior',
        heldout_used=False,production_updated=False)
    atomic_json(ROOT/'qualification_request.json',request);stats=[]
    engine.ROOT=ROOT/'geometry';engine.CONTROL=LOWER/'empty_ray';engine.BIAS=LOWER/'bias'
    engine.measured_pixel_veto=make_guard(images,stage,stats);engine.run()
    assert sha(ROOT/'geometry/proposal.npz')==request['raw_proposal_sha256']
    atomic_json(ROOT/'qualification_result.json',dict(request_sha256=sha(ROOT/'qualification_request.json'),
        geometry_result_sha256=sha(ROOT/'geometry/result.json'),checks=stats,
        original_guard_counts_not_zero_by_definition=True,visual_status='pending',production_updated=False))
    print('originally trusted',sum(s['original_trusted'] for s in stats),'disqualified',sum(s['disqualified'] for s in stats),flush=True)


if __name__=='__main__':build()
