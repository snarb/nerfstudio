"""Read-only attribution of fixed train-forearm misses across prior stages."""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras
from study_body_neighborhood_completion import ROOT
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from review_jaw_repair_transfer import panel
import study_forearm_plane_transfer_v3 as prior


def run(frame):
    root=ROOT/frame;dest=root/'admission_diagnosis';dest.mkdir(exist_ok=False)
    req=read(root/'request.json');result=read(root/'result.json')
    for n,h in result['hashes'].items():assert sha(root/n)==h
    rows,_,_=cameras(frame);camera=next(r for r in rows if r['physical_camera']=='H004_A005_1210M6')
    prior.configure();mask=np.rot90(prior.v2.v1.masks(frame)[camera['physical_camera']])
    old=o3d.io.read_triangle_mesh(req['source_mesh']);v=np.asarray(old.vertices);t=np.asarray(old.triangles);nt=len(t)
    p=np.load(root/'proposal.npz');mv=p['vertices'];pp=p['proposals'];a=np.load(root/'admission.npz')
    raw=o3d.io.read_triangle_mesh(str(root/'poisson_raw.ply'))
    accepted=a['semantic_ids'][a['strict']|a['prior']]
    variants=[('baseline',v,t),('full_raw_poisson',np.asarray(raw.vertices),np.asarray(raw.triangles)),
        ('local_proposals',mv,np.concatenate([t,pp])),
        ('semantic',mv,np.concatenate([t,pp[a['semantic_ids']]])),
        ('confidence',mv,np.concatenate([t,pp[accepted]])),
        ('guarded',mv,np.concatenate([t,pp[np.load(root/'retained.npz')['proposal_ids']]]))]
    baseline=None;records=[];images=[];depths={}
    for label,vv,tt in variants:
        d,ids,_=camera_depth(scene_for(vv,tt),camera);portrait=np.rot90(d);valid=np.isfinite(d)
        if baseline is None:baseline=mask&~np.isfinite(portrait)
        normals=np.cross(vv[tt[:,1]]-vv[tt[:,0]],vv[tt[:,2]]-vv[tt[:,0]])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        light=np.abs(normals@[.3,.4,.866]);rgb=np.zeros((*d.shape,3),np.uint8)
        rgb[valid]=(60+170*light[ids[valid],None]).astype(np.uint8)
        if label not in ['baseline','full_raw_poisson']:rgb[valid&(ids>=nt)]=[255,60,40]
        image=np.rot90(rgb).copy();images.append(image);depths[label]=np.where(np.isfinite(portrait),portrait,0)
        records.append(dict(stage=label,baseline_roi_misses=int(baseline.sum()),
            covered_baseline_misses=int((baseline&np.isfinite(portrait)).sum()),
            fixed_roi_misses=int((mask&~np.isfinite(portrait)).sum())))
    panel(dest/'train_forearm_stages.png',images,[r['stage'] for r in records],(0,1690,240,1920))
    panel(dest/'train_hand_stages.png',images,[r['stage'] for r in records],(0,1480,220,1760))
    np.savez_compressed(dest/'depths.npz',**depths,mask=mask)
    atomic_json(dest/'result.json',dict(frame=frame,records=records,
        source_result_sha256=sha(root/'result.json'),script_sha256=sha(__file__),
        heldout_used=False,production_changed=False,raw_poisson_is_not_accepted_geometry=True,
        region_protocol_sha256=sha(prior.OUT/'protocol.json'),
        no_quality_metrics=True,visual_status='pending'))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001029','001033','001037'],required=True)
    run(p.parse_args().frame)
