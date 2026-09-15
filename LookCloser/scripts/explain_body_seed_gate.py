"""Attribute rejected foreground ray hits to the separate confidence gates."""
import argparse
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json,cameras
from study_body_neighborhood_completion import ROOT
from study_body_single_depth_seed import ROOT as WEAK
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
import study_forearm_plane_transfer_v3 as prior


def run(frame):
    root=ROOT/frame;req=read(root/'request.json');e=np.load(root/'admission.npz');p=np.load(root/'proposal.npz')
    rows,_,_=cameras(frame);name='H004_A005_1210M6';camera=next(r for r in rows if r['physical_camera']==name)
    prior.configure();mask=np.rot90(prior.v2.v1.masks(frame)[name]);old=o3d.io.read_triangle_mesh(req['source_mesh'])
    v=np.asarray(old.vertices);t=np.asarray(old.triangles);nt=len(t);mv=p['vertices'];semantic=e['semantic_ids'];pp=p['proposals'][semantic]
    od,_,_=camera_depth(scene_for(v,t),camera);d,ids,_=camera_depth(scene_for(mv,np.concatenate([t,pp])),camera)
    od,d,ids=map(np.rot90,[od,d,ids]);selected=mask&~np.isfinite(od)&np.isfinite(d)&(ids>=nt)
    hit=ids[selected]-nt;free=e['free'].any(axis=(0,2));records=[]
    for label,certificate in [('three_depth',e['certificate']),('one_depth',np.load(WEAK/frame/'certificate_evidence.npz')['certificate'])]:
        lookup=np.zeros(len(mv),bool);lookup[e['query_ids']]=certificate;shape=lookup[pp].all(1)
        records.append(dict(variant=label,semantic_covered_old_misses=len(hit),
            shape_certificate_pass=int(shape[hit].sum()),sample_free_space_pass=int((~free[hit]).sum()),
            both_pass=int((shape[hit]&~free[hit]).sum()),strict_pass=int(e['strict'][hit].sum()),
            rejected_only_shape=int((~shape[hit]&~free[hit]).sum()),
            rejected_only_free=int((shape[hit]&free[hit]).sum()),rejected_both=int((~shape[hit]&free[hit]).sum())))
    atomic_json(root/'seed_gate_diagnosis.json',dict(frame=frame,records=records,
        source_admission_sha256=sha(root/'admission.npz'),weak_evidence_sha256=sha(WEAK/frame/'certificate_evidence.npz'),
        fixed_train_forearm_roi=True,counts_are_nearest_semantic_triangle_at_each_old_missing_ray=True,
        not_counterfactual_after_removing_occluders=True,heldout_used=False,geometry_changed=False))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001029','001033','001037'],required=True)
    run(p.parse_args().frame)
