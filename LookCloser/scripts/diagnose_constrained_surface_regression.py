"""Separate loss of existing production geometry from loss of earlier priors."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
import study_forearm_plane_transfer_v3 as prior

ROOT=Path('/mnt/data/dec5_constrained_forearm_surface_guard16')
PARENT=Path('/mnt/data/dec5_phase30_early_texture_dynamic_150')


def run(frame):
    folder=ROOT/frame;request=read(folder/'request.json');result=read(folder/'geometry_result.json')
    if result['request_sha256']!=sha(folder/'request.json'):raise ValueError('Changed request')
    for n,h in result['hashes'].items():
        if sha(folder/n)!=h:raise ValueError('Changed output')
    production=next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id']==frame)
    if sha(production['mesh'])!=production['mesh_sha256'] or sha(request['source_mesh'])!=request['source_mesh_sha256']:raise ValueError('Changed old mesh')
    old=o3d.io.read_triangle_mesh(request['source_mesh']);prod=o3d.io.read_triangle_mesh(production['mesh'])
    ov,ot=np.asarray(old.vertices),np.asarray(old.triangles);nt=len(prod.triangles)
    if not np.array_equal(ov[:len(prod.vertices)],np.asarray(prod.vertices)) or not np.array_equal(ot[:nt],np.asarray(prod.triangles)):raise ValueError('Not a production-prefix source')
    camera=request['reference_camera'];name=camera['physical_camera'];prior.configure();mask=prior.v2.v1.masks(frame)[name]
    old_depth,old_ids,_=camera_depth(scene_for(ov,ot),camera);maps={}
    for key,n in [('raw','transferred.ply'),('guarded','guarded.ply')]:
        mesh=o3d.io.read_triangle_mesh(str(folder/n));maps[key]=camera_depth(scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles)),camera)[0]
    old_hit=np.isfinite(old_depth);new_miss=~np.isfinite(maps['guarded']);new_holes=mask&old_hit&new_miss
    erased_production=new_holes&(old_ids<nt);erased_prior=new_holes&(old_ids>=nt)
    removal=np.load(folder/'evidence.npz')['removed_original_faces']
    records=dict(frame=frame,geometry_result_sha256=sha(folder/'geometry_result.json'),script_sha256=sha(__file__),
        production_prefix_verified=True,production_triangles=nt,removed_production_triangles=int(removal[:nt].sum()),
        removed_earlier_prior_triangles=int(removal[nt:].sum()),new_roi_geometry_holes=int(new_holes.sum()),
        new_holes_over_previous_production_faces=int(erased_production.sum()),new_holes_over_previous_inferred_faces=int(erased_prior.sum()),
        holes_created_during_assembly=int((new_holes&~np.isfinite(maps['raw'])).sum()),
        holes_created_by_final_carving=int((new_holes&np.isfinite(maps['raw'])).sum()),
        raw_geometry_not_target_masked=True,fixed_train_roi=True,no_new_face_metrics=True)
    atomic_json(folder/'production_loss_diagnosis.json',records);print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True);a=p.parse_args();run(a.frame)
