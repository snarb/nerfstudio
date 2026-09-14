"""Fresh ray audit of the experimental color-qualified, not depth-only, guard."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from diffusion_mesh_repair import scene_for
from photometric_forearm_depth_guard import make_guard
import study_forearm_plane_transfer_v3 as prior


def run(root,frame):
    prior.configure();v1=prior.v2.v1;rows,depths,hashes=v1.load_real(frame)
    result=read(root/frame/'geometry_result.json');request=read(root/frame/'request.json')
    if sha(root/frame/'request.json')!=result['request_sha256'] or hashes!=result['depth_hashes']:raise ValueError('Changed inputs')
    if request['observed_guard']['kind']!='depth_and_color_witnesses':raise ValueError('Wrong guard protocol')
    for p,h in result['hashes'].items():
        if sha(root/frame/p)!=h:raise ValueError('Changed mesh output')
    source=next(r for r in read('/mnt/data/dec5_phase30_dynamic_150/request.json')['inventory'] if r['frame_id']==frame)
    if sha(source['mesh'])!=source['mesh_sha256']:raise ValueError('Changed production input')
    old=o3d.io.read_triangle_mesh(source['mesh']);mesh=o3d.io.read_triangle_mesh(str(root/frame/'guarded.ply'))
    v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    if not np.array_equal(v[:len(old.vertices)],np.asarray(old.vertices)) or not np.array_equal(t[:len(old.triangles)],np.asarray(old.triangles)):
        raise ValueError('Changed old production geometry')
    guard,calls,provenance=make_guard(frame,rows,depths)
    if provenance!=result['color_guard_provenance']:raise ValueError('Changed RGB evidence')
    scene=scene_for(v,t)
    for camera,depth in zip(rows,depths):
        for offset in [0,.5]:
            bad,count,_=guard(scene,camera,depth,rows,depths,len(old.triangles),len(t),offset)
            if len(bad) or count:raise ValueError('Final color-qualified free-space contradiction')
    if calls!=result['color_guard_calls'][-124:]:raise ValueError('Ray evidence differs from completed run')
    dest=root/'fresh_audit';dest.mkdir(exist_ok=True)
    atomic_json(dest/(frame+'.json'),dict(frame=frame,script_sha256=sha(__file__),geometry_result_sha256=sha(root/frame/'geometry_result.json'),
        color_guard_kind=request['observed_guard']['kind'],checks=124,qualified_veto_pixels=0,
        original_depth_only_veto_pixels=sum(r['geometric_veto_pixels'] for r in calls),
        old_geometry_exact=True,production_accepted=False,visual_review_required=True))
    print(frame,'fresh audit pass; depth-only veto pixels',sum(r['geometric_veto_pixels'] for r in calls),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',required=True);a=p.parse_args();run(a.root,a.frame)
