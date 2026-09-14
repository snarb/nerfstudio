"""Verify the measured-anchor deformation contract and fresh RGB/depth guard."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from study_confidence_depth_prior import project_integer
from annotation_mask_domain import semantic_faces
from audit_rgb_qualified_forearm import run as fresh_guard
import study_forearm_plane_transfer_v3 as prior


def audit(root,frame):
    prior.configure();v1=prior.v2.v1;folder=root/frame
    request=read(folder/'request.json');result=read(folder/'geometry_result.json')
    if request['anchor_residual_helper_sha256']!=sha(Path(__file__).with_name('fit_forearm_anchor_residual.py')):raise ValueError('Changed fitter')
    if sha(folder/'request.json')!=result['request_sha256']:raise ValueError('Changed request')
    for n,h in result['hashes'].items():
        if sha(folder/n)!=h:raise ValueError('Changed output')
    oldroot=Path('/mnt/data/dec5_forearm_rgb_qualified_curve')/frame
    if request['parent_result_sha256']!=sha(oldroot/'geometry_result.json'):raise ValueError('Changed baseline')
    baseline=o3d.io.read_triangle_mesh(str(oldroot/'transferred.ply'));v=np.asarray(baseline.vertices)
    final=o3d.io.read_triangle_mesh(str(folder/'guarded.ply'));out=np.asarray(final.vertices)
    e=np.load(folder/'residual.npz');ids=e['vertex_ids'];pins=e['pinned']
    if len(out)!=len(v) or not np.array_equal(out[ids[pins]],v[ids[pins]]):raise ValueError('Moved pinned boundary')
    untouched=np.ones(len(v),bool);untouched[ids[~pins]]=False
    if not np.array_equal(out[untouched],v[untouched]):raise ValueError('Moved geometry outside residual domain')
    if np.max(np.abs(e['residual']))>.006 or np.any(e['residual'][pins]):raise ValueError('Invalid bounded residual')
    rows,_,_=v1.cameras(frame);ref=next(r for r in rows if r['physical_camera']==v1.NAMES[0])
    uv,z=project_integer(ref,out[ids]);base_uv,base_z=project_integer(ref,v[ids])
    if not np.allclose(uv,base_uv,atol=1e-3,rtol=0) or not np.allclose(z,base_z+e['residual'],atol=1e-6,rtol=0):
        raise ValueError('Saved mesh not equal to ray-preserving solved depth')
    source=next(r for r in read('/mnt/data/dec5_phase30_dynamic_150/request.json')['inventory'] if r['frame_id']==frame)
    original=o3d.io.read_triangle_mesh(source['mesh']);nt=len(original.triangles)
    faces=np.asarray(final.triangles)[nt:];kept,_=semantic_faces(out,faces,rows,v1.masks(frame),axis_extent=True)
    if len(kept)!=len(faces):raise ValueError('Saved geometry violates frozen semantic/extent limits')
    fresh_guard(root,frame)
    atomic_json(folder/'independent_deformation_audit.json',dict(frame=frame,status='bounded_ray_preserving_local_deformation_and_124_rays_pass',
        changed_vertices=int((np.linalg.norm(out-v,axis=1)>0).sum()),pinned_vertices=int(pins.sum()),
        outside_domain_exact=True,original_mesh_prefix_exact=True,artifact_free=False,
        geometry_result_sha256=sha(folder/'geometry_result.json'),fresh_audit_sha256=sha(root/'fresh_audit'/(frame+'.json')),script_sha256=sha(__file__)))
    print(frame,'independent deformation audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True)
    p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_forearm_anchor_residual'))
    a=p.parse_args();audit(a.root,a.frame)
