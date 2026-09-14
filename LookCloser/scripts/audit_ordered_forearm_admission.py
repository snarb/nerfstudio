"""Replay admission evidence and verify final geometry constraints and ray audit."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from annotation_mask_domain import semantic_faces
from ordered_forearm_admission import rebuild
from append_verified_mesh_delta import append_delta
import study_forearm_plane_transfer_v3 as prior


def audit(root,frame):
    folder=root/frame;request=read(folder/'request.json');result=read(folder/'geometry_result.json')
    spec=request['admission_order']
    if spec['helper_sha256']!=sha(Path(__file__).with_name('ordered_forearm_admission.py')):raise ValueError('Changed admission helper')
    if spec['grid_helper_sha256']!=sha(Path(__file__).with_name('confidence_boundary_completion.py')):raise ValueError('Changed grid helper')
    if sha(folder/'request.json')!=result['request_sha256']:raise ValueError('Changed request')
    for name,h in result['hashes'].items():
        if sha(folder/name)!=h:raise ValueError('Changed output')
    fitpath=Path(request['curvature_source'])/frame/'result.json'
    if sha(fitpath)!=request['curvature_result_sha256']:raise ValueError('Changed shape fit')
    fit=next(r for r in read(fitpath)['fit'] if r['model']=='quadratic')
    raw,accepted,record,arrays=rebuild(frame,spec['shape'],fit)
    saved=np.load(folder/'admission_evidence.npz')
    for name,value in arrays.items():
        if not np.array_equal(value,saved[name]):raise ValueError('Admission evidence replay mismatch: '+name)
    if record!=result['transfer']['admission_order']:raise ValueError('Admission result mismatch')
    mesh=o3d.io.read_triangle_mesh(str(folder/'guarded.ply'))
    source=next(r for r in read('/mnt/data/dec5_phase30_dynamic_150/request.json')['inventory'] if r['frame_id']==frame)
    original=o3d.io.read_triangle_mesh(source['mesh']);nv,nt=len(original.vertices),len(original.triangles)
    prior_spec=read(prior.OUT/frame/'input.json')
    base=o3d.io.read_triangle_mesh(prior_spec['mesh'])
    delta=np.asarray(raw.triangles)[len(base.triangles):]
    selected=np.unique(delta);selected=selected[selected>=len(base.vertices)]
    rows,_,_=prior.v2.v1.cameras(frame)
    reference=next(r for r in rows if r['physical_camera']==prior.v2.v1.NAMES[0])
    if 'observed_ring_feather' in request:
        helper=Path(__file__).with_name('observed_ring_curvature.py')
        if sha(helper)!=request['observed_ring_feather']['helper_sha256']:raise ValueError('Changed ring helper')
        from observed_ring_curvature import boundary_curve_vertices
        md=np.load(prior.OUT/frame/'diagnostic.npz')[prior.v2.v1.NAMES[0]+'_mesh']
        pv,curvature=boundary_curve_vertices(np.asarray(raw.vertices),len(base.vertices),reference,fit,selected,accepted,md)
    else:
        from curve_forearm_delta import boundary_curve_vertices
        pv,curvature=boundary_curve_vertices(np.asarray(raw.vertices),len(base.vertices),reference,fit,selected,accepted)
    additions,semantic=semantic_faces(pv,delta,rows,prior.v2.v1.masks(frame),axis_extent=True)
    curvature.update(semantic)
    if curvature!=result['transfer']['curvature']:raise ValueError('Curvature receipt replay mismatch')
    expected_v,expected_t,_=append_delta(np.asarray(original.vertices),np.asarray(original.triangles),
        np.asarray(base.vertices),np.asarray(base.triangles),pv,np.concatenate([np.asarray(base.triangles),additions]))
    transferred=o3d.io.read_triangle_mesh(str(folder/'transferred.ply'))
    if not np.array_equal(expected_v,np.asarray(transferred.vertices)) or not np.array_equal(expected_t,np.asarray(transferred.triangles)):
        raise ValueError('Curved transfer replay mismatch')
    v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    if not np.array_equal(v[:nv],np.asarray(original.vertices)) or not np.array_equal(t[:nt],np.asarray(original.triangles)):
        raise ValueError('Changed original production geometry')
    rows,_,_=prior.v2.v1.cameras(frame);kept,_=semantic_faces(v,t[nt:],rows,prior.v2.v1.masks(frame),axis_extent=True)
    if len(kept)!=len(t)-nt:raise ValueError('Final semantic/extent violation')
    fresh=root/'fresh_audit'/(frame+'.json')
    if read(fresh)['geometry_result_sha256']!=sha(folder/'geometry_result.json'):raise ValueError('Wrong fresh ray audit')
    plane=Path('/mnt/data/dec5_forearm_admission_plane_bounded')/frame
    baseline=Path('/mnt/data/dec5_forearm_contrastive_guard_supported')/frame
    exact={n:sha(plane/n)==sha(baseline/n) for n in ['transferred.ply','guarded.ply']}
    if not all(exact.values()):raise ValueError('Matched plane control is not exact')
    atomic_json(folder/'independent_admission_audit.json',dict(frame=frame,admission_replay_exact=True,
        accepted=int(accepted.sum()),original_geometry_exact=True,final_semantic_and_extent_pass=True,
        curvature_and_transfer_replay_exact=True,original_depth_ring_exact=curvature['boundary_ring_exact'],
        matched_plane_mesh_bytes_exact=exact,geometry_result_sha256=sha(folder/'geometry_result.json'),
        fresh_ray_audit_sha256=sha(fresh),script_sha256=sha(__file__),artifact_free=False,production_accepted=False))
    print(frame,'admission replay, final geometry and matched plane control pass',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True)
    p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_forearm_admission_quadric_bounded'))
    a=p.parse_args();audit(a.root,a.frame)
