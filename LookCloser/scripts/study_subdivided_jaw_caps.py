"""Matched planar/curved subdivision controls, preserving native depth vetoes."""
from pathlib import Path
import argparse,time
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from subdivide_boundary_caps import subdivide
from study_jaw_repair_transfer import mask_votes,render
from diagnose_jaw_measured_depth import barycentric_samples
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_train_confidence import footprint_veto
from guard_jaw_measured_depth import initial_admission,measured_pixel_veto
from study_confidence_depth_prior import load_real
from diffusion_mesh_repair import scene_for

SOURCE=Path('/mnt/data/dec5_jaw_measured_mask_control')
OUT=Path('/mnt/data/dec5_subdivided_jaw_caps')


def prepare(root,frame,arm,large=False):
    folder=root/arm/frame;folder.mkdir(parents=True,exist_ok=False)
    base=SOURCE/frame;req=read(base/'request.json');result=read(base/'result.json')
    assert sha(base/'request.json')==result['request_sha256']
    for path,digest in result['hashes'].items():assert sha(base/path)==digest
    assert sha(req['source_mesh'])==req['source_mesh_sha256']
    source=o3d.io.read_triangle_mesh(req['source_mesh']);v=np.asarray(source.vertices);t=np.asarray(source.triangles)
    evidence=np.load(base/'evidence.npz');proposals=evidence['proposals'];proposal_notes=result['proposal_notes']
    cap_settings=req['settings']
    if large:
        from study_jaw_boundary_notches import propose
        cap_settings=dict(cap_settings,max_edges=36,max_extent=.006,max_gap=.003,max_plane_rmse=.0004)
        raw,proposal_notes=propose(v,t,cap_settings);proposals=raw[len(t):]
    vv,pp,notes=subdivide(v,t,proposals,proposal_notes,curved=arm=='curved')
    names_scripts=['study_subdivided_jaw_caps.py','subdivide_boundary_caps.py','study_jaw_repair_transfer.py',
                   'study_jaw_depth_footprint.py','study_jaw_train_confidence.py','guard_jaw_measured_depth.py',
                   'study_confidence_depth_prior.py','diagnose_jaw_measured_depth.py','study_jaw_boundary_notches.py']
    request=dict(frame=frame,arm=arm,source_request_sha256=sha(base/'request.json'),source_result_sha256=sha(base/'result.json'),
        source_mesh=req['source_mesh'],source_mesh_sha256=req['source_mesh_sha256'],
        max_centroid_shift=.0005,adjacent_original_mesh_rings=2,unchanged_boundary_vertices=True,cap_settings=cap_settings,large=large,
        evidence_rule='unchanged train reference, two-view sample support and native 124-ray free-space veto',
        ray_offsets=[0,.5],max_guard_rounds=8,geometry_uses_target=False,heldout_rgb_used=False,
        scripts={n:sha(Path(__file__).with_name(n)) for n in names_scripts})
    atomic_json(folder/'request.json',request)
    rows,depths,receipt=load_real(Path(req['depth_root']),frame)
    if receipt!=req['depth_receipt']:raise ValueError('Changed depth maps')
    parent=read('/mnt/data/dec5_phase30_early_texture_dynamic_150/request.json')
    record=next(r for r in parent['inventory'] if r['frame_id']==frame);maskroot=Path(record['source_masks']['root'])
    if sha(maskroot/'masks.npz')!=req['source_mask_sha256']:raise ValueError('Changed masks')
    masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json')
    if 'mask_override' in req:
        override=Path(req['mask_override']['root'])/frame;mr=read(override/'result.json')
        if sha(override/'result.json')!=req['mask_override']['result_sha256']:raise ValueError('Changed override')
        for name,digest in mr['hashes'].items():assert sha(override/name)==digest
        masks=masks.copy();masks[names.index(mr['camera'])]=np.load(override/'mask.npy')
    points=barycentric_samples(vv[pp]);votes,refs=train_reference_votes(points.reshape(-1,3),rows,depths)
    free=footprint_veto(points.reshape(-1,3),rows,depths).reshape(62,-1,10)
    mask_support,mask_outside=mask_votes(vv,pp,rows,masks,names)
    keep=initial_admission(votes.reshape(-1,10),free,mask_support,mask_outside)
    ids=np.flatnonzero(keep);tt=np.concatenate([t,pp[keep]]);rounds=[]
    for iteration in range(8):
        scene=scene_for(vv,tt);remove=set();checks=[]
        for ci,(camera,depth) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                implicated,count,raw_count=measured_pixel_veto(scene,camera,depth,rows,depths,len(t),len(tt),offset)
                remove.update(implicated.tolist());checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
            if (ci+1)%10==0:atomic_json(folder/'progress.json',dict(stage='native_guard',iteration=iteration,cameras=ci+1,unix_time=time.time()))
        rounds.append(dict(removed_triangles=len(remove),checks=checks));print(arm,frame,'guard',iteration,'remove',len(remove),flush=True)
        if not remove:break
        take=np.ones(len(tt),bool);take[list(remove)]=False
        if not take[:len(t)].all():raise ValueError('Original triangle deletion')
        ids=ids[take[len(t):]];tt=tt[take]
    if rounds[-1]['removed_triangles']:raise ValueError('Native guard did not converge')
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt));mesh.compute_vertex_normals()
    old_nonmanifold=len(source.get_non_manifold_edges(allow_boundary_edges=True))
    new_nonmanifold=len(mesh.get_non_manifold_edges(allow_boundary_edges=True))
    if new_nonmanifold>old_nonmanifold:raise ValueError('New nonmanifold edge')
    if not o3d.io.write_triangle_mesh(str(folder/'mesh.ply'),mesh):raise IOError('Mesh write failure')
    saved=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(saved.vertices)[:len(v)],v);np.testing.assert_array_equal(np.asarray(saved.triangles)[:len(t)],t)
    np.savez_compressed(folder/'evidence.npz',raw_proposals=proposals,proposals=pp,points=points,votes=votes.reshape(-1,10),references=refs.reshape(-1,10),
        free=free,mask_support=mask_support,mask_outside=mask_outside,retained_proposal_ids=ids)
    atomic_json(folder/'result.json',dict(request_sha256=sha(folder/'request.json'),proposal_fit=notes,proposal_notes=proposal_notes,
        original_vertices=len(v),original_triangles=len(t),proposed=len(pp),initially_admitted=int(keep.sum()),added=len(ids),
        max_centroid_shift=float(np.linalg.norm(vv[len(v):]-v[proposals].mean(1),axis=1).max()),
        nonmanifold_edges=new_nonmanifold,rounds=rounds,observed_guard_passed=True,original_prefix_exact=True,
        hashes={p:sha(folder/p) for p in ['mesh.ply','evidence.npz']},visual_status='pending',production_accepted=False))
    print(arm,frame,'done',int(keep.sum()),len(ids),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render']);p.add_argument('--frame',default='001193')
    p.add_argument('--arm',choices=['planar','curved'],required=True);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--large',action='store_true')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.output,a.frame,a.arm,a.large)
    else:render(a.output/a.arm,a.frame)
