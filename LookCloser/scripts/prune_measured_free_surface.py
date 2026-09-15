"""Uniform opt-in removal of surfaces contradicted by measured free space.

No ROI, target view, RGB, semantic mask or learned depth selects geometry.
All three vertices and the centroid must have zero near observations in every
5x5 native footprint, plus six stable farther footprints and six separately
corroborated farther observations. An available near tap protects the face.
"""
from pathlib import Path
import argparse
import time
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json
from review_full_block_transfer import ROOT as DEPTH_ROOT, VIDEO
from study_confidence_depth_prior import load_real, project_integer, support, unproject
from diagnose_jaw_measured_depth import observed_at
from carve_patchmatch_mesh_free_space import free_space_evidence

ROOT=Path('/mnt/data/dec5_measured_free_surface_pruning')


def near_tap_evidence(depth,uv,z,tolerance=.0015,radius=2):
    if depth.ndim!=2 or uv.shape!=(len(z),2) or tolerance<=0 or radius<0:
        raise ValueError('Invalid footprint inputs')
    finite=np.isfinite(uv).all(1)&np.isfinite(z)&(z>0)
    xy=np.rint(np.where(np.isfinite(uv),uv,-99999)).astype(np.int64)
    answer=np.zeros(len(z),bool)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            x,y=xy[:,0]+dx,xy[:,1]+dy
            valid=finite&(x>=0)&(x<depth.shape[1])&(y>=0)&(y<depth.shape[0])
            j=np.flatnonzero(valid);d=depth[y[j],x[j]]
            answer[j]|=np.isfinite(d)&(d>0)&(np.abs(d-z[j])<=tolerance)
    return answer


def removable(near_counts,stable_far_counts,trusted_far_counts):
    a,b,c=map(np.asarray,(near_counts,stable_far_counts,trusted_far_counts))
    if a.shape!=b.shape or a.shape!=c.shape or a.ndim!=2 or a.shape[1]!=4:
        raise ValueError('Expected matched triangle x four-sample counts')
    return (a==0).all(1)&(b>=6).all(1)&(c>=6).all(1)


def run(frame):
    root=ROOT/frame;root.mkdir(parents=True,exist_ok=False)
    entry=next(r for r in read(VIDEO/'request.json')['inventory'] if r['frame_id']==frame)
    assert sha(entry['mesh'])==entry['mesh_sha256']
    rows,depths,receipt=load_real(DEPTH_ROOT,frame)
    received=read(DEPTH_ROOT/frame/'received.json')
    for name,digest in received['depth_hashes'].items(): assert sha(DEPTH_ROOT/frame/name)==digest
    mesh=o3d.io.read_triangle_mesh(entry['mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    # Shared vertices are evaluated once, without changing their coordinates.
    points=np.concatenate([v,v[t].mean(1)])
    sample_indices=np.column_stack([t,np.arange(len(t))+len(v)])
    q=dict(frame=frame,mesh=entry['mesh'],mesh_sha256=entry['mesh_sha256'],
        source_video_sha256=sha(VIDEO/'request.json'),depth_receipt=receipt,
        query='all mesh triangles; vertices plus centroid',
        parameters=dict(near_tolerance=.0015,near_native_radius=2,protect_any_near_tap=True,
            far_minimum_gap=.005,far_relative_gap=.01,far_native_radius=2,
            far_native_fraction=.8,far_middle_spread_fraction=.005,min_stable_far_views=6,
            min_trusted_far_views=6,far_observation_other_views=3,far_support_tolerance=.001,
            far_roundtrip_px=1.5,far_min_parallax_deg=1),
        geometry_uses_target=False,geometry_uses_rgb=False,geometry_uses_roi=False,
        geometry_uses_masks=False,heldout_used=False,vertices_moved=False,production_changed=False,
        scripts={str(Path(__file__).resolve().with_name(n)):sha(Path(__file__).with_name(n)) for n in
            [Path(__file__).name,'study_confidence_depth_prior.py','diagnose_jaw_measured_depth.py',
             'carve_patchmatch_mesh_free_space.py','joint_temporal_texture.py']})
    atomic_json(root/'request.json',q)
    near=np.zeros(len(points),np.uint8);stable=np.zeros_like(near)
    for ci,(row,d) in enumerate(zip(rows,depths)):
        uv,z=project_integer(row,points)
        near+=near_tap_evidence(d,uv,z)
        free,_=free_space_evidence(d,uv[:,0],uv[:,1],z,minimum_gap=.005,near_gap=.0015)
        stable+=free
        atomic_json(root/'progress.json',dict(stage='all_mesh_footprints',camera=ci+1,time=time.time()))
    near4=near[sample_indices];stable4=stable[sample_indices]
    candidates=np.flatnonzero((near4==0).all(1)&(stable4>=6).all(1))
    selected_points=points[sample_indices[candidates]].reshape(-1,3)
    trusted=np.zeros((62,len(candidates),4),bool)
    for ci,(row,d) in enumerate(zip(rows,depths)):
        xy,z,obs,ok=observed_at(selected_points,row,d)
        j=np.flatnonzero(ok&(obs>z+np.maximum(.005,.01*z)))
        if len(j):
            count,_=support(unproject(row,xy[j,0],xy[j,1],obs[j]),row,rows,depths)
            trusted[ci].reshape(-1)[j]=count>=3
        atomic_json(root/'progress.json',dict(stage='far_observation_corroboration',camera=ci+1,
            candidate_triangles=len(candidates),time=time.time()))
    remove_ids=candidates[removable(near4[candidates],stable4[candidates],trusted.sum(0))]
    keep=np.ones(len(t),bool);keep[remove_ids]=False
    assert keep.any()
    out=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[keep]))
    out.compute_vertex_normals();assert o3d.io.write_triangle_mesh(str(root/'mesh.ply'),out)
    saved=o3d.io.read_triangle_mesh(str(root/'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(saved.vertices),v)
    np.testing.assert_array_equal(np.asarray(saved.triangles),t[keep])
    np.savez_compressed(root/'evidence.npz',sample_indices=sample_indices,points=points,
        near_counts=near,stable_far_counts=stable,candidates=candidates,
        trusted_far_by_camera=trusted,removed_triangle_ids=remove_ids)
    _,components,_=out.cluster_connected_triangles()
    atomic_json(root/'result.json',dict(request_sha256=sha(root/'request.json'),
        before_triangles=len(t),after_triangles=int(keep.sum()),removed_triangles=len(remove_ids),
        candidate_triangles=len(candidates),components=sorted(map(int,components),reverse=True),
        vertices_unchanged=True,triangle_subset_exact=True,no_component_cleanup=True,
        production_changed=False,visual_status='pending',
        hashes={n:sha(root/n) for n in ('mesh.ply','evidence.npz')}))
    print(frame,'triangles',len(t),'candidates',len(candidates),'removed',len(remove_ids),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',default='000995',choices=['000995'])
    run(p.parse_args().frame)
