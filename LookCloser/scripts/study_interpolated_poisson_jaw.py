"""Use a local cross-validated surface prior only inside observed-anchor hulls."""
from pathlib import Path
import argparse,time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from joint_temporal_texture import read,sha,atomic_json
from study_poisson_jaw_completion import OUT,SOURCE,FRAME
from study_confidence_depth_prior import load_real
from study_jaw_depth_footprint import train_reference_votes
from guard_jaw_measured_depth import measured_pixel_veto
from local_surface_certificate import certify
from diffusion_mesh_repair import scene_for
from study_jaw_repair_transfer import render


def prepare():
    folder=OUT/'interpolated'/FRAME;folder.mkdir(parents=True,exist_ok=False)
    bq=read(SOURCE/'request.json');pr=read(OUT/'result.json');ar=read(OUT/'admission/result.json')
    for p,h in pr['hashes'].items():assert sha(OUT/p)==h
    assert sha(OUT/'admission/samples.npz')==ar['arrays_sha256']
    rows,depths,receipt=load_real(Path(bq['depth_root']),FRAME);assert receipt==bq['depth_receipt']
    req=dict(frame=FRAME,raw_result_sha256=sha(OUT/'result.json'),admission_result_sha256=sha(OUT/'admission/result.json'),
        source_mesh=bq['source_mesh'],source_mesh_sha256=bq['source_mesh_sha256'],seed_radius=.003,nearest_seeds=24,
        minimum_seed_depth_views=3,maximum_seed_poisson_distance=.0005,minimum_normal_dot=.5,
        minimum_fitting_seeds=8,maximum_loo_p90=.0005,maximum_predicted_offset=.0005,
        require_query_inside_seed_hull=True,original_geometry_preserved=True,heldout_used=False,
        prior_not_direct_observation=True,script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('local_surface_certificate.py')))
    atomic_json(folder/'request.json',req)
    old=o3d.io.read_triangle_mesh(bq['source_mesh']);old.compute_vertex_normals();v=np.asarray(old.vertices);nt=len(old.triangles)
    mesh=o3d.io.read_triangle_mesh(str(OUT/'local_raw.ply'));mv=np.asarray(mesh.vertices);mt=np.asarray(mesh.triangles)
    raw=o3d.io.read_triangle_mesh(str(OUT/'poisson_raw.ply'));raw.compute_vertex_normals();raw_scene=scene_for(np.asarray(raw.vertices),np.asarray(raw.triangles))
    pe=np.load(OUT/'proposal_evidence.npz');a=np.load(OUT/'admission/samples.npz');proposals=pe['proposals'];semantic=a['semantic_ids'];pp=proposals[semantic]
    used=np.unique(pp);q=mv[used];normals=np.asarray(raw.vertex_normals)[pe['raw_vertex_ids'][used-len(v)]]
    neighborhoods=cKDTree(v).query_ball_point(q,.003);seed_ids=np.unique(np.concatenate([np.asarray(n,int) for n in neighborhoods]))
    votes,_=train_reference_votes(v[seed_ids],rows,depths)
    nearest=raw_scene.compute_closest_points(o3d.core.Tensor(v[seed_ids].astype(np.float32)))['points'].numpy()
    discrepancy=np.linalg.norm(v[seed_ids]-nearest,axis=1);valid=(votes>=3)&(discrepancy<=.0005)
    seeds=v[seed_ids[valid]];seed_normals=np.asarray(old.vertex_normals)[seed_ids[valid]]
    distance,neighbors=cKDTree(seeds).query(q,k=min(24,len(seeds)));certificate=np.zeros(len(q),bool);notes=[];accepted_neighbors=[]
    if distance.ndim!=2:raise ValueError('Insufficient observed seeds')
    for i in range(len(q)):
        take=(distance[i]<=.003)&((seed_normals[neighbors[i]]@normals[i])>=.5);index=neighbors[i][take]
        certificate[i],note=certify(q[i],normals[i],seeds[index]);notes.append(note);accepted_neighbors.append(index.tolist())
    lookup=np.zeros(len(mv),bool);lookup[used]=certificate
    prior=lookup[pp].all(1)&~a['free'].any(axis=(0,2))
    keep=a['strict']|prior;ids=semantic[keep];t=np.concatenate([mt[:nt],proposals[ids]]);rounds=[]
    print('verified seeds',len(seeds),'certified vertices',int(certificate.sum()),'initial faces',len(ids),flush=True)
    for iteration in range(8):
        scene=scene_for(mv,t);remove=set();checks=[]
        for ci,(camera,depth) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                implicated,count,raw_count=measured_pixel_veto(scene,camera,depth,rows,depths,nt,len(t),offset)
                remove.update(implicated.tolist());checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
            if (ci+1)%10==0:atomic_json(folder/'progress.json',dict(stage='native_guard',iteration=iteration,cameras=ci+1,unix_time=time.time()))
        rounds.append(dict(removed_triangles=len(remove),checks=checks));print('guard',iteration,'remove',len(remove),flush=True)
        if not remove:break
        take=np.ones(len(t),bool);take[list(remove)]=False
        assert take[:nt].all();ids=ids[take[nt:]];t=t[take]
    if rounds[-1]['removed_triangles']:raise ValueError('Native guard failed to converge')
    candidate=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mv),o3d.utility.Vector3iVector(t));candidate.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(folder/'mesh.ply'),candidate)
    reread=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'));np.testing.assert_array_equal(np.asarray(reread.vertices),mv);np.testing.assert_array_equal(np.asarray(reread.triangles)[:nt],mt[:nt])
    np.savez_compressed(folder/'evidence.npz',seed_ids=seed_ids,seed_votes=votes,seed_poisson_distance=discrepancy,valid_seed_mask=valid,
        query_ids=used,query_normals=normals,certificate=certificate,prior=prior,retained_proposal_ids=ids)
    atomic_json(folder/'certificates.json',dict(notes=notes,seed_neighbors=accepted_neighbors))
    atomic_json(folder/'result.json',dict(request_sha256=sha(folder/'request.json'),verified_seeds=len(seeds),
        certified_vertices=int(certificate.sum()),initially_admitted=int(keep.sum()),added=len(ids),rounds=rounds,
        original_vertices=len(v),original_triangles=nt,original_prefix_exact=True,observed_guard_passed=True,
        hashes={p:sha(folder/p) for p in ['mesh.ply','evidence.npz','certificates.json']},visual_status='pending',production_accepted=False))
    print('interpolated complete',len(ids),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render']);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:render(OUT/'interpolated',FRAME)
