"""Recompute observed anchors, certificates and final native-ray safety."""
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree
from joint_temporal_texture import read,sha,atomic_json
from study_poisson_jaw_completion import OUT,SOURCE,FRAME
from study_confidence_depth_prior import load_real
from study_jaw_depth_footprint import train_reference_votes
from local_surface_certificate import certify
from guard_jaw_measured_depth import measured_pixel_veto
from diffusion_mesh_repair import scene_for


def run():
    folder=OUT/'interpolated'/FRAME;req=read(folder/'request.json');result=read(folder/'result.json')
    assert sha(folder/'request.json')==result['request_sha256']
    assert sha(req['source_mesh'])==req['source_mesh_sha256']
    for p,h in result['hashes'].items():assert sha(folder/p)==h
    assert sha(Path(__file__).with_name('study_interpolated_poisson_jaw.py'))==req['script_sha256']
    assert sha(Path(__file__).with_name('local_surface_certificate.py'))==req['helper_sha256']
    bq=read(SOURCE/'request.json');rows,depths,receipt=load_real(Path(bq['depth_root']),FRAME);assert receipt==bq['depth_receipt']
    old=o3d.io.read_triangle_mesh(req['source_mesh']);old.compute_vertex_normals();v=np.asarray(old.vertices);nt=len(old.triangles)
    raw=o3d.io.read_triangle_mesh(str(OUT/'poisson_raw.ply'));raw.compute_vertex_normals()
    raw_scene=scene_for(np.asarray(raw.vertices),np.asarray(raw.triangles))
    local=o3d.io.read_triangle_mesh(str(OUT/'local_raw.ply'));mv=np.asarray(local.vertices)
    evidence=np.load(folder/'evidence.npz');pe=np.load(OUT/'proposal_evidence.npz');admission=np.load(OUT/'admission/samples.npz')
    pp=pe['proposals'][admission['semantic_ids']];used=np.unique(pp);np.testing.assert_array_equal(used,evidence['query_ids']);q=mv[used]
    normals=np.asarray(raw.vertex_normals)[pe['raw_vertex_ids'][used-len(v)]];np.testing.assert_allclose(normals,evidence['query_normals'],atol=1e-12,rtol=0)
    seed_ids=np.unique(np.concatenate([np.asarray(n,int) for n in cKDTree(v).query_ball_point(q,.003)]))
    np.testing.assert_array_equal(seed_ids,evidence['seed_ids'])
    votes,_=train_reference_votes(v[seed_ids],rows,depths);np.testing.assert_array_equal(votes,evidence['seed_votes'])
    nearest=raw_scene.compute_closest_points(o3d.core.Tensor(v[seed_ids].astype(np.float32)))['points'].numpy()
    discrepancy=np.linalg.norm(v[seed_ids]-nearest,axis=1);np.testing.assert_allclose(discrepancy,evidence['seed_poisson_distance'],atol=1e-7,rtol=0)
    valid=(votes>=3)&(discrepancy<=.0005);np.testing.assert_array_equal(valid,evidence['valid_seed_mask'])
    seeds=v[seed_ids[valid]];sn=np.asarray(old.vertex_normals)[seed_ids[valid]]
    distances,neighbors=cKDTree(seeds).query(q,k=min(24,len(seeds)));certificate=np.zeros(len(q),bool)
    for i in range(len(q)):
        ids=neighbors[i][(distances[i]<=.003)&((sn[neighbors[i]]@normals[i])>=.5)]
        certificate[i],_=certify(q[i],normals[i],seeds[ids])
    np.testing.assert_array_equal(certificate,evidence['certificate']);lookup=np.zeros(len(mv),bool);lookup[used]=certificate
    prior=lookup[pp].all(1)&~admission['free'].any(axis=(0,2));np.testing.assert_array_equal(prior,evidence['prior'])
    eligible=admission['semantic_ids'][admission['strict']|prior];retained=evidence['retained_proposal_ids'];assert np.isin(retained,eligible).all()
    mesh=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'));mt=np.asarray(mesh.triangles)
    np.testing.assert_array_equal(np.asarray(mesh.vertices),mv);np.testing.assert_array_equal(mt,np.concatenate([np.asarray(old.triangles),pe['proposals'][retained]]))
    checks=[];scene=scene_for(mv,mt)
    for camera,depth in zip(rows,depths):
        for offset in [0,.5]:
            ids,count,raw_count=measured_pixel_veto(scene,camera,depth,rows,depths,nt,len(mt),offset)
            assert not len(ids) and not count
            checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
    _,components,_=mesh.cluster_connected_triangles()
    atomic_json(folder/'audit.json',dict(script_sha256=sha(__file__),mesh_sha256=sha(folder/'mesh.ply'),
        seed_depth_votes_recomputed=True,local_certificates_recomputed=True,original_prefix_exact=True,native_ray_checks=checks,
        components=len(components),nonmanifold_edges=len(mesh.get_non_manifold_edges(allow_boundary_edges=True)),
        reconstruction_is_an_inferred_prior=True))
    print('Replayed',len(seed_ids),'seed queries,',len(q),'certificates,124 native safety checks',flush=True)


if __name__=='__main__':run()
