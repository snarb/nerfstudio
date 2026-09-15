"""Replay body-prior evidence and newly added surface safety from native depths."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from joint_temporal_texture import read, sha, atomic_json
from study_body_neighborhood_completion import ROOT, SETTINGS, load_inputs
from body_surface_neighborhood import select_seeds
from local_surface_certificate import certify
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_train_confidence import footprint_veto
from study_jaw_repair_transfer import mask_votes
from diagnose_jaw_measured_depth import barycentric_samples
from guard_jaw_measured_depth import initial_admission, measured_pixel_veto
from diffusion_mesh_repair import scene_for


def run(frame):
    root = ROOT / frame; req = read(root/'request.json'); result = read(root/'result.json')
    assert req['settings'] == SETTINGS and result['request_sha256'] == sha(root/'request.json')
    for name, digest in req['scripts'].items(): assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in result['hashes'].items(): assert sha(root/name) == digest
    assert sha(req['source_mesh']) == req['source_mesh_sha256']
    rows, depths, hashes, _, maskroot = load_inputs(frame); assert hashes == req['source_depth_sha256']
    old = o3d.io.read_triangle_mesh(req['source_mesh']); old.compute_vertex_normals(); old.compute_triangle_normals()
    v = np.asarray(old.vertices); t = np.asarray(old.triangles); nt = len(t)
    raw = o3d.io.read_triangle_mesh(str(root/'poisson_raw.ply')); raw.compute_vertex_normals()
    rv = np.asarray(raw.vertices); rt = np.asarray(raw.triangles)
    proposal = np.load(root/'proposal.npz'); mv = proposal['vertices']; pp = proposal['proposals']
    np.testing.assert_array_equal(mv[:len(v)], v)
    np.testing.assert_array_equal(mv[len(v):], rv[proposal['raw_vertex_ids']])
    np.testing.assert_array_equal(mv[pp], rv[rt[proposal['raw_triangle_ids']]])
    assert (mv[pp][...,0] < SETTINGS['body_max_x']).all()
    assert np.linalg.norm(mv[pp]-mv[pp[:,[1,2,0]]],axis=2).max() <= .0015
    closest = scene_for(v,t).compute_closest_points(o3d.core.Tensor(mv[len(v):].astype(np.float32)))
    cp = closest['points'].numpy()
    np.testing.assert_allclose(cp, proposal['closest_points'], rtol=0, atol=1e-7)
    assert np.linalg.norm(mv[len(v):]-cp, axis=1).max() <= .006
    e = np.load(root/'admission.npz'); semantic = e['semantic_ids']; active = pp[semantic]
    masks = np.load(maskroot/'masks.npz')['masks']; names = read(maskroot/'cameras.json')
    ms, mo = mask_votes(mv, pp, rows, masks, names)
    np.testing.assert_array_equal(ms,e['mask_support']); np.testing.assert_array_equal(mo,e['mask_outside'])
    np.testing.assert_array_equal(semantic,np.flatnonzero((ms>=2)&(mo==0)))
    points = barycentric_samples(mv[active]); votes, refs = train_reference_votes(points.reshape(-1,3),rows,depths)
    free = footprint_veto(points.reshape(-1,3),rows,depths).reshape(62,-1,10)
    np.testing.assert_array_equal(votes.reshape(-1,10),e['votes']); np.testing.assert_array_equal(refs.reshape(-1,10),e['references'])
    np.testing.assert_array_equal(free,e['free'])
    strict=initial_admission(votes.reshape(-1,10),free,ms[semantic],mo[semantic]); np.testing.assert_array_equal(strict,e['strict'])
    used = np.unique(active); q = mv[used]; np.testing.assert_array_equal(used,e['query_ids'])
    normals=np.asarray(raw.vertex_normals)[proposal['raw_vertex_ids'][used-len(v)]]
    np.testing.assert_allclose(normals,e['query_normals'],rtol=0,atol=1e-12)
    seed_ids=np.unique(np.concatenate([np.asarray(x,int) for x in cKDTree(v).query_ball_point(q,.006)]))
    np.testing.assert_array_equal(seed_ids,e['seed_ids'])
    sv,_=train_reference_votes(v[seed_ids],rows,depths); np.testing.assert_array_equal(sv,e['seed_votes'])
    nearest=scene_for(rv,rt).compute_closest_points(o3d.core.Tensor(v[seed_ids].astype(np.float32)))['points'].numpy()
    discrepancy=np.linalg.norm(v[seed_ids]-nearest,axis=1)
    np.testing.assert_allclose(discrepancy,e['seed_poisson_distance'],rtol=0,atol=1e-7)
    valid=(sv>=3)&(discrepancy<=.0005);np.testing.assert_array_equal(valid,e['valid_seed_mask'])
    seeds=v[seed_ids[valid]]; sn=np.asarray(old.vertex_normals)[seed_ids[valid]]; certificates=[]
    recorded=read(root/'certificates.json')
    for i,ids in enumerate(cKDTree(seeds).query_ball_point(q,.006)):
        ids=np.asarray(ids,int); chosen=ids[select_seeds(q[i],normals[i],seeds[ids],sn[ids])]
        assert chosen.tolist()==recorded['seed_neighbors'][i]
        ok,_=certify(q[i],normals[i],seeds[chosen]);certificates.append(ok)
    np.testing.assert_array_equal(certificates,e['certificate'])
    lookup=np.zeros(len(mv),bool);lookup[used]=certificates
    prior=lookup[active].all(1)&~free.any(axis=(0,2));np.testing.assert_array_equal(prior,e['prior'])
    retained=np.load(root/'retained.npz')['proposal_ids'];assert np.isin(retained,semantic[strict|prior]).all()
    mesh=o3d.io.read_triangle_mesh(str(root/'mesh.ply')); mt=np.asarray(mesh.triangles)
    np.testing.assert_array_equal(np.asarray(mesh.vertices),mv);np.testing.assert_array_equal(mt,np.concatenate([t,pp[retained]]))
    current=scene_for(mv,mt); checks=[]
    for row,depth in zip(rows,depths):
        for offset in [0,.5]:
            ids,count,raw_count=measured_pixel_veto(current,row,depth,rows,depths,nt,len(mt),offset)
            assert not len(ids) and not count
            checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
    _,component_sizes,_=mesh.cluster_connected_triangles()
    atomic_json(root/'independent_audit.json',dict(script_sha256=sha(__file__),mesh_sha256=sha(root/'mesh.ply'),
        original_prefix_exact=True,semantic_and_sample_depth_evidence_recomputed=True,
        seed_votes_and_certificates_recomputed=True,native_checks=checks,
        components=len(component_sizes),nonmanifold_edges=len(mesh.get_non_manifold_edges(allow_boundary_edges=True)),
        production_accepted=False,raw_poisson_solve_repeated=False))
    print(frame,'replayed samples, seeds, certificates and',len(checks),'native checks',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001029','001033','001037'],required=True)
    run(p.parse_args().frame)
