"""Independent replay for the explicitly weaker single-depth seed control."""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from joint_temporal_texture import read,sha,atomic_json
import study_body_neighborhood_completion as base
from study_body_single_depth_seed import ROOT
from body_surface_neighborhood import select_seeds
from local_surface_certificate import certify
from study_jaw_depth_footprint import train_reference_votes
from guard_jaw_measured_depth import measured_pixel_veto
from diffusion_mesh_repair import scene_for


def run(frame):
    root=ROOT/frame;src=base.ROOT/frame;req=read(root/'request.json');result=read(root/'result.json')
    assert result['request_sha256']==sha(root/'request.json')
    assert req['script_sha256']==sha(Path(__file__).with_name('study_body_single_depth_seed.py'))
    assert req['parent_result_sha256']==sha(src/'result.json') and req['parent_admission_sha256']==sha(src/'admission.npz')
    expected=dict(base.SETTINGS);expected['minimum_seed_depth_views']=1;assert req['settings']==expected
    for n,h in result['hashes'].items():assert sha(root/n)==h
    parent=read(src/'result.json')
    for n,h in parent['hashes'].items():assert sha(src/n)==h
    pa=read(src/'independent_audit.json');assert pa['mesh_sha256']==sha(src/'mesh.ply')
    assert pa['semantic_and_sample_depth_evidence_recomputed'] and pa['seed_votes_and_certificates_recomputed']
    rows,depths,hashes,_,_=base.load_inputs(frame);assert hashes==req['source_depth_sha256']
    old=o3d.io.read_triangle_mesh(req['source_mesh']);old.compute_vertex_normals();v=np.asarray(old.vertices);t=np.asarray(old.triangles);nt=len(t)
    p=np.load(src/'proposal.npz');mv=p['vertices'];pp=p['proposals'];e=np.load(src/'admission.npz');new=np.load(root/'certificate_evidence.npz')
    sv,_=train_reference_votes(v[e['seed_ids']],rows,depths);np.testing.assert_array_equal(sv,e['seed_votes'])
    raw=o3d.io.read_triangle_mesh(str(src/'poisson_raw.ply'));raw.compute_vertex_normals()
    rawscene=scene_for(np.asarray(raw.vertices),np.asarray(raw.triangles))
    sp=v[e['seed_ids']];nearest=rawscene.compute_closest_points(o3d.core.Tensor(sp.astype(np.float32)))['points'].numpy()
    discrepancy=np.linalg.norm(sp-nearest,axis=1);valid=(sv>=1)&(discrepancy<=.0005)
    np.testing.assert_array_equal(valid,new['valid_seed_mask']);seeds=sp[valid];sn=np.asarray(old.vertex_normals)[e['seed_ids'][valid]]
    used=e['query_ids'];q=mv[used];normals=e['query_normals'];certificate=[];notes=read(root/'certificates.json')
    for i,ids in enumerate(cKDTree(seeds).query_ball_point(q,.006)):
        ids=np.asarray(ids,int);chosen=ids[select_seeds(q[i],normals[i],seeds[ids],sn[ids])]
        assert chosen.tolist()==notes['seed_neighbors'][i]
        ok,_=certify(q[i],normals[i],seeds[chosen]);certificate.append(ok)
    np.testing.assert_array_equal(certificate,new['certificate']);lookup=np.zeros(len(mv),bool);lookup[used]=certificate
    prior=lookup[pp[e['semantic_ids']]].all(1)&~e['free'].any(axis=(0,2));np.testing.assert_array_equal(prior,new['prior'])
    selected=np.load(root/'retained.npz')['proposal_ids'];assert np.isin(selected,e['semantic_ids'][e['strict']|prior]).all()
    mesh=o3d.io.read_triangle_mesh(str(root/'mesh.ply'));mt=np.asarray(mesh.triangles)
    np.testing.assert_array_equal(np.asarray(mesh.vertices),mv);np.testing.assert_array_equal(mt,np.concatenate([t,pp[selected]]))
    scene=scene_for(mv,mt);checks=[]
    for row,depth in zip(rows,depths):
        for offset in [0,.5]:
            ids,n,r=measured_pixel_veto(scene,row,depth,rows,depths,nt,len(mt),offset)
            assert not len(ids) and not n
            checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=n,raw_far_pixels=r))
    _,components,_=mesh.cluster_connected_triangles()
    atomic_json(root/'independent_audit.json',dict(script_sha256=sha(__file__),mesh_sha256=sha(root/'mesh.ply'),
        parent_full_evidence_audit_sha256=sha(src/'independent_audit.json'),seed_depth_votes_recomputed=True,
        certificates_and_assembly_recomputed=True,original_prefix_exact=True,native_checks=checks,
        components=len(components),nonmanifold_edges=len(mesh.get_non_manifold_edges(allow_boundary_edges=True)),
        production_accepted=False))
    print(frame,'single-depth control independently replayed;',len(checks),'native checks',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001029','001033','001037'],required=True)
    run(p.parse_args().frame)
