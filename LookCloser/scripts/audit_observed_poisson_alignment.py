"""Replay bounded deformation, final direct support and all native ray vetoes."""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
import study_body_neighborhood_completion as base
from align_poisson_to_measured_depth import ROOT,SETTINGS,depth_equations
from calibrated_depth_witness import load_images
from regularized_depth_displacement import solve_displacement
from diffusion_mesh_repair import scene_for
from study_jaw_repair_transfer import mask_votes
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_train_confidence import footprint_veto
from diagnose_jaw_measured_depth import barycentric_samples
from guard_jaw_measured_depth import initial_admission,measured_pixel_veto


def run(frame):
    out=ROOT/frame;src=base.ROOT/frame;req=read(out/'request.json');result=read(out/'result.json')
    assert req['settings']==SETTINGS and result['request_sha256']==sha(out/'request.json')
    assert req['parent_result_sha256']==sha(src/'result.json') and req['parent_request_sha256']==sha(src/'request.json')
    for n,h in req['scripts'].items():assert sha(Path(__file__).with_name(n))==h
    for n,h in result['hashes'].items():assert sha(out/n)==h
    rows,depths,hashes,_,maskroot=base.load_inputs(frame);assert hashes==req['source_depth_sha256']
    images,_,receipt=load_images(frame);assert receipt==req['rgb_receipt']
    old=o3d.io.read_triangle_mesh(req['source_mesh']);v=np.asarray(old.vertices);t=np.asarray(old.triangles);nt=len(t)
    p=np.load(src/'proposal.npz');e=np.load(src/'admission.npz');a=np.load(out/'alignment.npz')
    mv=p['vertices'].copy();pp=p['proposals'][e['semantic_ids']];used=np.unique(pp);initial=mv[used].copy();q=initial.copy()
    np.testing.assert_array_equal(used,a['query_ids'])
    index=np.full(len(mv),-1,int);index[used]=np.arange(len(used))
    edges=np.unique(np.sort(index[pp][:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0);np.testing.assert_array_equal(edges,a['edges'])
    for iteration in range(3):
        stage=np.load(out/('iteration_%d.npz'%iteration));np.testing.assert_allclose(q,stage['before'],rtol=0,atol=1e-12)
        h,g,count,_=depth_equations(q,rows,depths,images)
        np.testing.assert_allclose(h,stage['hessian'],rtol=0,atol=1e-12);np.testing.assert_allclose(g,stage['rhs'],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(count,stage['count']);delta,_=solve_displacement(h,g,edges)
        np.testing.assert_allclose(delta,stage['delta'],rtol=0,atol=1e-12)
        q=q+delta;shift=q-initial;q=initial+shift*np.minimum(1,.006/np.maximum(np.linalg.norm(shift,axis=1),1e-20))[:,None]
        np.testing.assert_allclose(q,stage['after'],rtol=0,atol=1e-12)
    assert np.linalg.norm(q-initial,axis=1).max()<=.006+1e-14
    mv[used]=q;np.testing.assert_allclose(mv,a['vertices'],rtol=0,atol=1e-12)
    oldp=p['vertices'][pp];newp=mv[pp]
    oldn=np.cross(oldp[:,1]-oldp[:,0],oldp[:,2]-oldp[:,0]);newn=np.cross(newp[:,1]-newp[:,0],newp[:,2]-newp[:,0])
    cosine=(oldn*newn).sum(1)/np.maximum(np.linalg.norm(oldn,axis=1)*np.linalg.norm(newn,axis=1),1e-20)
    edge=np.linalg.norm(newp-newp[:,[1,2,0]],axis=2).max(1)
    closest=scene_for(v,t).compute_closest_points(o3d.core.Tensor(q.astype(np.float32)))['points'].numpy()
    proximity=np.zeros(len(mv),bool);proximity[used]=np.linalg.norm(q-closest,axis=1)<=.006
    masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json');ms,mo=mask_votes(mv,pp,rows,masks,names)
    safe=(cosine>=.25)&(edge<=.0015)&(newp[...,0]<-.03).all(1)&proximity[pp].all(1)&(ms>=2)&(mo==0)
    np.testing.assert_array_equal(safe,a['safe']);np.testing.assert_array_equal(ms,a['mask_support']);np.testing.assert_array_equal(mo,a['mask_outside'])
    selected=e['semantic_ids'][safe];np.testing.assert_array_equal(selected,a['selected_proposal_ids'])
    points=barycentric_samples(mv[p['proposals'][selected]])
    votes,refs=train_reference_votes(points.reshape(-1,3),rows,depths);free=footprint_veto(points.reshape(-1,3),rows,depths).reshape(62,-1,10)
    np.testing.assert_array_equal(votes.reshape(-1,10),a['votes']);np.testing.assert_array_equal(refs.reshape(-1,10),a['references']);np.testing.assert_array_equal(free,a['free'])
    admitted=initial_admission(votes.reshape(-1,10),free,ms[safe],mo[safe]);np.testing.assert_array_equal(admitted,a['admitted'])
    retained=np.load(out/'retained.npz')['proposal_ids'];assert np.isin(retained,selected[admitted]).all()
    mesh=o3d.io.read_triangle_mesh(str(out/'mesh.ply'));mt=np.asarray(mesh.triangles)
    np.testing.assert_allclose(np.asarray(mesh.vertices),mv,rtol=0,atol=1e-12);np.testing.assert_array_equal(mt,np.concatenate([t,p['proposals'][retained]]))
    np.testing.assert_array_equal(np.asarray(mesh.vertices)[:len(v)],v);np.testing.assert_array_equal(mt[:nt],t)
    scene=scene_for(np.asarray(mesh.vertices),mt);checks=[]
    for row,depth in zip(rows,depths):
        for offset in [0,.5]:
            ids,n,r=measured_pixel_veto(scene,row,depth,rows,depths,nt,len(mt),offset);assert not len(ids) and not n
            checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=n,raw_far_pixels=r))
    _,components,_=mesh.cluster_connected_triangles()
    atomic_json(out/'independent_audit.json',dict(script_sha256=sha(__file__),mesh_sha256=sha(out/'mesh.ply'),
        depth_equations_and_solves_recomputed=True,direct_admission_recomputed=True,original_prefix_exact=True,
        maximum_total_displacement=float(np.linalg.norm(q-initial,axis=1).max()),native_checks=checks,
        components=len(components),nonmanifold_edges=len(mesh.get_non_manifold_edges(allow_boundary_edges=True)),production_accepted=False))
    print(frame,'deformation and direct support replayed;',len(checks),'native checks',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001029','001033','001037'],required=True)
    run(p.parse_args().frame)
