"""Opt-in bounded deformation of NEW Poisson patches toward measured surfaces."""
import argparse
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
import study_body_neighborhood_completion as base
from calibrated_depth_witness import load_images
from forearm_rgb_witnesses import color_errors
from study_confidence_depth_prior import project_integer,unproject
from regularized_depth_displacement import solve_displacement
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from study_jaw_repair_transfer import mask_votes
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_train_confidence import footprint_veto
from diagnose_jaw_measured_depth import barycentric_samples
from guard_jaw_measured_depth import initial_admission,measured_pixel_veto

ROOT=Path('/mnt/data/dec5_observed_poisson_alignment')
SETTINGS=dict(iterations=3,maximum_observation_distance=.006,reference_limit=8,
    minimum_other_depth_and_color_witnesses=3,minimum_equation_references=2,
    chroma_limit=.04,rgb_limit=.12,prior_weight=.05,smoothness=.2,
    maximum_step=.002,maximum_total_displacement=.006,maximum_edge=.0015,
    final_admission='unchanged direct multiview support; no inferred seed certificate',
    final_guard='unchanged depth-only 62 views, two ray offsets, eight passes')


def depth_equations(points,rows,depths,images):
    n=len(points);observations=[];errors=[]
    for row,depth in zip(rows,depths):
        uv,z=project_integer(row,points);xy=np.rint(uv).astype(int)
        inside=(z>0)&(xy[:,0]>=3)&(xy[:,0]<1917)&(xy[:,1]>=3)&(xy[:,1]<1077)
        ids=np.flatnonzero(inside);observed=np.zeros(n);observed[ids]=depth[xy[ids,1],xy[ids,0]]
        good=inside&np.isfinite(observed)&(observed>0)&(np.abs(observed-z)<=.006)
        observations.append((xy,z,observed));errors.append(np.where(good,np.abs(observed-z),np.inf))
    errors=np.asarray(errors);nearest=np.argsort(errors,axis=0,kind='stable')[:8]
    h=np.zeros((n,3,3));g=np.zeros((n,3));count=np.zeros(n,int);records=[]
    for ci,(row,depth) in enumerate(zip(rows,depths)):
        ids=np.flatnonzero((nearest==ci).any(0)&np.isfinite(errors[ci]))
        if not len(ids):continue
        xy,z,observed=observations[ci];obs=unproject(row,xy[ids,0],xy[ids,1],observed[ids])
        close=np.linalg.norm(obs-points[ids],axis=1)<=.006;ids=ids[close];obs=obs[close]
        chroma,rgb=color_errors(obs,row,rows,depths,images)
        compatible=((chroma<=.04)&(rgb<=.12)).sum(0);selected=ids[compatible>=3]
        direction=-np.asarray(row['transform_matrix'])[:3,2]
        h[selected]+=np.outer(direction,direction);g[selected]+=(observed[selected]-z[selected])[:,None]*direction
        count[selected]+=1
        records.append(dict(camera=row['physical_camera'],proposed=len(ids),accepted=len(selected)))
    denominator=np.maximum(count,1);h/=denominator[:,None,None];g/=denominator[:,None]
    h[count<2]=0;g[count<2]=0
    return h,g,count,records


def prepare(frame):
    src=base.ROOT/frame;out=ROOT/frame;out.mkdir(parents=True,exist_ok=False)
    request=read(src/'request.json');parent=read(src/'result.json')
    for n,h in parent['hashes'].items():assert sha(src/n)==h
    rows,depths,hashes,entry,maskroot=base.load_inputs(frame);assert hashes==request['source_depth_sha256']
    images,_,rgb_receipt=load_images(frame)
    req=dict(frame=frame,source_mesh=request['source_mesh'],source_mesh_sha256=request['source_mesh_sha256'],
        parent_request_sha256=sha(src/'request.json'),parent_result_sha256=sha(src/'result.json'),
        source_depth_sha256=hashes,settings=SETTINGS,rgb_receipt=rgb_receipt,
        scripts={n:sha(Path(__file__).with_name(n)) for n in base.SCRIPTS+[
            'align_poisson_to_measured_depth.py','regularized_depth_displacement.py','calibrated_depth_witness.py',
            'forearm_rgb_witnesses.py','diagnose_forearm_color_witnesses.py']},
        original_geometry_preserved=True,heldout_used=False,virtual_camera_used_in_fit=False,
        prior_is_inferred=True,production_accepted=False)
    atomic_json(out/'request.json',req)
    old=o3d.io.read_triangle_mesh(req['source_mesh']);v=np.asarray(old.vertices);t=np.asarray(old.triangles);nt=len(t)
    p=np.load(src/'proposal.npz');mv=p['vertices'].copy();all_pp=p['proposals'];e=np.load(src/'admission.npz')
    pp=all_pp[e['semantic_ids']];used=np.unique(pp);q=mv[used].copy();initial=q.copy()
    index=np.full(len(mv),-1,int);index[used]=np.arange(len(used))
    edges=np.unique(np.sort(index[pp][:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
    iterations=[]
    for iteration in range(3):
        atomic_json(out/'progress.json',dict(stage='collect_depth_equations',iteration=iteration,unix_time=time.time()))
        before=q.copy();h,g,count,records=depth_equations(q,rows,depths,images)
        delta,stats=solve_displacement(h,g,edges);q=q+delta
        displacement=q-initial;length=np.linalg.norm(displacement,axis=1)
        q=initial+displacement*np.minimum(1,.006/np.maximum(length,1e-20))[:,None]
        np.savez_compressed(out/('iteration_%d.npz'%iteration),before=before,hessian=h,rhs=g,count=count,delta=delta,after=q)
        iterations.append(dict(iteration=iteration,vertices_with_two_references=int((count>=2).sum()),
            observed_reference_counts=records,maximum_total_displacement=float(np.linalg.norm(q-initial,axis=1).max()),**stats))
        print(frame,'iteration',iteration,'constrained',int((count>=2).sum()),'max shift',iterations[-1]['maximum_total_displacement'],flush=True)
    mv[used]=q;original_points=p['vertices'][pp];new_points=mv[pp]
    oldn=np.cross(original_points[:,1]-original_points[:,0],original_points[:,2]-original_points[:,0])
    newn=np.cross(new_points[:,1]-new_points[:,0],new_points[:,2]-new_points[:,0])
    cosine=(oldn*newn).sum(1)/np.maximum(np.linalg.norm(oldn,axis=1)*np.linalg.norm(newn,axis=1),1e-20)
    edge=np.linalg.norm(new_points-new_points[:,[1,2,0]],axis=2).max(1)
    closest=scene_for(v,t).compute_closest_points(o3d.core.Tensor(q.astype(np.float32)))['points'].numpy()
    proximity=np.zeros(len(mv),bool);proximity[used]=np.linalg.norm(q-closest,axis=1)<=.006
    masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json');ms,mo=mask_votes(mv,pp,rows,masks,names)
    safe=(cosine>=.25)&(edge<=.0015)&(new_points[...,0]<-.03).all(1)&proximity[pp].all(1)&(ms>=2)&(mo==0)
    selected=e['semantic_ids'][safe];active=all_pp[selected];points=barycentric_samples(mv[active])
    votes,refs=train_reference_votes(points.reshape(-1,3),rows,depths)
    free=footprint_veto(points.reshape(-1,3),rows,depths).reshape(62,-1,10)
    admitted=initial_admission(votes.reshape(-1,10),free,ms[safe],mo[safe]);chosen=selected[admitted]
    np.savez_compressed(out/'alignment.npz',vertices=mv,query_ids=used,edges=edges,safe=safe,mask_support=ms,mask_outside=mo,
        cosine=cosine,maximum_edge=edge,selected_proposal_ids=selected,votes=votes.reshape(-1,10),references=refs.reshape(-1,10),free=free,admitted=admitted)
    triangles=np.concatenate([t,all_pp[chosen]]);initial_count=len(chosen);rounds=[]
    print(frame,'safe',int(safe.sum()),'direct admitted',initial_count,flush=True)
    for iteration in range(8):
        scene=scene_for(mv,triangles);remove=set();checks=[]
        for ci,(row,depth) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                ids,n,r=measured_pixel_veto(scene,row,depth,rows,depths,nt,len(triangles),offset)
                remove.update(ids.tolist());checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=n,raw_far_pixels=r))
            if (ci+1)%10==0:atomic_json(out/'progress.json',dict(stage='native_guard',iteration=iteration,cameras=ci+1,unix_time=time.time()))
        rounds.append(dict(removed=len(remove),checks=checks));print(frame,'guard',iteration,len(remove),flush=True)
        if not remove:break
        take=np.ones(len(triangles),bool);take[list(remove)]=False;assert take[:nt].all();chosen=chosen[take[nt:]];triangles=triangles[take]
    if rounds[-1]['removed']:raise ValueError('Native guard did not converge')
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mv),o3d.utility.Vector3iVector(triangles));mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out/'mesh.ply'),mesh);np.savez_compressed(out/'retained.npz',proposal_ids=chosen)
    checks=[];scene=scene_for(mv,triangles);oldscene=scene_for(v,t)
    for view,camera in [('moving',entry['camera']),('train_H_A',next(r for r in rows if r['physical_camera']=='H004_A005_1210M6'))]:
        d,ids,_=camera_depth(scene,camera);od,_,_=camera_depth(oldscene,camera);valid=np.isfinite(d)
        light=np.abs(np.asarray(mesh.triangle_normals)@[.3,.4,.866]);rgb=np.zeros((*d.shape,3),np.uint8)
        rgb[valid]=(60+170*light[ids[valid],None]).astype(np.uint8);rgb[valid&(ids>=nt)]=[255,60,40]
        Image.fromarray(np.rot90(rgb)).save(out/(view+'_added.png'))
        checks.append(dict(view=view,newly_visible=int((valid&~np.isfinite(od)).sum()),original_now_missing=int((~valid&np.isfinite(od)).sum())))
    files=['mesh.ply','alignment.npz','retained.npz','moving_added.png','train_H_A_added.png']+['iteration_%d.npz'%i for i in range(3)]
    atomic_json(out/'result.json',dict(request_sha256=sha(out/'request.json'),iterations=iterations,
        initially_admitted=initial_count,added=len(chosen),rounds=rounds,views=checks,observed_guard_passed=True,
        original_vertices=len(v),original_triangles=nt,production_accepted=False,visual_status='pending',
        hashes={n:sha(out/n) for n in files}))
    print(frame,'finished',len(chosen),checks,flush=True)


def render(frame):
    base.ROOT=ROOT;base.SCRIPTS=base.SCRIPTS+['align_poisson_to_measured_depth.py','regularized_depth_displacement.py','calibrated_depth_witness.py']
    base.render(frame)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render'])
    p.add_argument('--frame',choices=['001029','001033','001037'],required=True);a=p.parse_args()
    {'prepare':prepare,'render':render}[a.action](a.frame)
