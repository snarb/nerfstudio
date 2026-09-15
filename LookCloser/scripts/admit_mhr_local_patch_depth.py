"""Measured-depth admission of frozen local MHR proposals, CPU-only, no promotion."""
from pathlib import Path
import argparse,time
import numpy as np
from scipy.spatial import cKDTree
from study_multiview_face_prior import read,save,sha
from study_confidence_depth_prior import load_real,unproject
from study_jaw_repair_transfer import mask_votes
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_train_confidence import footprint_veto
from guard_jaw_measured_depth import initial_admission,measured_pixel_veto
from diagnose_jaw_measured_depth import barycentric_samples,observed_at
from local_surface_certificate import certify

OUT=Path('/mnt/data/dec5_mhr_local_patch_admission')
CANDIDATES=Path('/mnt/data/dec5_mhr_local_patch_candidates')
SOURCE=Path('/mnt/data/dec5_jaw_measured_mask_control/001193')
PRIOR=Path('/mnt/data/dec5_mhr_measured_conformance')
FRAME='001193'
ARMS=['smooth025','smooth100','smooth400']
HELPERS=['guard_poisson_jaw_completion.py','study_jaw_repair_transfer.py','study_jaw_depth_footprint.py','study_jaw_train_confidence.py',
    'guard_jaw_measured_depth.py','diagnose_jaw_measured_depth.py','local_surface_certificate.py','study_confidence_depth_prior.py','bake_joint_temporal_mesh.py','joint_temporal_texture.py']

class Scene2:
    """Execution-only wrapper: retain helper ray conventions with two CPU threads."""
    def __init__(self,v,t):
        import open3d as o3d
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=2)
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(v,np.float32)),o3d.core.Tensor(np.asarray(t,np.uint32)))
    def create_rays_pinhole(self,*args,**kwargs):return self.scene.create_rays_pinhole(*args,**kwargs)
    def cast_rays(self,rays):return self.scene.cast_rays(rays,nthreads=2)
    def compute_closest_points(self,points):return self.scene.compute_closest_points(points,nthreads=2)

def interpolation_admission(strict,certificate,proposals,free,mask_support,mask_outside):
    prior=np.asarray(certificate)[proposals].all(1)&~np.asarray(free).any(axis=(0,2))&(mask_support>=2)&(mask_outside==0)
    return np.asarray(strict)|prior,prior

def inputs():
    from joint_temporal_texture import HELD_CAMERAS
    bq=read(SOURCE/'request.json');cq=read(CANDIDATES/'request.json');summary=read(CANDIDATES/'result.json');assert summary['request_sha256']==sha(CANDIDATES/'request.json')
    rows,depths,receipt=load_real(Path(bq['depth_root']),FRAME);assert receipt==bq['depth_receipt'];assert len(rows)==62 and len({r['physical_camera'] for r in rows})==62
    assert not ({r['physical_camera'] for r in rows}&HELD_CAMERAS)
    # Deliberate scope: old SOURCE geometry is different and is NOT inherited.
    assert sha(cq['source_mesh'])==cq['source_mesh_sha256']
    parent_path=Path('/mnt/data/dec5_phase30_early_texture_dynamic_150/request.json');parent=read(parent_path);assert sha(parent_path)==bq['parent_request_sha256'];entry=next(r for r in parent['inventory'] if r['frame_id']==FRAME);maskroot=Path(entry['source_masks']['root'])
    assert sha(maskroot/'masks.npz')==bq['source_mask_sha256'];assert sha(maskroot/'cameras.json')==bq['mask_names_sha256'];masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json');assert not(set(names)&HELD_CAMERAS)
    override=Path(bq['mask_override']['root'])/FRAME;mr=read(override/'result.json');oq=read(override/'request.json');assert not oq['candidate_mesh_used']
    assert sha(override/'result.json')==bq['mask_override']['result_sha256'];assert mr['request_sha256']==sha(override/'request.json');assert mr['original_masks_sha256']==sha(maskroot/'masks.npz');assert oq['depth_receipt']==receipt
    for p,h in mr['hashes'].items():assert sha(override/p)==h
    masks=masks.copy();index=names.index(mr['camera']);replacement=np.load(override/'mask.npy');assert replacement.shape==masks[index].shape and replacement[masks[index].astype(bool)].all();masks[index]=replacement
    binding=dict(source_request_sha256=sha(SOURCE/'request.json'),source_geometry_sha256_not_inherited=bq['source_mesh_sha256'],actual_source_mesh=cq['source_mesh'],actual_source_mesh_sha256=cq['source_mesh_sha256'],
        explicit_geometry_rebinding=True,independent_mask_override_request_sha256=sha(override/'request.json'),mask_override_result_sha256=sha(override/'result.json'),
        mask_path=str(maskroot/'masks.npz'),mask_sha256=bq['source_mask_sha256'],mask_names_sha256=bq['mask_names_sha256'],override_path=str(override/'mask.npy'),override_sha256=sha(override/'mask.npy'),depth_receipt=receipt)
    return cq,rows,depths,masks,names,binding

def certificates(folder,v,proposals,old,prior_v,prior_t,rows,depths):
    import open3d as o3d
    old.compute_vertex_normals();old.compute_triangle_normals();ov=np.asarray(old.vertices);ot=np.asarray(old.triangles);oldscene=Scene2(ov,ot);prior=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(prior_v),o3d.utility.Vector3iVector(prior_t));prior.compute_triangle_normals();scene=Scene2(prior_v,prior_t)
    used=np.unique(proposals);q=v[used];lookup=np.zeros(len(v),bool)
    if not len(q):return lookup,dict(queries=0,observed_seeds=0,certified_vertices=0)
    raw_neighborhood=cKDTree(ov).query_ball_point(q,.003);seed_ids=np.unique(np.concatenate([np.asarray(x,int) for x in raw_neighborhood]));initial_votes,refs=train_reference_votes(ov[seed_ids],rows,depths)
    points=[];camera=[];pixel=[]
    for ci in np.unique(refs[(initial_votes>=3)&(refs>=0)]):
        select=(refs==ci)&(initial_votes>=3);xy,_,d,available=observed_at(ov[seed_ids[select]],rows[ci],depths[ci]);assert available.all();points.extend(unproject(rows[ci],xy[:,0],xy[:,1],d));camera.extend([ci]*len(xy));pixel.extend(xy)
    points=np.asarray(points,float).reshape(-1,3);camera=np.asarray(camera,int);pixel=np.asarray(pixel,int).reshape(-1,2)
    if len(points):
        _,unique=np.unique(np.column_stack((camera,pixel)),axis=0,return_index=True);unique=np.sort(unique);points=points[unique];camera=camera[unique];pixel=pixel[unique]
        votes,_=train_reference_votes(points,rows,depths);nearest=scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)));distance=np.linalg.norm(nearest['points'].numpy()-points,axis=1);original=oldscene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)));old_distance=np.linalg.norm(original['points'].numpy()-points,axis=1);normals=np.asarray(old.triangle_normals)[original['primitive_ids'].numpy()];valid=(votes>=3)&(distance<=.0005)&(old_distance<=.001)
    else:votes=np.zeros(0,np.uint8);distance=np.zeros(0);old_distance=np.zeros(0);normals=np.zeros((0,3));valid=np.zeros(0,bool)
    seeds=points[valid];seed_normals=normals[valid];qh=scene.compute_closest_points(o3d.core.Tensor(q.astype(np.float32)));query_normals=np.asarray(prior.triangle_normals)[qh['primitive_ids'].numpy()];notes=[];accepted_neighbors=[]
    if len(seeds)>=8:
        distances,neighbors=cKDTree(seeds).query(q,k=min(24,len(seeds)))
        for i in range(len(q)):
            take=(distances[i]<=.003)&((seed_normals[neighbors[i]]@query_normals[i])>=.5);selected=neighbors[i][take];lookup[used[i]],note=certify(q[i],query_normals[i],seeds[selected],tolerance=.0005);notes.append(note);accepted_neighbors.append(selected.tolist())
    else:
        notes=[dict(reason='insufficient_verified_observed_seeds') for _ in q];accepted_neighbors=[[] for _ in q]
    np.savez_compressed(folder/'certificates.npz',query_ids=used,query_normals=query_normals,certificate=lookup[used],old_seed_ids=seed_ids,old_seed_initial_votes=initial_votes,
        observed_seed_points=points,observed_seed_camera=camera,observed_seed_pixel=pixel,observed_seed_votes=votes,seed_prior_distance=distance,seed_original_distance=old_distance,valid_seed_mask=valid,seed_normals=normals)
    save(folder/'certificates.json',dict(notes=notes,accepted_seed_neighbors=accepted_neighbors,seed_indices_refer_to_valid_seed_subset=True))
    return lookup,dict(queries=len(q),observed_seeds=int(valid.sum()),certified_vertices=int(lookup.sum()),candidate_observed_seeds=len(points))

def native_guard(folder,v,old_t,proposals,ids,rows,depths):
    import open3d as o3d
    folder.mkdir(exist_ok=False);nt=len(old_t);initial=len(ids);t=np.concatenate((old_t,proposals[ids]));rounds=[]
    for iteration in range(8):
        scene=Scene2(v,t);remove=set();checks=[]
        for ci,(camera,depth) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                implicated,count,raw_count=measured_pixel_veto(scene,camera,depth,rows,depths,nt,len(t),offset);remove.update(implicated.tolist());checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
            if (ci+1)%10==0:save(folder/'progress.json',dict(stage='native_guard',iteration=iteration,cameras=ci+1,unix_time=time.time()))
        rounds.append(dict(removed_triangles=len(remove),checks=checks));print(folder.parent.name,folder.name,'round',iteration,'removed',len(remove),flush=True)
        if not remove:break
        take=np.ones(len(t),bool);take[list(remove)]=False;assert take[:nt].all();ids=ids[take[nt:]];t=t[take]
    if rounds[-1]['removed_triangles']:raise ValueError('Native gate did not converge within8rounds')
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));mesh.compute_vertex_normals();assert o3d.io.write_triangle_mesh(str(folder/'mesh.ply'),mesh)
    reread=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'));np.testing.assert_array_equal(np.asarray(reread.vertices),v);np.testing.assert_array_equal(np.asarray(reread.triangles)[:nt],old_t)
    np.savez_compressed(folder/'evidence.npz',retained_proposal_ids=ids);save(folder/'result.json',dict(initially_admitted=initial,added=len(ids),rounds=rounds,original_prefix_exact=True,observed_guard_passed=True,
        hashes={p:sha(folder/p) for p in ['mesh.ply','evidence.npz']},new_surface_is_inferred=True,visual_status='parent_review_pending',production_accepted=False))
    return dict(initial=initial,final=len(ids),rounds=len(rounds),result_sha256=sha(folder/'result.json'))

def run():
    import open3d as o3d
    started=time.monotonic();cq,rows,depths,masks,names,binding=inputs();OUT.mkdir(exist_ok=False)
    request=dict(frame=FRAME,candidate_request_sha256=sha(CANDIDATES/'request.json'),candidate_result_sha256=sha(CANDIDATES/'result.json'),inputs=binding,arms=ARMS,
        comparisons=['strict','interpolated'],strict_rule='initial_admission: >=2 vertices and median10samples >=2observedviews; masks>=2/noavailabledisagreement; nofootprintveto',
        interpolation=dict(seed_radius=.003,nearest_seeds=24,minimum_observed_seed_views=3,minimum_seeds=8,seed_prior_distance=.0005,seed_original_distance=.001,normal_dot=.5,maximum_loo_p90=.0005,maximum_predicted_offset=.0005,inside_seed_hull=True,all_proposal_vertices_required=True),
        no_nearby_anchor_votes_shortcut=True,final_ray_offsets=[0,.5],final_cameras=62,max_rounds=8,raycast_threads=2,geometry_uses_target=False,heldout_used=False,production_accepted=False,
        former_fit_validation_views_used_for_admission=True,script_sha256=sha(__file__),helpers={n:sha(Path(__file__).with_name(n)) for n in HELPERS})
    save(OUT/'request.json',request);old=o3d.io.read_triangle_mesh(cq['source_mesh']);ov=np.asarray(old.vertices);ot=np.asarray(old.triangles);summaries=[]
    for arm in ARMS:
        source=CANDIDATES/arm;result=read(source/'result.json');assert result['request_sha256']==sha(CANDIDATES/'request.json')
        for p,h in result['hashes'].items():assert sha(source/p)==h
        mesh=o3d.io.read_triangle_mesh(str(source/'local_raw.ply'));v=np.asarray(mesh.vertices);tt=np.asarray(mesh.triangles);proposals=np.load(source/'proposal_evidence.npz')['proposals'];np.testing.assert_array_equal(v[:len(ov)],ov);np.testing.assert_array_equal(tt,np.concatenate((ot,proposals)))
        folder=OUT/arm;folder.mkdir();save(folder/'request.json',dict(parent_request_sha256=sha(OUT/'request.json'),candidate_result_sha256=sha(source/'result.json'),raw_mesh_sha256=sha(source/'local_raw.ply')))
        ms,mo=mask_votes(v,proposals,rows,masks,names);semantic=np.flatnonzero((ms>=2)&(mo==0));pp=proposals[semantic];points=barycentric_samples(v[pp]);votes,refs=train_reference_votes(points.reshape(-1,3),rows,depths);free=footprint_veto(points.reshape(-1,3),rows,depths).reshape(62,-1,10)
        strict=initial_admission(votes.reshape(-1,10),free,ms[semantic],mo[semantic]);print(arm,'semantic',len(semantic),'strict',int(strict.sum()),flush=True)
        prior=np.load(PRIOR/arm/'fit.npz');initial=np.load(PRIOR/'initial.npz');certificate,stats=certificates(folder,v,pp,old,prior['vertices'],initial['triangles'],rows,depths)
        interpolated,certified=interpolation_admission(strict,certificate,pp,free,ms[semantic],mo[semantic])
        np.savez_compressed(folder/'admission.npz',semantic_ids=semantic,points=points,votes=votes.reshape(-1,10),references=refs.reshape(-1,10),trusted_free=free,mask_support=ms,mask_outside=mo,strict=strict,certified_prior=certified,interpolated=interpolated)
        print(arm,'certificates',stats,'interpolated',int(interpolated.sum()),flush=True)
        branches={}
        for name,keep in [('strict',strict),('interpolated',interpolated)]:branches[name]=native_guard(folder/name,v,ot,proposals,semantic[keep],rows,depths)
        record=dict(arm=arm,raw=len(proposals),semantic=len(semantic),strict_initial=int(strict.sum()),certified_additional=int((interpolated&~strict).sum()),certificate=stats,branches=branches,admission_sha256=sha(folder/'admission.npz'),production_accepted=False)
        save(folder/'result.json',record);summaries.append(record)
    save(OUT/'result.json',dict(request_sha256=sha(OUT/'request.json'),arms=summaries,seconds=time.monotonic()-started,production_accepted=False,original_geometry_unchanged=True));print('all arms terminal',time.monotonic()-started,flush=True)

if __name__=='__main__':run()
