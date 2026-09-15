"""Replay, depth-guard and render the local silhouette head prior, opt-in only."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from PIL import Image
from skimage.measure import marching_cubes
from scipy.spatial import cKDTree
from joint_temporal_texture import read,sha,atomic_json,cameras,project
from probe_head_silhouette_completion import ROOT,SETTINGS,INSET,MASKS,MOVIE,FRAMES
from transfer_close_boundary_completion import SOURCE
from local_silhouette_volume import combine_silhouettes,remove_box_caps
from silhouette_domain_surface import stable_domain_faces
from study_jaw_repair_transfer import mask_votes
from study_confidence_depth_prior import load_real,REGIONS
from guard_jaw_measured_depth import measured_pixel_veto
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from review_jaw_repair_transfer import panel,verified_image


def verify(frame):
    root=ROOT/frame;q=read(root/'request.json');r=read(root/'result.json')
    assert r['request_sha256']==sha(root/'request.json') and not r['production_updated']
    for record in [q['scripts'],q['dependencies']]:
        for p,h in record.items():assert sha(p)==h,p
    assert sha(q['source_mesh'])==q['source_mesh_sha256']
    for p,h in r['hashes'].items():assert sha(root/p)==h,p
    return root,q,r


def audit(frame):
    root,q,result=verify(frame);g=np.load(root/'field.npz');e=np.load(root/'evidence.npz')
    field,bits,available=g['field'],g['bits'],g['available'];lower=g['lower'];spacing=float(g['spacing'])
    fields=np.load(root/'silhouette_fields.npz')['fields'];rows=q['rows'];shape=field.shape
    # Independent CPU bilinear sampler at random and near-surface grid points.
    rng=np.random.default_rng(17);flat=rng.choice(field.size,2048,replace=False)
    near=np.flatnonzero(abs(field.ravel())<2);assert len(near)>0
    flat=np.unique(np.r_[flat,rng.choice(near,min(2048,len(near)),replace=False)])
    ijk=np.stack(np.unravel_index(flat,shape),1);points=lower+ijk*spacing;uv,z=project(points,rows)
    cpu,n,_=combine_silhouettes(uv,z,fields,[np.ones(f.shape,bool) for f in fields],3,-SETTINGS['inward_margin_pixels'])
    np.testing.assert_allclose(cpu,field.ravel()[flat],atol=.003,rtol=1e-5)
    expected_bits=np.zeros(len(flat),np.uint64)
    for ci,(xy,depth) in enumerate(zip(uv,z)):
        known=np.isfinite(xy).all(1)&np.isfinite(depth)&(depth>0)&(xy[:,0]>=0)&(xy[:,0]<=1919)&(xy[:,1]>=0)&(xy[:,1]<=1079)
        expected_bits[known]|=np.uint64(1<<ci)
    np.testing.assert_array_equal(bits.ravel()[flat],expected_bits);np.testing.assert_array_equal(n,available.ravel()[flat])
    rv,rt,_,_=marching_cubes(field,0,spacing=(spacing,)*3);rv=rv.astype(np.float64)+lower
    np.testing.assert_array_equal(rv,e['raw_vertices']);np.testing.assert_array_equal(rt,e['raw_triangles'])
    box=remove_box_caps(rv,rt,lower,q['upper'],spacing);domain=stable_domain_faces(rv,rt,lower,spacing,bits,3)
    np.testing.assert_array_equal(box,e['box_pass']);np.testing.assert_array_equal(domain,e['domain_pass'])
    original=o3d.io.read_triangle_mesh(q['source_mesh']);ov,ot=np.asarray(original.vertices),np.asarray(original.triangles)
    originalscene=scene_for(ov,ot);closest=originalscene.compute_closest_points(o3d.core.Tensor(rv.astype(np.float32)))
    distance=np.linalg.norm(rv-closest['points'].numpy(),axis=1)
    edges,counts=np.unique(np.sort(ot[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    bd=cKDTree(ov[np.unique(edges[counts==1])]).query(rv)[0]
    np.testing.assert_array_equal(distance,e['distance']);np.testing.assert_array_equal(bd,e['boundary_distance'])
    local=(distance<=.006)&(bd<=.006)&(rv[:,0]>-.03)
    edge=np.linalg.norm(rv[rt]-rv[rt[:,[1,2,0]]],axis=2).max(1)
    ids=np.flatnonzero(box&domain&local[rt].all(1)&(edge<=.0015));np.testing.assert_array_equal(ids,e['proposals'])
    masks=np.load(MASKS/frame/'masks.npz')['masks'];names=read(MASKS/frame/'cameras.json')
    s,o=mask_votes(rv,rt[ids],rows,masks,names);np.testing.assert_array_equal(s,e['mask_support']);np.testing.assert_array_equal(o,e['mask_outside'])
    kept=ids[(s>=2)&(o==0)];np.testing.assert_array_equal(kept,e['retained_raw_triangle_ids'])
    mesh=o3d.io.read_triangle_mesh(str(root/'unchecked_mesh.ply'));v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    np.testing.assert_array_equal(v,np.concatenate([ov,rv]));np.testing.assert_array_equal(t,np.concatenate([ot,rt[kept]+len(ov)]))
    native=next(r for r in rows if r['physical_camera']==REGIONS[frame]['camera'])
    moving=next(r['camera'] for r in read(MOVIE/'request.json')['inventory'] if r['frame_id']==frame)
    for name,cam in [('native_train',native),('moving',moving)]:
        saved=np.load(root/(name+'_depth.npz'))
        np.testing.assert_array_equal(camera_depth(originalscene,cam)[0],saved['baseline'])
        d,hit,_=camera_depth(scene_for(v,t),cam)
        np.testing.assert_array_equal(d,saved['candidate']);np.testing.assert_array_equal(hit,saved['triangle_ids'])
    atomic_json(root/'proposal_audit.json',dict(request_sha256=sha(root/'request.json'),result_sha256=sha(root/'result.json'),
        cpu_field_samples=len(flat),maximum_field_error=float(abs(cpu-field.ravel()[flat]).max()),
        availability_bits_replayed=True,marching_cubes_and_locality_replayed=True,mask_votes_replayed=True,
        original_prefix_exact=True,fresh_raycasts=4,measured_guard_passed=False,script_sha256=sha(__file__)))
    print(frame,'proposal replay passed; CPU field max error',float(abs(cpu-field.ravel()[flat]).max()),flush=True)


def guard(frame):
    root,q,r=verify(frame);a=read(root/'proposal_audit.json');assert a['result_sha256']==sha(root/'result.json')
    dest=root/'guarded';dest.mkdir(exist_ok=False)
    base=read(SOURCE/frame/'request.json');rows,depths,receipt=load_real(Path(base['depth_root']),frame);assert receipt==base['depth_receipt']
    original=o3d.io.read_triangle_mesh(q['source_mesh']);nt=len(original.triangles)
    mesh=o3d.io.read_triangle_mesh(str(root/'unchecked_mesh.ply'));v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    np.testing.assert_array_equal(t[:nt],np.asarray(original.triangles))
    atomic_json(dest/'request.json',dict(frame=frame,source_result_sha256=sha(root/'result.json'),
        proposal_audit_sha256=sha(root/'proposal_audit.json'),depth_receipt=receipt,source_depth_request_sha256=sha(SOURCE/frame/'request.json'),
        rule=dict(offsets=[0.,.5],free_separation=.003,minimum_other_views=3,maximum_rounds=8),
        scripts={str(Path(__file__).resolve().with_name(n)):sha(Path(__file__).with_name(n)) for n in
            [Path(__file__).name,'guard_jaw_measured_depth.py','study_confidence_depth_prior.py']},
        surface_is_inferred_envelope=True,production_updated=False))
    retained=np.arange(len(t)-nt);rounds=[]
    for iteration in range(8):
        scene=scene_for(v,t);remove=set();checks=[]
        for ci,(row,depth) in enumerate(zip(rows,depths)):
            for offset in [0.,.5]:
                ids,count,raw=measured_pixel_veto(scene,row,depth,rows,depths,nt,len(t),offset)
                remove.update(ids.tolist());checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw))
            if (ci+1)%10==0:atomic_json(dest/'progress.json',dict(stage='native_depth_guard',iteration=iteration,cameras=ci+1,unix_time=time.time()))
        rounds.append(dict(removed=len(remove),checks=checks));print(frame,'guard',iteration,'removed',len(remove),flush=True)
        if not remove:break
        keep=np.ones(len(t),bool);keep[list(remove)]=False;assert keep[:nt].all()
        retained=retained[keep[nt:]];t=t[keep]
    passed=not rounds[-1]['removed']
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));mesh.compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(dest/'mesh.ply'),mesh)
    np.savez_compressed(dest/'evidence.npz',retained_candidate_triangle_ids=retained)
    atomic_json(dest/'result.json',dict(request_sha256=sha(dest/'request.json'),added=len(retained),rounds=rounds,
        original_prefix_exact=True,native_free_space_guard_passed=passed,production_updated=False,visual_status='pending',
        hashes={p:sha(dest/p) for p in ['mesh.ply','evidence.npz']}))
    assert passed,'Guard not converged: do not accept'


def audit_guard(frame):
    root,q,r=verify(frame);dest=root/'guarded';g=read(dest/'result.json');gq=read(dest/'request.json')
    assert g['native_free_space_guard_passed'] and g['request_sha256']==sha(dest/'request.json')
    for p,h in gq['scripts'].items():assert sha(p)==h
    for p,h in g['hashes'].items():assert sha(dest/p)==h
    base=read(SOURCE/frame/'request.json');rows,depths,receipt=load_real(Path(base['depth_root']),frame);assert receipt==gq['depth_receipt']
    old=o3d.io.read_triangle_mesh(q['source_mesh']);raw=o3d.io.read_triangle_mesh(str(root/'unchecked_mesh.ply'))
    mesh=o3d.io.read_triangle_mesh(str(dest/'mesh.ply'));v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles);nt=len(old.triangles)
    np.testing.assert_array_equal(v,np.asarray(raw.vertices));np.testing.assert_array_equal(t[:nt],np.asarray(old.triangles))
    retained=np.load(dest/'evidence.npz')['retained_candidate_triangle_ids']
    np.testing.assert_array_equal(t[nt:],np.asarray(raw.triangles)[nt:][retained]);scene=scene_for(v,t);checks=[]
    for ci,(row,depth) in enumerate(zip(rows,depths)):
        for offset in [0.,.5]:
            ids,count,rawcount=measured_pixel_veto(scene,row,depth,rows,depths,nt,len(t),offset)
            assert not len(ids) and count==0
            checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=rawcount))
        if (ci+1)%16==0:print(frame,'fresh guard audit',ci+1,flush=True)
    atomic_json(dest/'audit.json',dict(result_sha256=sha(dest/'result.json'),mesh_sha256=sha(dest/'mesh.ply'),
        original_prefix_exact=True,retained_assembly_replayed=True,checks=checks,script_sha256=sha(__file__),production_updated=False))


def render(frame):
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    implementation=install();engine.torch.set_num_threads(2);parent=engine.verify_request(MOVIE)
    root=ROOT/frame;g=read(root/'guarded'/'result.json');a=read(root/'guarded'/'audit.json')
    assert g['native_free_space_guard_passed'] and a['result_sha256']==sha(root/'guarded'/'result.json')
    mesh=root/'guarded'/'mesh.ply';assert sha(mesh)==a['mesh_sha256'];rows,_,_=cameras(frame)
    for view in ['moving','native_unmasked']:
        q=deepcopy(parent);q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame];entry=q['inventory'][0]
        if view=='native_unmasked':
            name=REGIONS[frame]['camera'];target=deepcopy(next(r for r in rows if r['physical_camera']==name))
            target['physical_camera']='diagnostic_unmasked_target_'+name;target['reference_physical_camera']=name
            entry['camera']=target
        entry.update(mesh=str(mesh),mesh_sha256=sha(mesh))
        q.update(partial_diagnostic_only=True,full_video_candidate=False,geometry_changed=True,
            silhouette_head_prior=True,geometry_result_sha256=sha(root/'guarded'/'result.json'),
            geometry_audit_sha256=sha(root/'guarded'/'audit.json'),source_quality_implementation_sha256=implementation,
            native_target_mask_disabled=view=='native_unmasked',texture_source_masks_unchanged=True,production_promoted=False)
        q['script_hashes'][Path(__file__).name]=sha(__file__)
        out=ROOT/'rgb'/frame/view;out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
        if (out/'request.json').exists():assert read(out/'request.json')==q
        atomic_json(out/'request.json',q);engine.render(out,[frame])


def review():
    records=[]
    for frame in FRAMES:
        for view in ['moving','native_unmasked']:
            baseline=MOVIE if view=='moving' else INSET/'rgb'/frame/view/'baseline'
            roots=[baseline,INSET/'rgb'/frame/view/'completion',ROOT/'rgb'/frame/view]
            images=[];rs=[];depths=[]
            for root in roots:
                image,r=verified_image(root,frame);images.append(image);rs.append(r)
                depths.append(np.rot90(np.load(root/'frames'/frame/'target_depth.npz')['depth']))
            for r in rs[1:]:
                for k in ['camera','source_cameras','fixed_exposure']:assert rs[0][k]==r[k]
            old,new=depths[0]>0,depths[2]>0
            records.append(dict(frame=frame,view=view,gained_depth=int((~old&new).sum()),lost_depth=int((old&~new).sum()),
                changed_rgb=int(np.any(images[0]!=images[2],2).sum()),
                removed_black=int(((images[0].max(2)==0)&(images[2].max(2)>0)).sum()),
                introduced_black=int(((images[0].max(2)>0)&(images[2].max(2)==0)).sum()),counts_not_anatomical_metrics=True))
            names=['original','inset prior','silhouette prior']
            if view=='native_unmasked':
                gt=INSET/'review'/frame/'native_unmasked_gt.png'
                assert sha(gt)==read(INSET/'artifact_manifest.json')['hashes'][str(gt)]
                images.insert(0,np.asarray(Image.open(gt)));names.insert(0,'train GT')
            for part,box in [('crown',(170,450,970,850)),('jaw',(400,850,900,1250))]:
                panel(ROOT/'review'/frame/(view+'_'+part+'.png'),images,names,box)
    atomic_json(ROOT/'review'/'result.json',dict(records=records,script_sha256=sha(__file__),visual_status='pending',
        production_promoted=False,quality_metrics=False));print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['audit','guard','audit_guard','render','review'])
    p.add_argument('--frame',choices=FRAMES);a=p.parse_args()
    if a.action=='review':review()
    else:
        if not a.frame:p.error('--frame required')
        globals()[a.action](a.frame)
