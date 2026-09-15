"""Bounded coherent reference-grid replacement, not repeated append-only patches.

An opt-in canary: no production/source edits. Measured pins constrain one
inferred height field; the same native contrastive guard checks all new faces.
"""
from pathlib import Path
import argparse,time
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt
from joint_temporal_texture import read,sha,atomic_json
from study_confidence_depth_prior import unproject,project_integer
from forearm_quadric_rays import world_quadric,world_plane,intersect_near_plane
from confidence_boundary_completion import solve_depth,grid_faces
from bounded_surface_replacement import removable_faces,displacement_within_bound
from ordered_forearm_admission import point_votes
from annotation_mask_domain import semantic_faces
from diffusion_mesh_repair import scene_for
import study_forearm_plane_transfer_v3 as prior

BASE=Path('/mnt/data/dec5_forearm_admission_quadric_bounded')
OUT=Path('/mnt/data/dec5_coherent_forearm_replacement')


def prepare(output,frame,feasible_depth_constraints=False,guard_max_rounds=8,preserve_production_surface=False,positive_only_annotations=False):
    if not 1<=guard_max_rounds<=64:raise ValueError('Guard limit must be in 1..64')
    if preserve_production_surface and not feasible_depth_constraints:raise ValueError('Protection control requires constrained surface')
    if positive_only_annotations and not preserve_production_surface:raise ValueError('Partial annotation control requires production protection')
    if feasible_depth_constraints and output.resolve()==OUT.resolve():raise ValueError('Use a separate output for constrained surface study')
    prior.configure();v1=prior.v2.v1;root=prior.OUT/frame;folder=output/frame;folder.mkdir(parents=True,exist_ok=True)
    rows,depths,hashes=v1.load_real(frame);analysis=read(root/'analysis.json');data=np.load(root/'diagnostic.npz')
    if hashes!=analysis['source_depth_sha256']:raise ValueError('Changed depth maps')
    masks=v1.masks(frame);name=v1.NAMES[1];camera=next(r for r in rows if r['physical_camera']==name)
    reference=next(r for r in rows if r['physical_camera']==v1.NAMES[0])
    fitpath=Path('/mnt/data/dec5_forearm_multiview_anchors')/frame/'result.json'
    fit=next(r for r in read(fitpath)['fit'] if r['model']=='quadratic')
    source=BASE/frame/'guarded.ply'
    if sha(source)!=read(BASE/frame/'geometry_result.json')['hashes']['guarded.ply']:raise ValueError('Changed starting mesh')
    helper_names=[Path(__file__).name,'bounded_surface_replacement.py','forearm_quadric_rays.py','confidence_boundary_completion.py',
        'ordered_forearm_admission.py','annotation_mask_domain.py','contrastive_forearm_depth_guard.py','contrastive_forearm_witnesses.py',
        'study_forearm_plane_transfer.py','study_forearm_plane_transfer_v2.py','study_forearm_plane_transfer_v3.py']
    request=dict(frame=frame,source_mesh=str(source),source_mesh_sha256=sha(source),source_depth_sha256=hashes,
        fit_sha256=sha(fitpath),diagnostic_sha256=sha(root/'diagnostic.npz'),reference_camera=camera,fit_reference_camera=reference,
        skin_polygons=v1.POLYGONS[frame],maximum_model_plane_depth_change=.01,maximum_replaced_vertex_depth_distance=.012,
        regularization=.05,minimum_trusted_pins=30,depth_displacement_bound=.012,guard_max_rounds=guard_max_rounds,
        guard_witness_comparison_margin=.01,heldout_used=False,production_changed=False,inferred_surface_not_measured_anatomy=True,
        scripts={n:sha(Path(__file__).with_name(n)) for n in helper_names})
    if feasible_depth_constraints:
        request['feasible_depth_constraints']=dict(solver='discrete_checkerboard_coordinate_descent',sample_count=49,
            offset_bounds=[-.012,.012],step=.0005,initial_and_final_mask_rules=True,plane_bound_applied_to_solved_points=True,
            trusted_constraint_conflicts_excluded_not_moved=True)
        request['scripts'].update({n:sha(Path(__file__).with_name(n)) for n in ['feasible_forearm_surface.py','discrete_surface_constraints.py']})
    if positive_only_annotations:
        request['positive_only_annotations']=dict(outside_inset_roi='unknown, not negative skin evidence',
            minimum_positive_views=2,measured_free_space_guard_unchanged=True,masks_dilated=False)
    if preserve_production_surface:
        parent=Path('/mnt/data/dec5_phase30_early_texture_dynamic_150/request.json')
        production=next(r for r in read(parent)['inventory'] if r['frame_id']==frame)
        if sha(production['mesh'])!=production['mesh_sha256']:raise ValueError('Changed protected production mesh')
        request['protected_production']=dict(mesh=production['mesh'],mesh_sha256=production['mesh_sha256'],
            parent_request_sha256=sha(parent),source_holes_only=True,boundary_depth_from_production=True)
        request['scripts']['protected_production_grid.py']=sha(Path(__file__).with_name('protected_production_grid.py'))
    if (folder/'request.json').exists() and read(folder/'request.json')!=request:raise ValueError('Frozen request mismatch')
    atomic_json(folder/'request.json',request)
    if (folder/'geometry_result.json').exists():
        result=read(folder/'geometry_result.json')
        if result['request_sha256']!=sha(folder/'request.json'):raise ValueError('Changed completed request')
        for n,h in result['hashes'].items():
            if sha(folder/n)!=h:raise ValueError('Changed completed output')
        print(frame,'verified completed',flush=True);return
    y,x=np.nonzero(masks[name]);center=np.asarray(camera['transform_matrix'])[:3,3]
    directions=unproject(camera,x,y,np.ones(len(x)))-center
    z,_=intersect_near_plane(center,directions,world_quadric(reference,fit),world_plane(reference,analysis['plane_inverse_coefficients']))
    valid=np.isfinite(z)&(z>0);x,y,z=x[valid],y[valid],z[valid];points=unproject(camera,x,y,z)
    uv,rz=project_integer(reference,points);inverse=np.column_stack([uv/100,np.ones(len(uv))])@analysis['plane_inverse_coefficients']
    bounded=(inverse>0)&(np.abs(rz-1/np.maximum(inverse,1e-12))<=.01)
    support,negative,free=point_votes(points,rows,v1.NAMES,masks,data,depths,prior.v2.semantic_domain,positive_only_annotations)
    distance=distance_transform_edt(~data[name+'_trusted'])[y,x]
    selected=bounded&(support>=2)&(negative==0)&(free==0)&(distance<=100)
    model=np.zeros(masks[name].shape,float);model[y[selected],x[selected]]=z[selected];domain=model>0
    observed=depths[next(i for i,r in enumerate(rows) if r['physical_camera']==name)]
    pins=domain&data[name+'_trusted']&(np.abs(observed-model)<=.012)
    if pins.sum()<30:raise ValueError('Too few native trusted pins')
    if feasible_depth_constraints:
        from feasible_forearm_surface import build
        domain,model,solved,pins,stats,arrays=build(camera,reference,fit,analysis,rows,depths,masks,data,positive_only_annotations)
        np.savez_compressed(folder/'constraint_samples.npz',**arrays)
    else:
        solved,stats=solve_depth(domain,model,observed,pins,.05)
    if not displacement_within_bound(solved[domain],model[domain],.012):raise ValueError('Unbounded solve displacement')
    original=o3d.io.read_triangle_mesh(str(source));ov,ot=np.asarray(original.vertices),np.asarray(original.triangles)
    uv,oldz=project_integer(camera,ov);removed=removable_faces(uv,oldz,ot,domain,solved,.012)
    retained=ot[~removed].copy();yy,xx=np.nonzero(domain)
    vertices=np.concatenate([ov,unproject(camera,xx,yy,solved[yy,xx])]);index=np.full(domain.shape,-1,int)
    index[yy,xx]=np.arange(len(xx))+len(ov);faces=grid_faces(domain,domain,index)
    if preserve_production_surface:
        from protected_production_grid import assemble
        protected=o3d.io.read_triangle_mesh(request['protected_production']['mesh'])
        vertices,retained,faces,removed,protection_arrays,protection_stats=assemble(ov,ot,np.asarray(protected.vertices),np.asarray(protected.triangles),camera,domain,solved)
        stats['production_protection']=protection_stats
        np.savez_compressed(folder/'protection_evidence.npz',**protection_arrays)
    faces,semantic=semantic_faces(vertices,faces,rows,masks,axis_extent=True,positive_only_annotations=positive_only_annotations);triangles=np.concatenate([retained,faces])
    stats.update(domain_pixels=int(domain.sum()),removed_old_triangles=int(removed.sum()),initial_new_triangles=len(faces),semantic=semantic)
    np.savez_compressed(folder/'evidence.npz',domain=domain,model=model,solved=solved,pins=pins,removed_original_faces=removed)
    def save(name,tt):
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(tt));mesh.compute_vertex_normals()
        if not o3d.io.write_triangle_mesh(str(folder/name),mesh):raise IOError('Mesh write failed')
    save('transferred.ply',triangles)
    from contrastive_forearm_depth_guard import make_guard
    veto,calls,provenance=make_guard(frame,rows,depths,.01);rounds=[]
    for iteration in range(guard_max_rounds):
        scene=scene_for(vertices,triangles);remove=set();checks=[]
        for ci,(row,depth) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                ids,count,raw=veto(scene,row,depth,rows,depths,len(retained),len(triangles),offset)
                remove.update(ids.tolist());checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw))
            if (ci+1)%10==0:
                atomic_json(folder/'progress.json',dict(stage='native_guard',iteration=iteration,cameras=ci+1,flagged=len(remove),unix_time=time.time()))
                print(frame,iteration,ci+1,len(remove),flush=True)
        rounds.append(dict(iteration=iteration,removed_triangles=len(remove),checks=checks))
        if not remove:break
        if min(remove)<len(retained):raise ValueError('Attempted removal of protected old faces')
        keep=np.ones(len(triangles),bool);keep[list(remove)]=False;triangles=triangles[keep]
    save('guarded.ply',triangles)
    atomic_json(folder/'geometry_result.json',dict(request_sha256=sha(folder/'request.json'),stats=stats,rounds=rounds,
        observed_guard_passed=not rounds[-1]['removed_triangles'],retained_original_triangles=len(retained),
        final_added_triangles=len(triangles)-len(retained),color_guard_calls=calls,color_guard_provenance=provenance,
        visual_status='pending',production_accepted=False,
        hashes={n:sha(folder/n) for n in ['transferred.ply','guarded.ply','evidence.npz']+(['constraint_samples.npz'] if feasible_depth_constraints else [])+(['protection_evidence.npz'] if preserve_production_surface else [])}))
    print(frame,'finished',stats,'final',len(triangles)-len(retained),'guard',not rounds[-1]['removed_triangles'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001029','001033','001037'],required=True)
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--feasible-depth-constraints',action='store_true')
    p.add_argument('--guard-max-rounds',type=int,default=8)
    p.add_argument('--preserve-production-surface',action='store_true')
    p.add_argument('--positive-only-annotations',action='store_true')
    a=p.parse_args();prepare(a.output,a.frame,a.feasible_depth_constraints,a.guard_max_rounds,a.preserve_production_surface,a.positive_only_annotations)
