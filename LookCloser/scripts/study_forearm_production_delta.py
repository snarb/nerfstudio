"""Transfer the frozen v3 forearm delta while preserving production head/carving.

All 62 native measured-depth views then guard the added geometry. No production
mesh is replaced. Defaults of the model and existing renderer remain unchanged.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import time
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from append_verified_mesh_delta import append_delta
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from guard_jaw_measured_depth import measured_pixel_veto
import study_forearm_plane_transfer as evidence

PRIOR=Path('/mnt/data/dec5_forearm_plane_transfer_v3')
PARENT=Path('/mnt/data/dec5_phase30_dynamic_150')
FRAMES=['001029','001033','001037']


def prepare(root,frame,curve_source=None,boundary_conditioned=False,photometric_free_space=False,matched_plane=False,known_annotation_domain=False,witness_rgb_limit=None,witness_comparison_margin=None,admission_shape=None,observed_ring_feather=False):
    if boundary_conditioned and curve_source is None:raise ValueError('Boundary condition requires curved source')
    if matched_plane and (curve_source is None or not boundary_conditioned):raise ValueError('Matched plane requires boundary-conditioned comparison')
    if known_annotation_domain and (curve_source is None or not boundary_conditioned):raise ValueError('Known annotation domain requires boundary-conditioned comparison')
    if witness_rgb_limit is not None and (not photometric_free_space or not 0<witness_rgb_limit<=1):raise ValueError('RGB witness limit requires photometric guard and valid limit')
    if witness_comparison_margin is not None and (witness_rgb_limit!=.12 or not 0<witness_comparison_margin<1):raise ValueError('Comparison requires RGB limit .12 and positive margin')
    if admission_shape is not None and (admission_shape not in ['plane','quadric'] or curve_source is None or not boundary_conditioned or matched_plane):raise ValueError('Admission order requires boundary-conditioned curved comparison')
    if observed_ring_feather and (curve_source is None or not boundary_conditioned or matched_plane):raise ValueError('Observed-ring feather requires boundary-conditioned curvature')
    folder=root/frame;folder.mkdir(parents=True,exist_ok=True)
    source=next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id']==frame)
    prior_spec=read(PRIOR/frame/'input.json');prior_result=read(PRIOR/frame/'plane_clipped/result.json')
    if sha(source['mesh'])!=source['mesh_sha256'] or sha(prior_spec['mesh'])!=prior_spec['mesh_sha256'] or sha(PRIOR/frame/'plane_clipped/mesh.ply')!=prior_result['mesh_sha256']:raise ValueError('Changed source mesh')
    if not prior_result['guard_zero_in_last_pass'] or prior_result['protocol_sha256']!=sha(PRIOR/'protocol.json'):raise ValueError('Invalid v3 pilot')
    if sha(source['metadata'])!=sha(prior_spec['metadata']):raise ValueError('Coordinate normalization mismatch')
    request=dict(frame=frame,parent_request_sha256=sha(PARENT/'request.json'),prior_protocol_sha256=sha(PRIOR/'protocol.json'),
        prior_result_sha256=sha(PRIOR/frame/'plane_clipped/result.json'),script_sha256=sha(__file__),
        delta_helper_sha256=sha(Path(__file__).with_name('append_verified_mesh_delta.py')),
        ray_veto_helper_sha256=sha(Path(__file__).with_name('guard_jaw_measured_depth.py')),
        preserved_mesh_sha256=source['mesh_sha256'],original_video_camera=source['camera'],
        observed_guard=dict(camera_count=62,ray_offsets=[0,.5],free_depth_separation=.003,other_observed_support=3,max_pruning_rounds=8),
        production_changed=False,inferred_local_plane_not_measured_anatomy=True)
    if curve_source is not None:
        curve_result=read(curve_source/frame/'result.json')
        if curve_result['anchors_sha256']!=sha(curve_source/frame/'anchors.npz'):raise ValueError('Changed curvature anchors')
        fits={r['model']:r for r in curve_result['fit']}
        if any(r['status']!='diagnostic_fit_only' for r in fits.values()) or fits['quadratic']['camera_partition_test_p90_absolute_depth_error']>=fits['plane']['camera_partition_test_p90_absolute_depth_error']:
            raise ValueError('Quadratic lacks partitioned observed-depth improvement')
        request.update(curvature_result_sha256=sha(curve_source/frame/'result.json'),curvature_source=str(curve_source),
            curvature_helper_sha256=sha(Path(__file__).with_name('curve_forearm_delta.py')),
            source_unclipped_mesh_sha256=sha(PRIOR/frame/'plane/mesh.ply'),
            inferred_local_plane_not_measured_anatomy=False,inferred_quadric_not_measured_anatomy=True,
            curvature_policy=dict(max_depth_displacement=.01,max_triangle_extent=.002,recheck_same_available_skin_masks=True,
                                  minimum_skin_views=2,require_better_partitioned_p90_than_plane=True))
        if boundary_conditioned:
            request['curvature_policy'].update(boundary_ring_exact=True,interior_feather_px=10,extent_metric='axis_extent_strict')
        if matched_plane:
            request.update(inferred_local_plane_not_measured_anatomy=True,inferred_quadric_not_measured_anatomy=False)
            request['curvature_policy'].update(shape='matched_unchanged_plane',max_depth_displacement=0,interior_feather_px=0)
    if photometric_free_space:
        request['observed_guard'].update(kind='depth_and_color_witnesses',min_color_witnesses=3,chroma_mean_abs_limit=.04,patch_size=5)
        request['color_guard_script_hashes']={n:sha(Path(__file__).with_name(n)) for n in ['photometric_forearm_depth_guard.py','diagnose_forearm_color_witnesses.py']}
    if known_annotation_domain:
        request['curvature_policy']['known_annotation_margin']=3
        request['annotation_domain_helper_sha256']=sha(Path(__file__).with_name('annotation_mask_domain.py'))
    if witness_rgb_limit is not None:
        request['observed_guard']['rgb_mean_abs_limit']=witness_rgb_limit
        request['rgb_guard_script_hashes']={n:sha(Path(__file__).with_name(n)) for n in ['rgb_qualified_forearm_depth_guard.py','forearm_rgb_witnesses.py']}
    if witness_comparison_margin is not None:
        request['observed_guard']['comparison_margin']=witness_comparison_margin
        request['observed_guard']['unavailable_comparison_uses_original_rgb_rule']=True
        request['comparison_guard_script_hashes']={n:sha(Path(__file__).with_name(n)) for n in ['contrastive_forearm_depth_guard.py','contrastive_forearm_witnesses.py']}
    if admission_shape is not None:
        request['admission_order']=dict(shape=admission_shape,helper_sha256=sha(Path(__file__).with_name('ordered_forearm_admission.py')),
            grid_helper_sha256=sha(Path(__file__).with_name('confidence_boundary_completion.py')),initial_lookup_and_grid_rules_unchanged=True)
    if observed_ring_feather:
        request['observed_ring_feather']=dict(helper_sha256=sha(Path(__file__).with_name('observed_ring_curvature.py')),
            feather_px=10,original_depth_ring_fixed=True,unknown_boundary_forces_plane=False)
    if (folder/'request.json').exists() and read(folder/'request.json')!=request:raise ValueError('Frozen production transfer mismatch')
    atomic_json(folder/'request.json',request)
    if (folder/'geometry_result.json').exists():
        result=read(folder/'geometry_result.json')
        if result['request_sha256']!=sha(folder/'request.json'):raise ValueError('Changed completed transfer')
        for p,h in result['hashes'].items():
            if sha(folder/p)!=h:raise ValueError('Changed transfer artifact')
        print(frame,'verified transfer',flush=True);return
    target=o3d.io.read_triangle_mesh(source['mesh']);base=o3d.io.read_triangle_mesh(prior_spec['mesh'])
    prior=o3d.io.read_triangle_mesh(str(PRIOR/frame/('plane' if curve_source is not None else 'plane_clipped')/'mesh.ply'))
    curve_record=None;admission_record=None;admission_accepted=None
    if curve_source is not None:
        from curve_forearm_delta import curve_vertices,boundary_curve_vertices,semantic_faces
        if known_annotation_domain:
            from annotation_mask_domain import semantic_faces
        import study_forearm_plane_transfer_v3 as v3
        v3.configure();rows,_,_=evidence.cameras(frame);reference=next(r for r in rows if r['physical_camera']==evidence.NAMES[0])
        if admission_shape is not None:
            from ordered_forearm_admission import rebuild
            prior,admission_accepted,admission_record,arrays=rebuild(frame,admission_shape,fits['quadratic'])
            np.savez_compressed(folder/'admission_evidence.npz',**arrays)
        delta_faces=np.asarray(prior.triangles)[len(base.triangles):]
        selected=np.unique(delta_faces);selected=selected[selected>=len(base.vertices)]
        if matched_plane:
            pv=np.asarray(prior.vertices).copy();curve_record=dict(shape='matched_unchanged_plane',all_input_vertices_exact=True)
        elif boundary_conditioned:
            accepted=np.load(PRIOR/frame/'plane/evidence.npz')['accepted'] if admission_accepted is None else admission_accepted
            if observed_ring_feather:
                from observed_ring_curvature import boundary_curve_vertices as observed_curve
                mesh_depth=np.load(PRIOR/frame/'diagnostic.npz')[evidence.NAMES[0]+'_mesh']
                pv,curve_record=observed_curve(np.asarray(prior.vertices),len(base.vertices),reference,fits['quadratic'],selected,accepted,mesh_depth)
            else:
                pv,curve_record=boundary_curve_vertices(np.asarray(prior.vertices),len(base.vertices),reference,fits['quadratic'],selected,accepted)
        else:
            pv,curve_record=curve_vertices(np.asarray(prior.vertices),len(base.vertices),reference,fits['quadratic'],selected)
        additions,semantic=semantic_faces(pv,delta_faces,rows,evidence.masks(frame),axis_extent=boundary_conditioned);curve_record.update(semantic)
        prior.vertices=o3d.utility.Vector3dVector(pv);prior.triangles=o3d.utility.Vector3iVector(np.concatenate([np.asarray(base.triangles),additions]))
    ov,ot=np.asarray(target.vertices),np.asarray(target.triangles)
    v,t,transfer=append_delta(ov,ot,np.asarray(base.vertices),np.asarray(base.triangles),np.asarray(prior.vertices),np.asarray(prior.triangles))
    if curve_record is not None:transfer['curvature']=curve_record
    if admission_record is not None:transfer['admission_order']=admission_record
    def save(name,triangles):
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(triangles));mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(folder/name),mesh)
    save('transferred.ply',t)
    evidence.OUT=PRIOR;evidence.CONTROLS=PRIOR/'controls';rows,depths,hashes=evidence.load_real(frame)
    if hashes!=read(PRIOR/frame/'analysis.json')['source_depth_sha256']:raise ValueError('Changed native observed depth')
    veto=measured_pixel_veto;color_calls=[];color_provenance=None
    if photometric_free_space:
        from photometric_forearm_depth_guard import make_guard
        if witness_comparison_margin is not None:
            from contrastive_forearm_depth_guard import make_guard as comparison_guard
            veto,color_calls,color_provenance=comparison_guard(frame,rows,depths,witness_comparison_margin)
        elif witness_rgb_limit is None:
            veto,color_calls,color_provenance=make_guard(frame,rows,depths)
        else:
            from rgb_qualified_forearm_depth_guard import make_guard as rgb_guard
            veto,color_calls,color_provenance=rgb_guard(frame,rows,depths,witness_rgb_limit)
    rounds=[]
    for iteration in range(8):
        scene=scene_for(v,t);remove=set();checks=[]
        for ci,(camera,depth) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                ids,count,raw=veto(scene,camera,depth,rows,depths,len(ot),len(t),offset)
                remove.update(ids.tolist());checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw))
            if (ci+1)%10==0:
                atomic_json(folder/'progress.json',dict(stage='observed_depth_guard',iteration=iteration,cameras_done=ci+1,triangles_flagged=len(remove),unix_time=time.time()))
                print(frame,'guard',iteration,ci+1,'flagged',len(remove),flush=True)
        rounds.append(dict(iteration=iteration,removed_triangles=len(remove),checks=checks))
        if not remove:break
        keep=np.ones(len(t),bool);keep[list(remove)]=False
        if not keep[:len(ot)].all():raise ValueError('Attempted removal of production triangles')
        t=t[keep]
    passed=not rounds[-1]['removed_triangles'];save('guarded.ply',t)
    saved=o3d.io.read_triangle_mesh(str(folder/'guarded.ply'))
    if not np.array_equal(np.asarray(saved.vertices)[:len(ov)],ov) or not np.array_equal(np.asarray(saved.triangles)[:len(ot)],ot):raise ValueError('Saved production prefix changed')
    visual=[];panel=Image.new('RGB',(1290,495));draw=ImageDraw.Draw(panel)
    for i,(name,path) in enumerate([('production',Path(source['mesh'])),('transferred',folder/'transferred.ply'),('guarded',folder/'guarded.ply')]):
        mesh=o3d.io.read_triangle_mesh(str(path));mesh.compute_triangle_normals();scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
        d,ids,_=camera_depth(scene,source['camera']);valid=np.isfinite(d)
        light=np.abs(np.asarray(mesh.triangle_normals)@np.array([.3,.4,.866]));rgb=np.zeros((*ids.shape,3),np.uint8)
        rgb[valid]=(60+170*light[ids[valid],None]).astype(np.uint8)
        if name!='production':rgb[valid&(ids>=len(ot))]=[240,60,50]
        image=Image.fromarray(np.rot90(rgb));image.save(folder/f'{name}_clay.png')
        panel.paste(image.crop((0,1450,430,1920)),(430*i,25));draw.text((430*i+4,5),name,fill='white')
        visual.append(dict(variant=name,added_visible_pixels=int((valid&(ids>=len(ot))).sum()) if name!='production' else 0))
    panel.save(folder/'moving_forearm_clay_native.png')
    atomic_json(folder/'geometry_result.json',dict(request_sha256=sha(folder/'request.json'),transfer=transfer,final_added_triangles=len(t)-len(ot),
        observed_guard_passed=passed,rounds=rounds,depth_hashes=hashes,visual=visual,visual_status='pending',production_accepted=False,
        color_guard_calls=color_calls,color_guard_provenance=color_provenance,
        hashes={n:sha(folder/n) for n in ['transferred.ply','guarded.ply','moving_forearm_clay_native.png']+(['admission_evidence.npz'] if admission_shape is not None else [])}))
    print(frame,'finished',transfer['transferred_triangles'],'->',len(t)-len(ot),'pass',passed,visual,flush=True)


def render(root,frame):
    import render_smooth_temporal_mesh_video as renderer
    from temporal_texture_view_prior import install
    from wide_dynamic_camera_flight import install_source_masks
    renderer.torch.set_num_threads(2);install(renderer);install_source_masks(renderer)
    result=read(root/frame/'geometry_result.json')
    if not result['observed_guard_passed']:raise ValueError('Geometry guard failed')
    parent=renderer.verify_request(PARENT)
    evidence.OUT=PRIOR;evidence.CONTROLS=PRIOR/'controls';rows,_,_=evidence.load_real(frame)
    for view in ['moving','H004_A005_1210M6']:
        for variant in ['baseline','guarded']:
            out=root/'rgb'/frame/view/variant;request=deepcopy(parent)
            request['inventory']=[r for r in request['inventory'] if r['frame_id']==frame];row=request['inventory'][0]
            if view!='moving':row['camera']=next(r for r in rows if r['physical_camera']==view)
            if variant=='guarded':row['mesh']=str(root/frame/'guarded.ply');row['mesh_sha256']=sha(row['mesh'])
            request.update(partial_diagnostic_only=True,full_video_candidate=False,forearm_transfer_result_sha256=sha(root/frame/'geometry_result.json'))
            request['script_hashes'][Path(__file__).name]=sha(__file__)
            out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
            if (out/'request.json').exists() and read(out/'request.json')!=request:raise ValueError('Frozen RGB mismatch')
            atomic_json(out/'request.json',request);renderer.render(out,[frame])
        panel=Image.new('RGB',(1080,870));draw=ImageDraw.Draw(panel)
        for i,variant in enumerate(['baseline','guarded']):
            path=root/'rgb'/frame/view/variant/'frames'/frame/'frame.png'
            panel.paste(Image.open(path).crop((0,1080,540,1920)),(540*i,30));draw.text((540*i+4,5),variant,fill='white')
        panel.save(root/frame/f'{view}_rgb_native.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render']);p.add_argument('--frame',required=True,choices=FRAMES)
    p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_forearm_production_delta'))
    p.add_argument('--curved-anchor-root',type=Path);p.add_argument('--boundary-conditioned',action='store_true')
    p.add_argument('--photometric-free-space',action='store_true');p.add_argument('--matched-plane',action='store_true');p.add_argument('--known-annotation-domain',action='store_true')
    p.add_argument('--witness-rgb-limit',type=float);p.add_argument('--witness-comparison-margin',type=float);p.add_argument('--admission-shape',choices=['plane','quadric'])
    p.add_argument('--observed-ring-feather',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.frame,a.curved_anchor_root,a.boundary_conditioned,a.photometric_free_space,a.matched_plane,a.known_annotation_domain,a.witness_rgb_limit,a.witness_comparison_margin,a.admission_shape,a.observed_ring_feather)
    else:render(a.root,a.frame)
