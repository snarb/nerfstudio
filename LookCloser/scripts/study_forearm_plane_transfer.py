"""Two-time transfer of the frozen 001033 boundary-plane/clipping canary.

Read-only original data; unique experiment outputs; no training or defaults.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
from joint_temporal_texture import cameras, read, atomic_json, sha, ROOT, display, exr
from study_confidence_depth_prior import robust_fit, support, unproject, project_integer, raycast_integer

OUT=Path('/mnt/data/dec5_forearm_plane_transfer')
CONTROLS=Path('/mnt/data/dec5_forearm_temporal_transfer/controls')
FRAMES=['001029','001037']
NAMES=['G004_A005_121071','H004_A005_1210M6','H004_C005_1210SZ']
PARENT=Path('/mnt/data/dec5_elevated_camera_dynamic_150')
POLYGONS={
    '001029':{
        NAMES[0]:[(209,1600),(297,1600),(275,1690),(255,1770),(234,1825),(172,1840),(151,1800),(142,1750),(157,1680)],
        NAMES[1]:[(170,1580),(256,1580),(244,1680),(235,1748),(218,1795),(154,1815),(127,1775),(108,1730),(132,1650)],
        NAMES[2]:[(169,1620),(251,1620),(242,1720),(233,1810),(218,1904),(138,1917),(115,1870),(107,1820),(132,1730)]},
    '001037':{
        NAMES[0]:[(91,1760),(177,1760),(163,1825),(145,1865),(124,1888),(83,1900),(63,1869),(47,1820),(65,1780)],
        NAMES[1]:[(71,1745),(153,1745),(153,1813),(133,1850),(118,1870),(83,1878),(58,1848),(42,1805),(57,1763)],
        NAMES[2]:[(65,1760),(150,1760),(146,1830),(144,1916),(26,1916),(42,1835),(59,1790)]}}

def masks(frame):
    result={}
    for name,poly in POLYGONS[frame].items():
        image=np.zeros((1920,1080),np.uint8);cv2.fillPoly(image,[np.array(poly,np.int32)],1)
        result[name]=np.rot90(image,-1).astype(bool)
    return result

def freeze():
    protocol=dict(frames=FRAMES,reference=NAMES[0],train_skin_cameras=NAMES,skin_polygons=POLYGONS,
        polygons_traced_from_train_rgb_before_depth_control_or_candidate_inspection=True,
        parent_protocol_sha256=sha('/mnt/data/dec5_forearm_confidence_prior/experiment_request.json'),
        parent_clip_request_sha256=sha('/mnt/data/dec5_forearm_confidence_prior/clip_request.json'),
        rule='Same robust inverse-depth skin-boundary plane plus append-only four-view clipping',
        minimum_boundary_other_measured_views=3,depth_tolerance=.001,reprojection_pixels=1.5,minimum_parallax_degrees=1,
        maximum_boundary_distance_pixels=100,minimum_reviewed_skin_views=3,trusted_free_space_tolerance=.003,
        allowed_trusted_free_space_contradictions=0,minimum_interior_measured_support=0,maximum_triangle_extent=.002,
        clip_depth_guard=.001,clip_minimum_other_measured_votes=3,clip_maximum_passes=4,
        minimum_plane_anchors_safety_floor=30,variants=['baseline','plane','plane_clipped'],
        heldout_camera='F004_B005_1210O9',heldout_use='evaluation only',learned_model_inference=False,
        thresholds_not_tuned_per_time=True,original_geometry_immutable=True,script_sha256_at_freeze=sha(__file__))
    if (OUT/'protocol.json').exists():raise ValueError('Protocol is already frozen')
    atomic_json(OUT/'protocol.json',protocol)
    for frame in FRAMES:
        out=OUT/frame;panel=Image.new('RGB',(1290,450));draw=ImageDraw.Draw(panel)
        for i,name in enumerate(NAMES):
            rgb=np.rot90(np.array(Image.open(out/'rgb'/(name+'.png')))).copy()
            cv2.polylines(rgb,[np.array(POLYGONS[frame][name],np.int32)],True,(255,255,0),2)
            Image.fromarray(rgb).crop((0,1500,430,1920)).save(out/(name+'_mask_native.png'))
            panel.paste(Image.fromarray(rgb).crop((0,1500,430,1920)),(430*i,30));draw.text((430*i+4,8),name,fill='white')
        panel.save(out/'skin_masks_native.png')
    print('Frozen transfer thresholds and train-only polygons',flush=True)

def load_real(frame):
    from import_colmap_mvs_depth_dataset import read_colmap_dense_array
    from render_patchmatch_camera_path import normalize_frame
    control=CONTROLS/frame
    if not (control/'complete.json').exists():raise ValueError('Parent depth control is not complete; do not consume partial maps')
    spec=read(OUT/frame/'input.json');metadata=read(spec['metadata']);rows,_,_=cameras(frame)
    raw=read(control/'staged63/transforms.json');lookup={r['physical_camera']:r for r in raw['frames']};depths=[];hashes={}
    for row in rows:
        f=lookup[row['physical_camera']];normalized=normalize_frame(f,raw,metadata)
        for key in ['transform_matrix','fl_x','fl_y','cx','cy']:
            assert np.allclose(normalized[key],row[key],rtol=0,atol=1e-6)
        p=control/'pipeline/dense/stereo/depth_maps'/(f['file_path']+'.geometric.bin');d=read_colmap_dense_array(p)
        assert d.shape==(1080,1920,1) and np.isfinite(d).all();depths.append(d[...,0]*metadata['dataparser_scale']);hashes[str(p)]=sha(p)
    assert len(depths)==62
    return rows,depths,hashes

def analyze(frame):
    import time
    import open3d as o3d
    from scipy import ndimage
    from diffusion_mesh_repair import scene_for
    started=time.monotonic();out=OUT/frame;spec=read(out/'input.json');protocol=read(OUT/'protocol.json')
    assert protocol['skin_polygons']==json.loads(json.dumps(POLYGONS))
    if (out/'analysis.json').exists():raise ValueError('Analysis exists; retain it rather than retune')
    rows,depths,hashes=load_real(frame);byname={r['physical_camera']:i for i,r in enumerate(rows)};skin=masks(frame)
    old=o3d.io.read_triangle_mesh(spec['mesh']);v=np.asarray(old.vertices);t=np.asarray(old.triangles);scene=scene_for(v,t)
    maps={};diagnostics=[]
    for name in NAMES:
        i=byname[name];row=rows[i];pm=depths[i];md=raycast_integer(scene,row);mask=skin[name]
        y,x=np.nonzero(mask&(pm>0));count,_=support(unproject(row,x,y,pm[y,x]),row,rows,depths)
        votes=np.zeros(pm.shape,np.uint8);votes[y,x]=count;trusted=mask&(votes>=3)&(md>0)&(np.abs(pm-md)<.001)
        maps[name+'_mesh']=md;maps[name+'_trusted']=trusted;maps[name+'_pm']=pm;maps[name+'_votes']=votes
        rgb=np.array(Image.open(out/'rgb'/(name+'.png')));rgb[mask&(md==0)]=[255,0,0];rgb[trusted]=[0,255,0]
        Image.fromarray(np.rot90(rgb)).crop((0,1500,430,1920)).save(out/(name+'_support_native.png'))
        diagnostics.append(dict(camera=name,skin_pixels=int(mask.sum()),original_missing=int((mask&(md==0)).sum()),trusted_pixels=int(trusted.sum())))
    ref=rows[byname[NAMES[0]]];pm=depths[byname[NAMES[0]]];md=maps[NAMES[0]+'_mesh'];trusted=maps[NAMES[0]+'_trusted']
    hole=skin[NAMES[0]]&(md==0);hy,hx=np.nonzero(hole);ty,tx=np.nonzero(trusted)
    if len(tx)<30:raise ValueError(f'Only {len(tx)} trusted anchors: do not invent a plane')
    coef,rmse=robust_fit(np.column_stack((tx/100,ty/100,np.ones(len(tx)))),1/pm[ty,tx])
    z=1/(np.column_stack((hx/100,hy/100,np.ones(len(hx))))@coef);points=unproject(ref,hx,hy,z)
    observed,free=support(points,ref,rows,depths);skinvotes=np.zeros(len(z),np.uint8);trusted_free=np.zeros(len(z),np.uint8)
    for name in NAMES:
        row=rows[byname[name]];uv,cz=project_integer(row,points);xy=np.rint(uv).astype(int)
        inside=(cz>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080);ids=np.flatnonzero(inside);qx,qy=xy[ids].T
        skinvotes[ids]+=skin[name][qy,qx]
        trusted_free[ids]+=maps[name+'_trusted'][qy,qx]&(depths[byname[name]][qy,qx]>cz[ids]+.003)
    distance=ndimage.distance_transform_edt(~trusted)[hy,hx]
    eligible=np.isfinite(z)&(z>0)&(distance<=100)&(skinvotes==3)&(trusted_free==0)
    added=np.zeros(md.shape,np.float32);added[hy[eligible],hx[eligible]]=z[eligible];accepted=added>0
    domain=ndimage.binary_dilation(accepted)&((md>0)|accepted);y,x=np.nonzero(domain)
    newv=unproject(ref,x,y,np.where(accepted,added,md)[y,x]);index=np.full(md.shape,-1,np.int32);index[y,x]=np.arange(len(x))+len(v)
    a=index[:-1,:-1];b=index[:-1,1:];c=index[1:,:-1];d=index[1:,1:];tri=[]
    for aa,bb,cc,h in [(a,b,c,accepted[:-1,:-1]|accepted[:-1,1:]|accepted[1:,:-1]),(b,d,c,accepted[:-1,1:]|accepted[1:,1:]|accepted[1:,:-1])]:
        ok=(aa>=0)&(bb>=0)&(cc>=0)&h;tri.append(np.column_stack((aa[ok],bb[ok],cc[ok])))
    triangles=np.concatenate(tri);vv=np.concatenate((v,newv));triangles=triangles[np.ptp(vv[triangles],axis=1).max(1)<.002]
    tt=np.concatenate((t,triangles));dest=out/'plane';dest.mkdir(exist_ok=True)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt));mesh.compute_vertex_normals();o3d.io.write_triangle_mesh(str(dest/'mesh.ply'),mesh)
    assert np.array_equal(vv[:len(v)],v) and np.array_equal(tt[:len(t)],t)
    np.savez_compressed(out/'diagnostic.npz',**maps);np.savez_compressed(dest/'evidence.npz',depth=added,accepted=accepted,all_candidate_xy=np.column_stack((hx,hy)),
        observed=observed,raw_free_space=free,trusted_free_space=trusted_free,skin_views=skinvotes)
    record=dict(frame=frame,reference=NAMES[0],diagnostics=diagnostics,source_depth_sha256=hashes,protocol_sha256=sha(OUT/'protocol.json'),
        control_complete_sha256=sha(CONTROLS/frame/'complete.json'),plane_inverse_coefficients=coef.tolist(),plane_inverse_rmse=rmse,
        trusted_reference_anchors=len(tx),original_missing_pixels=len(hx),accepted_pixels=int(eligible.sum()),added_triangles=len(triangles),
        accepted_zero_measured_votes=int((eligible&(observed==0)).sum()),accepted_ge2_measured_votes=int((eligible&(observed>=2)).sum()),
        rejected_skin_limit=int((skinvotes<3).sum()),rejected_trusted_free_space=int((trusted_free>0).sum()),rejected_distance=int((distance>100).sum()),
        mesh_sha256=sha(dest/'mesh.ply'),original_geometry_preserved=True,heldout_used=False,inferred_not_measured=True,elapsed_seconds=time.monotonic()-started)
    atomic_json(out/'analysis.json',record);print({k:v for k,v in record.items() if k!='source_depth_sha256'},flush=True)

def clip_plane(frame):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    out=OUT/frame;spec=read(out/'input.json');rows,depths,_=load_real(frame);old=o3d.io.read_triangle_mesh(spec['mesh'])
    ov=np.asarray(old.vertices);ot=np.asarray(old.triangles);sceneold=scene_for(ov,ot)
    mesh=o3d.io.read_triangle_mesh(str(out/'plane/mesh.ply'));v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles).copy()
    selected={'moving':spec['moving_camera']};selected.update({r['physical_camera']:r for r in rows if r['physical_camera'] in NAMES})
    olddepth={name:camera_depth(sceneold,camera)[0] for name,camera in selected.items()};rounds=[]
    if (out/'plane_clipped/mesh.ply').exists():raise ValueError('Clipped candidate exists')
    for iteration in range(4):
        scene=scene_for(v,t);remove=set();checks=[]
        for name,camera in selected.items():
            d,ids,_=camera_depth(scene,camera);od=olddepth[name];front=np.isfinite(od)&np.isfinite(d)&(d<od-.001)
            y,x=np.nonzero(front);votes,_=support(unproject(camera,x,y,od[y,x],offset=.5),camera,rows,depths)
            implicated=ids[y[votes>=3],x[votes>=3]];assert (implicated>=len(ot)).all();remove.update(implicated.tolist())
            checks.append(dict(camera=name,trusted_old_occluded=int((votes>=3).sum())))
        rounds.append(dict(iteration=iteration,removed_triangles=len(remove),checks=checks))
        if not remove:break
        keep=np.ones(len(t),bool);keep[list(remove)]=False;t=t[keep]
    assert np.array_equal(t[:len(ot)],ot)
    dest=out/'plane_clipped';dest.mkdir(exist_ok=True);result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t))
    result.compute_vertex_normals();o3d.io.write_triangle_mesh(str(dest/'mesh.ply'),result)
    atomic_json(dest/'result.json',dict(rounds=rounds,parent_plane_sha256=sha(out/'plane/mesh.ply'),protocol_sha256=sha(OUT/'protocol.json'),
        retained_added_triangles=len(t)-len(ot),removed_total=len(np.asarray(mesh.triangles))-len(t),mesh_sha256=sha(dest/'mesh.ply'),
        original_geometry_preserved=True,heldout_used=False,guard_zero_in_last_pass=all(r['trusted_old_occluded']==0 for r in rounds[-1]['checks'])))
    print(rounds,flush=True)

def replay_001033():
    """CPU-only regression: transfer helper must reproduce the original canary."""
    import open3d as o3d
    import study_forearm_confidence_prior as prior
    global OUT, CONTROLS, FRAMES, POLYGONS
    root=OUT;OUT=root/'algorithm_replay_001033';CONTROLS=OUT/'controls';FRAMES=['001033'];POLYGONS={'001033':prior.POLYGONS}
    CONTROLS.mkdir(parents=True,exist_ok=True)
    (CONTROLS/'001033').symlink_to('/mnt/data/dec5_forearm_depth_control_001033',target_is_directory=True)
    stage();freeze();analyze('001033');clip_plane('001033');records=[]
    for variant in ['plane','plane_clipped']:
        expected=prior.OUT/'001033'/variant/'mesh.ply';actual=OUT/'001033'/variant/'mesh.ply'
        a=o3d.io.read_triangle_mesh(str(expected));b=o3d.io.read_triangle_mesh(str(actual))
        assert np.array_equal(np.asarray(a.vertices),np.asarray(b.vertices))
        assert np.array_equal(np.asarray(a.triangles),np.asarray(b.triangles))
        records.append(dict(variant=variant,expected_sha256=sha(expected),actual_sha256=sha(actual),arrays_exact=True,bytes_exact=sha(expected)==sha(actual)))
    atomic_json(root/'algorithm_replay.json',dict(rows=records,passed=True,gpu_used=False,original_artifacts_untouched=True))
    print('001033 frozen plane/clipping replay passed',flush=True)

def views(frame):
    from render_patchmatch_camera_path import normalize_frame
    from joint_temporal_texture import CALIBRATION
    spec=read(OUT/frame/'input.json');rows,_,_=cameras(frame);train=next(r for r in rows if r['physical_camera']==NAMES[1])
    cal=read(CALIBRATION);held=next(r for r in cal['frames'] if r['physical_camera']=='F004_B005_1210O9')
    return dict(moving=spec['moving_camera'],train_H_A=train,heldout_F_B=normalize_frame(held,cal,read(spec['metadata'])))

def geometry_review(frame):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    out=OUT/frame;spec=read(out/'input.json');rows,depths,_=load_real(frame);records=[]
    original=o3d.io.read_triangle_mesh(spec['mesh']);original_v=np.asarray(original.vertices);original_t=np.asarray(original.triangles)
    for view,camera in views(frame).items():
        dest=out/'review'/view;dest.mkdir(parents=True,exist_ok=True);images=[]
        for name in ['baseline','plane','plane_clipped']:
            path=Path(spec['mesh']) if name=='baseline' else out/name/'mesh.ply';mesh=o3d.io.read_triangle_mesh(str(path))
            v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles);mesh.compute_triangle_normals();normal=np.asarray(mesh.triangle_normals)
            d,ids,_=camera_depth(scene_for(v,t),camera);hit=np.isfinite(d);depth=np.where(hit,d,0)
            rgb=np.zeros((1080,1920,3),np.uint8);light=np.array(camera['transform_matrix'])[:3,2]
            shade=.2+.8*np.abs(normal[ids[hit]]@light);rgb[hit]=(shade[:,None]*255).clip(0,255).astype(np.uint8)
            image=Image.fromarray(np.rot90(rgb));image.save(dest/(name+'_clay_native.png'));images.append(image)
            if name=='baseline':first=depth
            new=(first==0)&hit;front=(first>0)&hit&(depth<first-.001);y,x=np.nonzero(front)
            votes,_=support(unproject(camera,x,y,first[y,x],offset=.5),camera,rows,depths)
            np.savez_compressed(dest/(name+'_depth.npz'),depth=depth,newly_visible=new,in_front=front)
            records.append(dict(frame=frame,view=view,variant=name,newly_visible=int(new.sum()),
                old_pixels_occluded_by_001=int(front.sum()),occluded_old_pixels_with_ge3_measured_votes=int((votes>=3).sum()),
                old_geometry_arrays_preserved=np.array_equal(t[:len(original_t)],original_t) and np.array_equal(v[:len(original_v)],original_v)))
        box=(150,1450,560,1920) if view=='moving' else (0,1500,430,1920);w,h=box[2]-box[0],box[3]-box[1]
        panel=Image.new('RGB',(w*3,h+28));draw=ImageDraw.Draw(panel)
        for i,(im,name) in enumerate(zip(images,['baseline','plane','plane_clipped'])):
            panel.paste(im.crop(box),(w*i,28));draw.text((w*i+4,7),name,fill='white')
        panel.save(dest/'clay_comparison_native.png')
    atomic_json(out/'geometry_review.json',dict(rows=records,heldout_geometry_used_for_clipping=False));print(records,flush=True)

def render(frame):
    from copy import deepcopy
    import torch
    from render_smooth_temporal_mesh_video import render_one
    torch.set_num_threads(2);out=OUT/frame;spec=read(out/'input.json');base=read(PARENT/'request.json')
    record=next(r for r in base['inventory'] if r['frame_id']==frame);source=next(r for r in base['source_rows'] if Path(r['source_dataset']).name==frame)
    # Call only after parent GPU handoff; no local depth job is launched here.
    for view,camera in views(frame).items():
        for variant in ['baseline','plane_clipped']:
            mesh=Path(spec['mesh']) if variant=='baseline' else out/variant/'mesh.ply'
            dest=out/variant/('render_'+view);dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
            row=deepcopy(record);row['mesh']=str(mesh);row['mesh_sha256']=sha(mesh);row['camera']=camera
            request=dict(comparison='fixed-rule forearm plane transfer',variant=variant,view=view,inventory=[row],source_rows=[source],
                profiles_sha256=spec['profiles_sha256'],exposure_sha256=spec['exposure_sha256'],uses_heldout_rgb=False,
                renderer_sha256=sha(Path(__file__).with_name('render_smooth_temporal_mesh_video.py')),protocol_sha256=sha(OUT/'protocol.json'))
            if (dest/'request.json').exists():assert read(dest/'request.json')==request
            atomic_json(dest/'request.json',request);render_one(dest,row,source)
            print(f'render complete {frame} {view} {variant}',flush=True)

def evaluation_inputs(frame):
    from joint_temporal_texture import SOURCE
    out=OUT/frame;assert (out/'plane_clipped/result.json').exists()
    raw=read(SOURCE/frame/'transforms.json');f=next(r for r in raw['frames'] if r['physical_camera']=='F004_B005_1210O9')
    path=SOURCE/frame/f['file_path'];exposure=read(ROOT/'exposure.json')['fixed_exposure_gain']
    rgb=np.rint(255*display(exr(path),exposure)).clip(0,255).astype(np.uint8);dest=out/'evaluation';dest.mkdir(exist_ok=True)
    Image.fromarray(np.rot90(rgb)).save(dest/'heldout_gt_native.png');Image.fromarray(np.rot90(rgb)).crop((0,1450,550,1920)).save(dest/'heldout_forearm_gt_crop.png')
    atomic_json(dest/'input.json',dict(heldout_source=str(path),heldout_sha256=sha(path),evaluation_only=True,
        candidate_mesh_sha256=sha(out/'plane_clipped/mesh.ply'),protocol_sha256=sha(OUT/'protocol.json')))

def evaluation_mask(frame,polygon,not_visible=False):
    dest=OUT/frame/'evaluation'
    if not not_visible and polygon is None:raise ValueError('Supply a polygon or explicitly mark --not-visible')
    points=np.empty((0,2),np.int32) if not_visible else np.array(polygon,np.int32).reshape(-1,2)
    if (dest/'regions.json').exists():raise ValueError('Evaluation region is already frozen')
    mask=np.zeros((1920,1080),np.uint8)
    if not not_visible:cv2.fillPoly(mask,[points],1);assert mask.sum()>100
    np.savez_compressed(dest/'regions.npz',forearm_skin=mask.astype(bool));image=np.array(Image.open(dest/'heldout_gt_native.png'))
    if not not_visible:cv2.polylines(image,[points],True,(255,255,0),2)
    preview=Image.fromarray(image).crop((0,1450,550,1920))
    if not_visible:ImageDraw.Draw(preview).text((8,8),'Forearm below frame: no substitute ROI',fill='yellow')
    preview.save(dest/'heldout_mask_native.png')
    atomic_json(dest/'regions.json',dict(points=points.tolist(),manual_gt_only=True,region_used_for_geometry=False,
        not_visible=not_visible,no_hand_roi_substituted=not_visible,masks_sha256=sha(dest/'regions.npz'),gt_sha256=sha(dest/'heldout_gt_native.png')))

def semantic_review(frame):
    """Validate visible new-surface footprints against the frozen skin bounds."""
    import open3d as o3d
    from scipy import ndimage
    from diffusion_mesh_repair import scene_for
    out=OUT/frame;spec=read(out/'input.json');maps=np.load(out/'diagnostic.npz');skin=masks(frame)
    mesh=o3d.io.read_triangle_mesh(str(out/'plane_clipped/mesh.ply'));scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    rows={r['physical_camera']:r for r in spec['frames']};records=[];panel=Image.new('RGB',(1290,450));draw=ImageDraw.Draw(panel)
    realrows,realdepths,_=load_real(frame)
    for i,name in enumerate(NAMES):
        old=maps[name+'_mesh'];new=raycast_integer(scene,rows[name]);added=(old==0)&(new>0)
        front=(old>0)&(new>0)&(new<old-.001);changed=added|front;outside=changed&~skin[name]
        y,x=np.nonzero(front);votes,_=support(unproject(rows[name],x,y,old[y,x]),rows[name],realrows,realdepths)
        distances=ndimage.distance_transform_edt(~skin[name]);rgb=np.array(Image.open(rows[name]['file_path']))
        rgb[changed]=[0,255,255];rgb[outside]=[255,0,0]
        image=Image.fromarray(np.rot90(rgb)).crop((0,1500,430,1920));image.save(out/(name+'_semantic_footprint_native.png'))
        panel.paste(image,(430*i,30));draw.text((430*i+4,8),name,fill='white')
        records.append(dict(camera=name,newly_visible_integer_pixels=int(added.sum()),newly_front_integer_pixels=int(front.sum()),
            integer_front_old_pixels_with_ge3_measured_votes=int((votes>=3).sum()),
            changed_outside_inset_skin_polygon=int(outside.sum()),maximum_outside_distance_pixels=float(distances[outside].max()) if outside.any() else 0.0,
            changed_more_than_two_pixels_outside=int((outside&(distances>2)).sum())))
    panel.save(out/'semantic_footprints_native.png')
    atomic_json(out/'semantic_review.json',dict(rows=records,cyan='new visible/front surface',red='outside inset train-skin polygon',
        masks_are_manual_inset_regions_not_ground_truth_silhouettes=True,integer_rays_match_mask_and_depth_convention=True,
        new_geometry_not_modified_by_this_audit=True));print(records,flush=True)

def boundary_limit_audit(frame):
    out=OUT/frame;spec=read(out/'input.json');analysis=read(out/'analysis.json');skin=masks(frame)
    xy=np.load(out/'plane/evidence.npz')['all_candidate_xy'];x,y=xy.T
    z=1/(np.column_stack((x/100,y/100,np.ones(len(x))))@np.array(analysis['plane_inverse_coefficients']))
    reference=next(r for r in spec['frames'] if r['physical_camera']==NAMES[0]);points=unproject(reference,x,y,z);records=[]
    for row in spec['frames']:
        name=row['physical_camera'];uv,cz=project_integer(row,points);q=np.rint(uv).astype(int)
        inside=(cz>0)&(q[:,0]>=0)&(q[:,0]<1920)&(q[:,1]>=0)&(q[:,1]<1080);ids=np.flatnonzero(inside)
        good=np.zeros(len(x),bool);good[ids]=skin[name][q[ids,1],q[ids,0]]
        records.append(dict(camera=name,outside_image_under_plane=int((~inside).sum()),inside_image_outside_skin_polygon=int((inside&~good).sum()),inside_skin_polygon=int(good.sum())))
    atomic_json(out/'boundary_limits.json',dict(rows=records,points_are_inferred_plane_not_ground_truth=True,
        no_rule_or_geometry_changed=True,source_candidate_pixels=len(x)));print(records,flush=True)

def semantic_replay_001033():
    """Check the original canary's other pixel lattice without changing it."""
    import study_forearm_confidence_prior as prior
    global OUT, CONTROLS, POLYGONS
    root=OUT;OUT=root/'algorithm_replay_001033';CONTROLS=OUT/'controls';POLYGONS={'001033':prior.POLYGONS}
    semantic_review('001033')
    atomic_json(root/'algorithm_replay_semantics.json',read(OUT/'001033/semantic_review.json'))

def score(frame):
    import torch
    from score_colmap_patchmatch_tsdf_face import masked_display_metrics
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    out=OUT/frame;dest=out/'evaluation';torch.set_num_threads(2)
    model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval();records=[]
    for view in views(frame):
        if view=='moving':gt=None;regions={}
        elif view=='train_H_A':
            gt=np.rot90(np.array(Image.open(out/'rgb'/(NAMES[1]+'.png')))).copy();mask=np.rot90(masks(frame)[NAMES[1]]).copy()
            d=np.rot90(np.load(out/'review'/view/'baseline_depth.npz')['depth']);regions=dict(forearm_skin=mask,original_missing_skin=mask&(d==0))
        else:gt=np.array(Image.open(dest/'heldout_gt_native.png'));regions=dict(forearm_skin=np.load(dest/'regions.npz')['forearm_skin'])
        predictions=[];labels=[]
        if gt is not None:predictions.append(gt);labels.append('Held-out GT' if view=='heldout_F_B' else 'Train reference')
        base=np.array(Image.open(out/'baseline'/('render_'+view)/'frames'/frame/'frame.png'))
        for variant in ['baseline','plane_clipped']:
            path=out/variant/('render_'+view)/'frames'/frame/'frame.png';pred=np.array(Image.open(path));predictions.append(pred);labels.append(variant)
            changed=(pred!=base).any(-1)
            for name,mask in regions.items():
                if not mask.any():
                    records.append(dict(frame=frame,view=view,variant=variant,region=name,pixels=0,metrics_available=False,
                        reason='Requested anatomy below frame; no surrogate hand/wrist ROI',full_image_changed_pixels=int(changed.sum()),prediction_sha256=sha(path)));continue
                with torch.inference_mode():
                    metric=masked_display_metrics(torch.tensor(pred.transpose(2,0,1)/255,dtype=torch.float32,device='cuda'),
                        torch.tensor(gt.transpose(2,0,1)/255,dtype=torch.float32,device='cuda'),torch.tensor(mask,device='cuda'),model)
                metric={k.removeprefix('face_'):v for k,v in metric.items()}
                records.append(dict(frame=frame,view=view,variant=variant,region=name,**metric,pixels=int(mask.sum()),
                    changed_pixels=int((changed&mask).sum()),full_image_changed_pixels=int(changed.sum()),prediction_sha256=sha(path),
                    train_reference_not_heldout=view=='train_H_A',metrics_available=True))
        box=(150,1450,560,1920) if view=='moving' else (0,1450,550,1920);w,h=box[2]-box[0],box[3]-box[1]
        panel=Image.new('RGB',(w*len(predictions),h+28));draw=ImageDraw.Draw(panel)
        for i,(im,label) in enumerate(zip(predictions,labels)):
            panel.paste(Image.fromarray(im).crop(box),(w*i,28));draw.text((w*i+4,7),label,fill='white')
        panel.save(dest/(view+'_rgb_comparison_native.png'))
    atomic_json(out/'metrics.json',dict(rows=records,protocol='Masked RGB PSNR; tight zero-outside bbox SSIM/AlexNet LPIPS',
        train_metrics_not_heldout_generalization=True,heldout_not_geometry_input=True));print(records,flush=True)

def audit(frame):
    import open3d as o3d
    from joint_temporal_texture import HELD_CAMERAS
    out=OUT/frame;spec=read(out/'input.json');analysis=read(out/'analysis.json');rows,depths,hashes=load_real(frame)
    assert hashes==analysis['source_depth_sha256'];assert sha(spec['mesh'])==spec['mesh_sha256'];assert sha(spec['metadata'])==spec['metadata_sha256']
    assert sha(CONTROLS/frame/'complete.json')==analysis['control_complete_sha256']
    assert sha(OUT/'protocol.json')==analysis['protocol_sha256']
    assert read(OUT/'protocol.json')['skin_polygons']==json.loads(json.dumps(POLYGONS))
    assert sha(ROOT/'camera_profiles.json')==spec['profiles_sha256'];assert sha(ROOT/'exposure.json')==spec['exposure_sha256']
    assert not ({r['physical_camera'] for r in rows}&HELD_CAMERAS)
    for row in spec['frames']:assert sha(row['source_file_path'])==row['source_sha256']
    old=o3d.io.read_triangle_mesh(spec['mesh']);v=np.asarray(old.vertices);t=np.asarray(old.triangles);records=[]
    for variant in ['plane','plane_clipped']:
        path=out/variant/'mesh.ply';mesh=o3d.io.read_triangle_mesh(str(path));vv=np.asarray(mesh.vertices);tt=np.asarray(mesh.triangles)
        assert np.array_equal(vv[:len(v)],v) and np.array_equal(tt[:len(t)],t)
        expected=analysis['mesh_sha256'] if variant=='plane' else read(out/variant/'result.json')['mesh_sha256'];assert sha(path)==expected
        records.append(dict(variant=variant,original_arrays_preserved=True,added_triangles=len(tt)-len(t),mesh_sha256=sha(path)))
    render_records=[]
    for view in views(frame):
        for variant in ['baseline','plane_clipped']:
            path=out/variant/('render_'+view)/'frames'/frame;receipt=read(path/'complete.json')
            for filename,digest in receipt['hashes'].items():assert sha(path/filename)==digest
            render_records.append(read(path/'result.json'))
    metrics=read(out/'metrics.json')['rows'];assert len(metrics)==6
    for row in metrics:
        if row['metrics_available']:
            path=out/row['variant']/('render_'+row['view'])/'frames'/frame/'frame.png';assert sha(path)==row['prediction_sha256']
            assert all(np.isfinite(row[k]) for k in ['psnr','ssim','lpips'])
    geometry=read(out/'geometry_review.json')['rows'];clipped=read(out/'plane_clipped/result.json')
    semantic=read(out/'semantic_review.json')
    atomic_json(out/'audit.json',dict(geometry=records,source_depth_hashes_passed=True,source_rgb_hashes_passed=True,
        original_metadata_profiles_exposure_unchanged=True,heldout_excluded_from_geometry=True,
        all_render_hashes_passed=True,render_count=len(render_records),metric_rows=len(metrics),
        clipping_guard_passed=clipped['guard_zero_in_last_pass'],
        review_guard_passed=all(r['occluded_old_pixels_with_ge3_measured_votes']==0 for r in geometry if r['variant']=='plane_clipped'),
        integer_lattice_guard_passed=all(r['integer_front_old_pixels_with_ge3_measured_votes']==0 for r in semantic['rows']),
        render_seconds=sum(r['elapsed_seconds'] for r in render_records),protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(__file__),
        semantic_footprint_audit=semantic['rows'],visual_acceptance_is_separate=True))
    print(f'audited {frame}: sources, arrays, six renders, six metrics',flush=True)

def summarize():
    results=[];metrics=[];audits=[];images={}
    for frame in FRAMES:
        out=OUT/frame;results.append(read(out/'analysis.json'));metrics.extend(read(out/'metrics.json')['rows']);audits.append(read(out/'audit.json'))
        for view in views(frame):
            path=out/'evaluation'/(view+'_rgb_comparison_native.png');images[str(path)]=sha(path)
    # Constant physical H_A, no alignment/stabilization: compare actual motion.
    order=['001029','001033','001037'];panel=Image.new('RGB',(1290,1455));draw=ImageDraw.Draw(panel)
    for j,frame in enumerate(order):
        root=OUT if frame!='001033' else Path('/mnt/data/dec5_forearm_confidence_prior')
        out=root/frame;gt=np.rot90(np.array(Image.open(out/'rgb'/(NAMES[1]+'.png')))).copy()
        parts=[Image.fromarray(gt)]+[Image.open(out/v/'render_train_H_A/frames'/frame/'frame.png') for v in ['baseline','plane_clipped']]
        for i,(im,label) in enumerate(zip(parts,['Train reference','Original TSDF','Clipped plane'])):
            panel.paste(im.crop((0,1500,430,1920)),(i*430,j*485+30));draw.text((i*430+4,j*485+8),frame+' '+label,fill='white')
    panel.save(OUT/'three_time_H_A_native.png');images[str(OUT/'three_time_H_A_native.png')]=sha(OUT/'three_time_H_A_native.png')
    reference={r['physical_camera']:r for r in cameras('001033')[0]};pose_deltas={}
    for frame in FRAMES:
        pose_deltas[frame]=max(float(np.abs(np.asarray(r['transform_matrix'])-np.asarray(reference[r['physical_camera']]['transform_matrix'])).max()) for r in cameras(frame)[0])
        assert pose_deltas[frame]<1e-6
    assert sum(a['render_count'] for a in audits)==12 and len(metrics)==12
    atomic_json(OUT/'summary.json',dict(results=results,metrics=metrics,audits=audits,native_review_images=images,
        pose_maximum_absolute_difference_from_001033=pose_deltas,calibration_and_profiles_frozen=True,
        algorithm_replay=read(OUT/'algorithm_replay.json'),protocol_sha256=sha(OUT/'protocol.json'),
        algorithm_replay_semantics=read(OUT/'algorithm_replay_semantics.json'),
        actual_temporal_stability_not_established_by_three_sparse_times=True,learned_inference_runs=0,
        source_dataset_unchanged=True,production_defaults_changed=False,full_video_rerun=False,
        visual_acceptance_requires_manual_review=True,report='experiments/dec5_forearm_plane_transfer.md'))
    print(f'Two transfer times: 12 renders, {sum(r["metrics_available"] for r in metrics)} metric triplets / 12 region records; manual visual decision remains explicit',flush=True)

def stage():
    OUT.mkdir(parents=True,exist_ok=True)
    profiles=read(ROOT/'camera_profiles.json');gains=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']))
    exposure=read(ROOT/'exposure.json')['fixed_exposure_gain'];parent=read(PARENT/'request.json')
    for frame in FRAMES:
        out=OUT/frame;out.mkdir(exist_ok=True)
        if (out/'input.json').exists():raise ValueError('Already staged; preserve the existing transfer workspace')
        rows,mesh,meta=cameras(frame);byname={r['physical_camera']:r for r in rows};staged=[];panel=Image.new('RGB',(1290,450));draw=ImageDraw.Draw(panel)
        for i,name in enumerate(NAMES):
            r=byname[name];path=out/'rgb'/(name+'.png');path.parent.mkdir(exist_ok=True)
            rgb=np.rint(255*display(exr(r['file_path'])*np.array(gains[name]),exposure)).clip(0,255).astype(np.uint8)
            Image.fromarray(rgb).save(path);image=Image.fromarray(np.rot90(rgb));image.crop((0,1500,430,1920)).save(out/(name+'_forearm_native.png'))
            panel.paste(image.crop((0,1500,430,1920)),(430*i,30));draw.text((430*i+4,8),name,fill='white')
            staged.append(dict(r,source_file_path=r['file_path'],source_sha256=sha(r['file_path']),file_path=str(path)))
        panel.save(out/'train_previews_native.png')
        record=next(r for r in parent['inventory'] if r['frame_id']==frame)
        Image.open(PARENT/'frames'/frame/'frame.png').crop((150,1450,560,1920)).save(out/'moving_parent_crop.png')
        atomic_json(out/'input.json',dict(frame=frame,frames=staged,mesh=str(mesh),mesh_sha256=sha(mesh),metadata=str(meta),
            metadata_sha256=sha(meta),profiles_sha256=sha(ROOT/'camera_profiles.json'),exposure_sha256=sha(ROOT/'exposure.json'),
            moving_camera=record['camera'],parent_request_sha256=sha(PARENT/'request.json'),heldout_used=False,
            control_root=str(CONTROLS/frame),script_sha256=sha(__file__)))
    print('Staged two times, three train views each; no GPU or held-out input',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['stage','freeze','replay_001033','analyze','clip_plane','geometry_review','semantic_review','boundary_limit_audit','semantic_replay_001033','evaluation_inputs','evaluation_mask','render','score','audit','summarize'])
    parser.add_argument('--frame',choices=FRAMES);parser.add_argument('--polygon',type=int,nargs='+');parser.add_argument('--not-visible',action='store_true')
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--controls',type=Path,default=CONTROLS);args=parser.parse_args();OUT=args.output;CONTROLS=args.controls
    cv2.setNumThreads(2)
    if args.command=='evaluation_mask':evaluation_mask(args.frame,args.polygon,args.not_visible)
    elif args.frame:globals()[args.command](args.frame)
    else:globals()[args.command]()
