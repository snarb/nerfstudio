"""One-frame, append-only forearm prior pilot; not a production completion rule."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw
from joint_temporal_texture import cameras, read, atomic_json, sha, ROOT, display, exr
from study_confidence_depth_prior import infer, robust_fit, unproject, project_integer, support, raycast_integer

OUT=Path('/mnt/data/dec5_forearm_confidence_prior')
FRAME='001033'
CONTROL=Path('/mnt/data/dec5_forearm_depth_control_001033')
PARENT=Path('/mnt/data/dec5_elevated_camera_dynamic_150')
REFERENCE='G004_A005_121071'
POLYGONS={
    REFERENCE:[(160,1640),(243,1640),(224,1730),(209,1790),(183,1849),(124,1862),(101,1817),(96,1770),(115,1705)],
    'H004_A005_1210M6':[(123,1640),(205,1640),(193,1710),(188,1778),(165,1818),(113,1836),(95,1808),(82,1760),(90,1700)],
    'H004_C005_1210SZ':[(115,1670),(206,1670),(197,1760),(187,1850),(183,1919),(71,1919),(65,1890),(84,1810),(107,1710)],
}

def masks():
    result={}
    for name,poly in POLYGONS.items():
        mask=np.zeros((1920,1080),np.uint8);cv2.fillPoly(mask,[np.array(poly,np.int32)],1)
        result[name]=np.rot90(mask,-1).astype(bool)
    return result

def load_real():
    from import_colmap_mvs_depth_dataset import read_colmap_dense_array
    from render_patchmatch_camera_path import normalize_frame
    rows,_,meta=cameras(FRAME);metadata=read(meta);raw=read(CONTROL/'staged63/transforms.json')
    lookup={r['physical_camera']:r for r in raw['frames']};depths=[];hashes={}
    for r in rows:
        f=lookup[r['physical_camera']];norm=normalize_frame(f,raw,metadata)
        for key in ['transform_matrix','fl_x','fl_y','cx','cy']:
            assert np.allclose(norm[key],r[key],atol=1e-6,rtol=0)
        path=CONTROL/'pipeline/dense/stereo/depth_maps'/(f['file_path']+'.geometric.bin')
        d=read_colmap_dense_array(path);assert d.shape==(1080,1920,1)
        depths.append(d[...,0]*metadata['dataparser_scale']);hashes[str(path)]=sha(path)
    return rows,depths,hashes

def diagnose():
    import open3d as o3d
    from scipy import ndimage
    from diffusion_mesh_repair import scene_for
    out=OUT/FRAME;spec=read(out/'input.json');rows,depths,hashes=load_real();lookup={r['physical_camera']:i for i,r in enumerate(rows)}
    mesh=o3d.io.read_triangle_mesh(spec['mesh']);scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles));records=[]
    maps={}
    for name,mask in masks().items():
        index=lookup[name];row=rows[index];pm=depths[index];md=raycast_integer(scene,row)
        y,x=np.nonzero(mask&(pm>0));count,free=support(unproject(row,x,y,pm[y,x]),row,rows,depths)
        votes=np.zeros(pm.shape,np.uint8);votes[y,x]=count
        trusted=mask&(votes>=3)&(md>0)&(np.abs(pm-md)<.001)
        maps[name+'_mesh']=md;maps[name+'_pm']=pm;maps[name+'_trusted']=trusted;maps[name+'_votes']=votes
        rgb=np.array(Image.open(out/'rgb'/(name+'.png')));mark=rgb.copy();mark[mask&(md==0)]=[255,0,0];mark[trusted]=[0,255,0]
        mark=np.rot90(mark).copy();cv2.polylines(mark,[np.array(POLYGONS[name],np.int32)],True,(255,255,0),2)
        Image.fromarray(mark).crop((0,1550,430,1920)).save(out/(name+'_support_native.png'))
        labels,n=ndimage.label(mask&(md==0));components=sorted(np.bincount(labels.ravel())[1:].tolist(),reverse=True)[:10]
        records.append(dict(camera=name,skin_pixels=int(mask.sum()),original_missing=int((mask&(md==0)).sum()),trusted_boundary_pixels=int(trusted.sum()),largest_missing_components=components))
    np.savez_compressed(out/'diagnostic.npz',**maps)
    atomic_json(out/'diagnostic.json',dict(rows=records,real_depth_sha256=hashes,polygons=POLYGONS,polygon_coordinates='native portrait',train_rgb_only=True))
    print(records,flush=True)

def analyze():
    import open3d as o3d
    from scipy import ndimage
    started=time.monotonic();out=OUT/FRAME;spec=read(out/'input.json');rows,depths,hashes=load_real()
    byname={r['physical_camera']:i for i,r in enumerate(rows)};priornames={r['physical_camera']:i for i,r in enumerate(spec['frames'])}
    maps=np.load(out/'diagnostic.npz');priors=np.load(out/'prior_portrait.npz')['depth'];skin=masks();aligned=[];alignedrows=[];alignments=[]
    yy,xx=np.indices((1080,1920));designfull=np.stack((xx/100,yy/100,np.ones_like(xx)),-1)
    for name,mask in skin.items():
        row=rows[byname[name]];pm=depths[byname[name]];md=maps[name+'_mesh'];raw=priors[priornames[name]]
        # Global affine gauge alignment uses measured depths and >=3 other votes.
        y,x=np.nonzero((md>0)&(pm>0)&(np.abs(md-pm)<.001)&(raw>0));take=np.arange(0,len(x),max(1,len(x)//1200));x=x[take];y=y[take]
        votes,_=support(unproject(row,x,y,pm[y,x]),row,rows,depths);x=x[votes>=3];y=y[votes>=3]
        assert len(x)>100
        coef,rmse=robust_fit(np.column_stack((raw[y,x],np.ones(len(x)))),pm[y,x]);base=raw*coef[0]+coef[1]
        # One fixed local residual plane leaves DA3's higher-order shape intact.
        ty,tx=np.nonzero(maps[name+'_trusted']);train=np.arange(len(tx))%5!=0
        residual,lrmse=robust_fit(designfull[ty[train],tx[train]],(pm-base)[ty[train],tx[train]])
        local=base+designfull@residual;aligned.append(local);alignedrows.append(row)
        alignments.append(dict(camera=name,global_scale_shift=coef.tolist(),global_rmse=rmse,global_measured_anchors=len(x),
            local_residual_coefficients=residual.tolist(),local_fit_anchors=int(train.sum()),local_rmse=lrmse,
            held_boundary_mae=float(np.abs(local-pm)[ty[~train],tx[~train]].mean()),held_boundary_count=int((~train).sum())))
    aligned=np.stack(aligned);ref=rows[byname[REFERENCE]];pm=depths[byname[REFERENCE]];md=maps[REFERENCE+'_mesh'];trusted=maps[REFERENCE+'_trusted']
    hole=skin[REFERENCE]&(md==0);hy,hx=np.nonzero(hole);ty,tx=np.nonzero(trusted)
    plane,p_rmse=robust_fit(designfull[ty,tx],1/pm[ty,tx]);planez=1/(designfull[hy,hx]@plane)
    learned=aligned[0,hy,hx];distance=ndimage.distance_transform_edt(~trusted)[hy,hx]
    protocol=dict(frame=FRAME,variants=['plane','da3_local','da3_consistent'],reference=REFERENCE,skin_polygons=POLYGONS,
        minimum_boundary_other_measured_views=3,maximum_boundary_distance_pixels=100,maximum_local_rmse=.0015,
        measured_depth_tolerance=.001,measured_reprojection_pixels=1.5,minimum_parallax_degrees=1,
        maximum_plane_departure=.004,maximum_triangle_extent=.002,minimum_reviewed_skin_views=3,
        trusted_free_space_tolerance=.003,allowed_trusted_free_space_contradictions=0,
        strict_learned_other_views=2,learned_depth_tolerance=.003,learned_reprojection_pixels=3,
        minimum_interior_measured_support=0,existing_vertices_and_triangles_immutable=True,
        confidence_claim='Interior additions may be inferred, not measured; masks are semantic limits, never measured support',
        heldout_camera='F004_B005_1210O9',heldout_use='evaluation only',protocol_frozen_before_heldout_rgb=True)
    atomic_json(OUT/'experiment_request.json',protocol)
    mesh=o3d.io.read_triangle_mesh(spec['mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles);records=[]
    for name,z in [('plane',planez),('da3_local',learned),('da3_consistent',learned)]:
        points=unproject(ref,hx,hy,z);observed,rawfree=support(points,ref,rows,depths)
        learnedvotes,_=support(points,ref,alignedrows,list(aligned),tolerance=.003,reprojection=3.)
        skinvotes=np.zeros(len(z),np.uint8);trustfree=np.zeros(len(z),np.uint8)
        for camera in alignedrows:
            physical=camera['physical_camera'];uv,cz=project_integer(camera,points);xy=np.rint(uv).astype(int)
            inside=(cz>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080);ids=np.flatnonzero(inside)
            qx,qy=xy[ids].T;skinvotes[ids]+=skin[physical][qy,qx]
            trustfree[ids]+=maps[physical+'_trusted'][qy,qx]&(depths[byname[physical]][qy,qx]>cz[ids]+.003)
        eligible=(distance<=100)&(z>0)&np.isfinite(z)&(skinvotes==3)&(trustfree==0)&(np.abs(z-planez)<=.004)
        if name!='plane':eligible&=alignments[0]['local_rmse']<=.0015
        if name=='da3_consistent':eligible&=learnedvotes>=2
        added=np.zeros(md.shape,np.float32);added[hy[eligible],hx[eligible]]=z[eligible];accepted=added>0
        # A one-pixel old-depth ring only supplies shared-boundary tiles. No
        # existing vertex/triangle is modified; long/gap-spanning edges rejected.
        domain=ndimage.binary_dilation(accepted)&((md>0)|accepted);y,x=np.nonzero(domain)
        vertices=unproject(ref,x,y,np.where(accepted,added,md)[y,x]);index=np.full(md.shape,-1,np.int32);index[y,x]=np.arange(len(x))+len(v)
        a=index[:-1,:-1];b=index[:-1,1:];c=index[1:,:-1];d=index[1:,1:];tri=[]
        for aa,bb,cc,h in [(a,b,c,accepted[:-1,:-1]|accepted[:-1,1:]|accepted[1:,:-1]),(b,d,c,accepted[:-1,1:]|accepted[1:,1:]|accepted[1:,:-1])]:
            ok=(aa>=0)&(bb>=0)&(cc>=0)&h;tri.append(np.column_stack((aa[ok],bb[ok],cc[ok])))
        triangles=np.concatenate(tri);vv=np.concatenate((v,vertices));length=np.ptp(vv[triangles],axis=1).max(1)
        triangles=triangles[length<.002];tt=np.concatenate((t,triangles));dest=out/name;dest.mkdir(exist_ok=True)
        result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt));result.compute_vertex_normals();o3d.io.write_triangle_mesh(str(dest/'mesh.ply'),result)
        assert np.array_equal(vv[:len(v)],v) and np.array_equal(tt[:len(t)],t)
        np.savez_compressed(dest/'evidence.npz',depth=added,accepted=accepted,all_candidate_xy=np.column_stack((hx,hy)),observed=observed,raw_free_space=rawfree,trusted_free_space=trustfree,skin_views=skinvotes,learned_other_votes=learnedvotes)
        records.append(dict(variant=name,original_missing_pixels=len(hx),accepted_pixels=int(eligible.sum()),added_triangles=len(triangles),
            accepted_zero_measured_votes=int((eligible&(observed==0)).sum()),accepted_ge2_measured_votes=int((eligible&(observed>=2)).sum()),
            median_observed_votes=float(np.median(observed[eligible])) if eligible.any() else None,
            rejected_skin_limit=int((skinvotes<3).sum()),rejected_trusted_free_space=int((trustfree>0).sum()),
            rejected_plane_departure=int((np.abs(z-planez)>.004).sum()),rejected_distance=int((distance>100).sum()),
            median_learned_other_votes=float(np.median(learnedvotes[eligible])) if eligible.any() else None,
            mesh_sha256=sha(dest/'mesh.ply'),original_geometry_preserved=True))
    np.savez_compressed(out/'aligned.npz',depth=aligned,plane_inverse_coefficients=plane,reference_hole=hole)
    atomic_json(out/'analysis.json',dict(variants=records,alignments=alignments,plane_inverse_rmse=p_rmse,
        measured_boundary_anchors=len(tx),elapsed_seconds=time.monotonic()-started,real_depth_sha256=hashes,
        heldout_used=False,inferred_not_measured=True,baseline_original_unrepaired=True))
    print(records,flush=True)

def views():
    from render_patchmatch_camera_path import normalize_frame
    from joint_temporal_texture import CALIBRATION
    out=OUT/FRAME;spec=read(out/'input.json');base=read(PARENT/'request.json')
    moving=next(r for r in base['inventory'] if r['frame_id']==FRAME)['camera']
    rows,_,_=cameras(FRAME);train=next(r for r in rows if r['physical_camera']=='H004_A005_1210M6')
    cal=read(CALIBRATION);held=next(r for r in cal['frames'] if r['physical_camera']=='F004_B005_1210O9')
    held=normalize_frame(held,cal,read(spec['metadata']))
    return {'moving':moving,'train_H_A':train,'heldout_F_B':held}

def variant_names():
    variants=read(OUT/'experiment_request.json')['variants']
    return variants+(['plane_clipped'] if (OUT/'clip_request.json').exists() else [])

def clip_plane():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    out=OUT/FRAME;spec=read(out/'input.json');rows,depths,_=load_real();old=o3d.io.read_triangle_mesh(spec['mesh'])
    ov=np.asarray(old.vertices);ot=np.asarray(old.triangles);sceneold=scene_for(ov,ot)
    mesh=o3d.io.read_triangle_mesh(str(out/'plane/mesh.ply'));v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles).copy()
    selected={'moving':views()['moving']}
    selected.update({r['physical_camera']:r for r in rows if r['physical_camera'] in POLYGONS})
    olddepth={name:camera_depth(sceneold,camera)[0] for name,camera in selected.items()}
    protocol=dict(parent_protocol_sha256=sha(OUT/'experiment_request.json'),parent_plane_sha256=sha(out/'plane/mesh.ply'),
        authorized_bounded_followup=True,rule='Remove appended triangles hiding old depth by >.001 with >=3 other measured votes; repeat at most 4 passes',
        depth_guard=.001,minimum_old_other_measured_votes=3,cameras=list(selected),heldout_not_used=True,
        old_triangles_never_removed=True,threshold_not_relaxed=True)
    assert not (out/'plane_clipped/mesh.ply').exists()
    rounds=[]
    for iteration in range(4):
        scene=scene_for(v,t);remove=set();checks=[]
        for name,camera in selected.items():
            d,ids,_=camera_depth(scene,camera);od=olddepth[name];front=np.isfinite(od)&np.isfinite(d)&(d<od-.001)
            y,x=np.nonzero(front);votes,_=support(unproject(camera,x,y,od[y,x],offset=.5),camera,rows,depths)
            implicated=ids[y[votes>=3],x[votes>=3]];assert (implicated>=len(ot)).all()
            remove.update(implicated.tolist());checks.append(dict(camera=name,trusted_old_occluded=int((votes>=3).sum())))
        rounds.append(dict(iteration=iteration,removed_triangles=len(remove),checks=checks))
        if not remove:break
        keep=np.ones(len(t),bool);keep[list(remove)]=False;t=t[keep]
    assert np.array_equal(t[:len(ot)],ot)
    dest=out/'plane_clipped';dest.mkdir(exist_ok=True);result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t))
    result.compute_vertex_normals();o3d.io.write_triangle_mesh(str(dest/'mesh.ply'),result)
    atomic_json(OUT/'clip_request.json',protocol)
    atomic_json(dest/'result.json',dict(rounds=rounds,original_added_triangles=len(np.asarray(mesh.triangles))-len(ot),
        retained_added_triangles=len(t)-len(ot),removed_total=len(np.asarray(mesh.triangles))-len(t),mesh_sha256=sha(dest/'mesh.ply'),
        original_vertices_triangles_preserved=True,guard_zero_in_last_pass=all(r['trusted_old_occluded']==0 for r in rounds[-1]['checks'])))
    print(rounds,flush=True)

def geometry_review():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    out=OUT/FRAME;spec=read(out/'input.json');variants=['baseline']+variant_names();records=[]
    for view,row in views().items():
        images=[];depths=[];dest=out/'review'/view;dest.mkdir(parents=True,exist_ok=True)
        for name in variants:
            path=Path(spec['mesh']) if name=='baseline' else out/name/'mesh.ply'
            mesh=o3d.io.read_triangle_mesh(str(path));v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles);mesh.compute_triangle_normals();normal=np.asarray(mesh.triangle_normals)
            depth,ids,bary=camera_depth(scene_for(v,t),row);hit=np.isfinite(depth);d=np.where(hit,depth,0);depths.append(d)
            rgb=np.zeros((1080,1920,3),np.uint8);light=np.array(row['transform_matrix'])[:3,2]
            shade=.2+.8*np.abs(normal[ids[hit]]@light);rgb[hit]=(shade[:,None]*255).clip(0,255).astype(np.uint8)
            image=Image.fromarray(np.rot90(rgb));image.save(dest/(name+'_clay_native.png'));images.append(image)
            if name=='baseline':first=d
            new=(first==0)&hit;front=(first>0)&hit&(d<first-.001)
            region=np.zeros((1920,1080),bool)
            if view=='moving':region[1740:1905,305:445]=True
            else:region[1600:1920,:450]=True
            region=np.rot90(region,-1);query=dict()
            if view=='moving':query=dict(target_query_portrait=[365,1850],target_query_hit=bool(d[365,69]>0),target_query_depth=float(d[365,69]))
            np.savez_compressed(dest/(name+'_depth.npz'),depth=d,newly_visible=new,in_front=front)
            records.append(dict(view=view,variant=name,newly_visible=int(new.sum()),newly_visible_forearm_window=int((new&region).sum()),
                occluding_old_by_001=int(front.sum()),**query))
        box=(180,1580,530,1920) if view=='moving' else (0,1550,430,1920)
        w,h=box[2]-box[0],box[3]-box[1];panel=Image.new('RGB',(w*len(images),h+28));draw=ImageDraw.Draw(panel)
        for i,(im,name) in enumerate(zip(images,variants)):
            panel.paste(im.crop(box),(w*i,28));draw.text((w*i+4,7),name,fill='white')
        panel.save(dest/'clay_comparison_native.png')
    atomic_json(out/'geometry_review.json',dict(rows=records,original_is_unrepaired=True,actual_hole_query_included=True))
    print(records,flush=True)

def render():
    import torch
    from render_smooth_temporal_mesh_video import render_one
    torch.set_num_threads(2)
    out=OUT/FRAME;spec=read(out/'input.json');base=read(PARENT/'request.json')
    record=next(r for r in base['inventory'] if r['frame_id']==FRAME);source=next(r for r in base['source_rows'] if Path(r['source_dataset']).name==FRAME)
    for view,camera in views().items():
        for variant in ['baseline']+variant_names():
            mesh=Path(spec['mesh']) if variant=='baseline' else out/variant/'mesh.ply'
            dest=out/variant/('render_'+view);dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
            r=deepcopy(record);r['mesh']=str(mesh);r['mesh_sha256']=sha(mesh);r['camera']=camera
            request=dict(comparison='isolated append-only forearm pilot',variant=variant,view=view,inventory=[r],source_rows=[source],
                profiles_sha256=spec['profiles_sha256'],exposure_sha256=spec['exposure_sha256'],uses_heldout_rgb=False,
                renderer_sha256=sha(Path(__file__).with_name('render_smooth_temporal_mesh_video.py')))
            if (dest/'request.json').exists():assert read(dest/'request.json')==request
            atomic_json(dest/'request.json',request);render_one(dest,r,source)
            print(f'render complete {view} {variant}',flush=True)

def evaluation_inputs():
    from joint_temporal_texture import SOURCE
    out=OUT/FRAME;assert (OUT/'experiment_request.json').exists()
    exposure=read(ROOT/'exposure.json')['fixed_exposure_gain'];raw=read(SOURCE/FRAME/'transforms.json')
    f=next(r for r in raw['frames'] if r['physical_camera']=='F004_B005_1210O9');path=SOURCE/FRAME/f['file_path']
    rgb=np.rint(255*display(exr(path),exposure)).clip(0,255).astype(np.uint8)
    dest=out/'evaluation';dest.mkdir(exist_ok=True);Image.fromarray(np.rot90(rgb)).save(dest/'heldout_gt_native.png')
    Image.fromarray(np.rot90(rgb)).crop((0,1500,500,1920)).save(dest/'heldout_forearm_gt_crop.png')
    atomic_json(dest/'input.json',dict(heldout_source=str(path),heldout_sha256=sha(path),protocol_sha256=sha(OUT/'experiment_request.json'),evaluation_only=True))

def score():
    import torch
    from score_colmap_patchmatch_tsdf_face import masked_display_metrics
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    out=OUT/FRAME;dest=out/'evaluation';variants=['baseline']+variant_names()
    torch.set_num_threads(2);model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    heldpoly=[(314,1800),(386,1800),(372,1850),(350,1915),(250,1915),(271,1874),(300,1830)]
    heldmask=np.zeros((1920,1080),np.uint8);cv2.fillPoly(heldmask,[np.array(heldpoly,np.int32)],1)
    trainmask=np.rot90(masks()['H004_A005_1210M6']).copy();rows=[]
    for view in views():
        if view=='moving':gt=None;region=None
        elif view=='heldout_F_B':gt=np.asarray(Image.open(dest/'heldout_gt_native.png'));region=heldmask.astype(bool)
        else:gt=np.rot90(np.asarray(Image.open(out/'rgb/H004_A005_1210M6.png')));region=trainmask
        predictions=[];labels=[]
        if gt is not None:predictions.append(gt);labels.append('Held-out GT' if view=='heldout_F_B' else 'Train reference')
        base=np.asarray(Image.open(out/'baseline'/('render_'+view)/'frames'/FRAME/'frame.png'))
        for variant in variants:
            p=out/variant/('render_'+view)/'frames'/FRAME/'frame.png';pred=np.asarray(Image.open(p));predictions.append(pred);labels.append(variant)
            changed=np.any(pred!=base,-1)
            if gt is not None:
                d=np.rot90(np.load(out/'review'/view/'baseline_depth.npz')['depth'])
                regions={'forearm_skin':region}
                if view=='train_H_A':regions['original_missing_skin']=region&(d==0)
                for name,mask in regions.items():
                    with torch.inference_mode():
                        metric=masked_display_metrics(torch.tensor(pred.transpose(2,0,1)/255,dtype=torch.float32,device='cuda'),
                            torch.tensor(gt.transpose(2,0,1)/255,dtype=torch.float32,device='cuda'),torch.tensor(mask,device='cuda'),model)
                    metric={k.removeprefix('face_'):v for k,v in metric.items()}
                    rows.append(dict(view=view,variant=variant,region=name,**metric,pixels=int(mask.sum()),changed_pixels=int((changed&mask).sum()),
                        full_image_changed_pixels=int(changed.sum()),prediction_sha256=sha(p),gt_is_model_input=view=='train_H_A'))
            box=(180,1580,530,1920) if view=='moving' else (0,1550,430,1920)
        w,h=box[2]-box[0],box[3]-box[1];panel=Image.new('RGB',(w*len(predictions),h+28));draw=ImageDraw.Draw(panel)
        for i,(im,label) in enumerate(zip(predictions,labels)):
            panel.paste(Image.fromarray(im).crop(box),(w*i,28));draw.text((w*i+4,7),label,fill='white')
        panel.save(dest/(view+'_rgb_comparison_native.png'))
        if gt is not None:
            review=gt.copy();cv2.polylines(review,[np.array(heldpoly if view=='heldout_F_B' else POLYGONS['H004_A005_1210M6'],np.int32)],True,(255,255,0),2)
            Image.fromarray(review).crop(box).save(dest/(view+'_mask_native.png'))
    atomic_json(OUT/'metrics.json',dict(rows=rows,protocol='Masked RGB PSNR; zero-outside-mask tight bbox SSIM/AlexNet LPIPS',
        heldout_forearm_polygon=heldpoly,heldout_roi_used_for_geometry=False,
        train_metrics_are_reprojection_checks_not_heldout_generalization=True,heldout_lower_patch_out_of_frame=True))
    print(rows,flush=True)

def audit():
    import open3d as o3d
    from joint_temporal_texture import HELD_CAMERAS
    started=time.monotonic();out=OUT/FRAME;spec=read(out/'input.json');rows,depths,hashes=load_real();analysis=read(out/'analysis.json')
    assert hashes==analysis['real_depth_sha256'];assert sha(spec['mesh'])==spec['mesh_sha256']
    assert sha(ROOT/'camera_profiles.json')==spec['profiles_sha256']
    assert sha(ROOT/'exposure.json')==spec['exposure_sha256']
    assert not ({r['physical_camera'] for r in spec['frames']}&HELD_CAMERAS)
    for r in spec['frames']:assert sha(r['source_file_path'])==r['source_sha256']
    old=o3d.io.read_triangle_mesh(spec['mesh']);v=np.asarray(old.vertices);t=np.asarray(old.triangles);records=[]
    candidates=list(analysis['variants'])
    if (OUT/'clip_request.json').exists():
        clipped=read(out/'plane_clipped/result.json');candidates.append(dict(variant='plane_clipped',mesh_sha256=clipped['mesh_sha256'],added_triangles=clipped['retained_added_triangles']))
    for rec in candidates:
        variant=rec['variant'];p=out/variant/'mesh.ply';mesh=o3d.io.read_triangle_mesh(str(p));assert sha(p)==rec['mesh_sha256']
        assert np.array_equal(np.asarray(mesh.vertices)[:len(v)],v) and np.array_equal(np.asarray(mesh.triangles)[:len(t)],t)
        assert len(np.asarray(mesh.triangles))==len(t)+rec['added_triangles']
        for view,camera in views().items():
            baseline=np.load(out/'review'/view/'baseline_depth.npz')['depth'];data=np.load(out/'review'/view/(variant+'_depth.npz'))
            front=data['in_front'];y,x=np.nonzero(front)
            votes,_=support(unproject(camera,x,y,baseline[y,x],offset=.5),camera,rows,depths)
            # Append-only topology does not imply visibility preservation. Audit
            # whether any newly front-facing patch hides measured old surfaces.
            trusted_old=int((votes>=3).sum())
            basepath=out/'baseline'/('render_'+view)/'frames'/FRAME/'frame.png';predpath=out/variant/('render_'+view)/'frames'/FRAME/'frame.png'
            assert predpath.exists() and basepath.exists()
            for rendered in [predpath.parent,basepath.parent]:
                receipt=read(rendered/'complete.json')
                for filename,digest in receipt['hashes'].items():assert sha(rendered/filename)==digest
            records.append(dict(variant=variant,view=view,original_geometry_arrays_preserved=True,
                old_pixels_occluded_by_more_than_001=len(x),occluded_old_pixels_with_ge3_measured_votes=trusted_old,
                prediction_sha256=sha(predpath),mesh_sha256=sha(p),high_confidence_visibility_preserved=trusted_old==0))
    maps=np.load(out/'diagnostic.npz');ty,tx=np.nonzero(maps[REFERENCE+'_trusted']);pm=maps[REFERENCE+'_pm'];train=np.arange(len(tx))%5!=0
    design=np.column_stack((tx/100,ty/100,np.ones(len(tx))));coef,_=robust_fit(design[train],1/pm[ty[train],tx[train]])
    pred=1/(design[~train]@coef);plane_mae=float(np.abs(pred-pm[ty[~train],tx[~train]]).mean())
    atomic_json(OUT/'audit.json',dict(rows=records,array_preservation_passed=True,source_hashes_passed=True,heldout_exclusion_passed=True,
        protocol_sha256=sha(OUT/'experiment_request.json'),script_sha256=sha(__file__),
        boundary_control=dict(plane_withheld_anchor_mae=plane_mae,da3_withheld_anchor_mae=analysis['alignments'][0]['held_boundary_mae'],
            held_boundary_points=analysis['alignments'][0]['held_boundary_count'],not_true_missing_interior_ground_truth=True),
        elapsed_seconds=time.monotonic()-started,visual_acceptance_is_separate=True))
    print(records,flush=True)

def finalize():
    out=OUT/FRAME;metrics=read(OUT/'metrics.json');audit=read(OUT/'audit.json');geometry=read(out/'geometry_review.json')
    assert len(metrics['rows'])==len(variant_names()+['baseline'])*3
    for r in metrics['rows']:
        p=out/r['variant']/('render_'+r['view'])/'frames'/FRAME/'frame.png';assert sha(p)==r['prediction_sha256']
        assert all(np.isfinite(r[k]) for k in ['psnr','ssim','lpips'])
    for r in audit['rows']:
        if r['variant']=='plane_clipped':assert r['high_confidence_visibility_preserved']
    receipts=[read(p) for p in out.glob('*/render_*/frames/001033/result.json')]
    assert len(receipts)==15
    images=['evaluation/moving_rgb_comparison_native.png','evaluation/train_H_A_rgb_comparison_native.png',
        'evaluation/heldout_F_B_mask_native.png','review/moving/clay_comparison_native.png','review/train_H_A/clay_comparison_native.png',
        'review/moving/plane_trusted_occlusion_native.png','review/train_H_A/plane_trusted_occlusion_native.png']
    atomic_json(OUT/'visual_review.json',dict(reviewed_native_images={p:sha(out/p) for p in images},
        decisions={'plane':'Useful smooth partial fill; original 5-pixel attachment guard flags, kept as failed control',
            'da3_local':'Reject: corrugated inferred shape, notch, greater supported-surface occlusion',
            'da3_consistent':'Reject: fragmented fill; consistency did not recover smooth complete anatomy',
            'plane_clipped':'Positive limited single-time canary, not production: fixed-view preservation guard passes, much of hole filled, straight inset edge and cuff gap remain'},
        old_geometry_arrays_immutable=True,heldout_lower_patch_not_visible=True,temporal_transfer_not_tested=True,
        inferred_not_measured=True,production_defaults_changed=False,full_video_started=False,
        metrics_sha256=sha(OUT/'metrics.json'),audit_sha256=sha(OUT/'audit.json'),
        completed_rgb_renders=len(receipts),render_seconds_total=sum(r['elapsed_seconds'] for r in receipts),
        render_seconds_range=[min(r['elapsed_seconds'] for r in receipts),max(r['elapsed_seconds'] for r in receipts)]))
    print('Final receipts passed; 15 renders, 15 metric rows, no production promotion',flush=True)

def conflict_evidence():
    out=OUT/FRAME;rows,depths,_=load_real();records=[]
    for view,camera in views().items():
        base=np.load(out/'review'/view/'baseline_depth.npz')['depth'];candidate=np.load(out/'review'/view/'plane_depth.npz')
        y,x=np.nonzero(candidate['in_front']);votes,_=support(unproject(camera,x,y,base[y,x],offset=.5),camera,rows,depths)
        chosen=votes>=3;ys=y[chosen];xs=x[chosen]
        im=Image.open(out/'plane'/('render_'+view)/'frames'/FRAME/'frame.png');draw=ImageDraw.Draw(im)
        for yy,xx,vote in zip(ys,xs,votes[chosen]):
            px,py=int(yy),int(1919-xx);draw.ellipse((px-5,py-5,px+5,py+5),outline='cyan',width=1)
            records.append(dict(view=view,portrait_xy=[px,py],old_depth=float(base[yy,xx]),new_depth=float(candidate['depth'][yy,xx]),
                depth_difference=float(base[yy,xx]-candidate['depth'][yy,xx]),old_other_measured_votes=int(vote)))
        box=(180,1580,530,1920) if view=='moving' else (0,1550,430,1920);im.crop(box).save(out/'review'/view/'plane_trusted_occlusion_native.png')
    atomic_json(out/'plane_conflicts.json',dict(rows=records,threshold=.001,threshold_not_changed=True))
    print(records,flush=True)

def stage():
    out=OUT/FRAME;out.mkdir(parents=True,exist_ok=True)
    if (out/'input.json').exists():raise ValueError('Inputs already staged; choose a fresh --output root')
    profiles=read(ROOT/'camera_profiles.json');gains=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']))
    exposure=read(ROOT/'exposure.json')['fixed_exposure_gain']
    prefixes={'E004_A005','E004_B005','E004_C005','F004_A005','F004_C005',
              'G004_A005','G004_B005','G004_C005','H004_A005','H004_B005','H004_C005','H004_D005',
              'I004_A005','I004_B005','I004_C005','I004_D005'}
    rows,mesh,meta=cameras(FRAME);rows=[r for r in rows if r['physical_camera'][:9] in prefixes]
    assert len(rows)==16
    staged=[]
    for r in rows:
        p=out/'rgb'/(r['physical_camera']+'.png');p.parent.mkdir(exist_ok=True)
        rgb=np.rint(255*display(exr(r['file_path'])*np.array(gains[r['physical_camera']]),exposure)).clip(0,255).astype(np.uint8)
        Image.fromarray(rgb).save(p)
        staged.append(dict(r,source_file_path=r['file_path'],file_path=str(p),source_sha256=sha(r['file_path'])))
        if r['physical_camera'] in [REFERENCE,'H004_A005_1210M6','H004_C005_1210SZ']:
            Image.fromarray(np.rot90(rgb)).crop((0,1550,430,1920)).save(out/(r['physical_camera']+'_forearm_native.png'))
    atomic_json(out/'input.json',dict(frame=FRAME,frames=staged,mesh=str(mesh),mesh_sha256=sha(mesh),metadata=str(meta),
        profiles_sha256=sha(ROOT/'camera_profiles.json'),exposure_sha256=sha(ROOT/'exposure.json'),heldout_rgb_used=False,script_sha256=sha(__file__)))
    print('staged 16 train views',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['stage','infer','diagnose','analyze','geometry_review','render','evaluation_inputs','score','audit','conflict_evidence','clip_plane','finalize'])
    p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();OUT=a.output;cv2.setNumThreads(2)
    if a.command=='infer':infer(OUT,[FRAME],portrait=True)
    else:globals()[a.command]()
