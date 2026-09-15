"""Bounded train-only MHR head20 fit, never replace original COLMAP geometry."""
from pathlib import Path
import argparse,time
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation
from study_multiview_face_prior import read,save,sha,portrait_to_native,CROP
from study_canonical_face_prior import similarity
from triangulate_face_prior import projection_matrices,quantiles

OUT=Path('/mnt/data/dec5_mhr_local_head_prior')
ASSET=Path('/mnt/data/dec5_mhr_head_prior_preflight')
CANONICAL=Path('/mnt/data/dec5_canonical_face_prior')
RGB=Path('/mnt/data/dec5_multiview_face_prior')
FRAME='001193'
SEMANTIC=Path('/mnt/data/dec5_train_hair_semantics/models/selfie_multiclass_256x256.tflite')
# Approximate anatomical correspondences on the fixed neutral-model 600px front
# render. They are model annotations, NOT measured 3D actor landmarks.
MODEL_POINTS={1:(300,236),33:(245,196),133:(278,196),362:(321,196),263:(354,196),61:(270,279),291:(329,279),152:(300,337)}
LANDMARKS=list(MODEL_POINTS)[:-1]

def init():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    if OUT.exists():raise ValueError('Fresh output required')
    OUT.mkdir();neutral=np.load(ASSET/'neutral.npz');v=neutral['vertices'];t=neutral['triangles'];scene=scene_for(v,t)
    xy=np.array(list(MODEL_POINTS.values()),float);origins=np.column_stack(((xy[:,0]-299.5)/600*48,154+(299.5-xy[:,1])/600*48,np.full(len(xy),100.)))
    rays=np.column_stack((origins,np.tile([0,0,-1],(len(xy),1)))).astype(np.float32);hit=scene.cast_rays(o3d.core.Tensor(rays));depth=hit['t_hit'].numpy();assert np.isfinite(depth).all()
    points=origins+depth[:,None]*[0,0,-1];tid=hit['primitive_ids'].numpy();uv=hit['primitive_uvs'].numpy();bary=np.column_stack((1-uv.sum(1),uv))
    c=np.load(CANONICAL/'canonical.npz')['vertices'];sc,rc,tc=similarity(points,c[list(MODEL_POINTS)])
    cp=np.load(CANONICAL/'similarity/fit.npz')['parameters'];rw=Rotation.from_rotvec(cp[:3]).as_matrix();sw=np.exp(cp[6]);tw=cp[3:6]
    scale=sc*sw;rotation=rw@rc;translation=sw*tc@rw.T+tw;world=scale*v@rotation.T+translation
    np.savez_compressed(OUT/'initial.npz',vertices=world,neutral=v,triangles=t,scale=scale,rotation=rotation,translation=translation,landmark_triangles=t[tid[:-1]],landmark_bary=bary[:-1],model_annotation_points=points)
    im=Image.open(ASSET/'review/front.png').copy();draw=ImageDraw.Draw(im)
    for k,(x,y) in MODEL_POINTS.items():draw.ellipse((x-3,y-3,x+3,y+3),fill='red');draw.text((x+4,y),str(k),fill='yellow')
    im.save(OUT/'neutral_annotations.png')
    prior=read(CANONICAL/'protocol.json');source=read(RGB/FRAME/'input.json')
    save(OUT/'protocol.json',dict(frame=FRAME,model_path=str(ASSET/'mhr_model.pt'),model_sha256=sha(ASSET/'mhr_model.pt'),
        asset_receipt_sha256=sha(ASSET/'assets.json'),initial_sha256=sha(OUT/'initial.npz'),canonical_pose_input_sha256=sha(CANONICAL/'similarity/fit.npz'),
        rgb_input_sha256=sha(RGB/FRAME/'input.json'),inference_sha256=sha(RGB/'inference.json'),original_mesh=source['mesh'],original_mesh_sha256=source['mesh_sha256'],
        semantic_path=str(SEMANTIC),semantic_sha256=sha(SEMANTIC),validation_prefixes=prior['validation_prefixes'],
        landmark_indices=LANDMARKS,approximate_neutral_model_annotations=MODEL_POINTS,model_annotations_not_measured_truth=True,
        arms=['similarity','head20'],head_identity_indices=list(range(20,40)),identity_sigma=.5,identity_bound=1.5,
        articulated_pose_frozen_zero=True,expression_frozen_zero=True,body_identity_frozen_zero=True,
        skin_probability_minimum=.9,semantic_erosion_pixels=5,sample_grid=8,max_anchors_per_class_camera=100,
        depth_tolerance=.001,min_other_depth_votes=3,depth_reprojection_pixels=1.5,minimum_parallax_degrees=1,
        initial_model_domain_y=[143,176],initial_model_domain_abs_x=12,maximum_anchor_association_distance=.006,
        fit_outer_iterations=4,fit_inner_iterations=25,point_plane_sigma=.001,point_distance_sigma=.004,landmark_sigma_pixels=4,
        validation_gate=dict(point_plane_p90=.002,landmark_median_pixels=4,landmark_p90_pixels=8),
        locality=dict(surface_distance=.002,boundary_distance=.003),heldout_used=False,model_z_used=False,production_changed=False,
        script_sha256=sha(__file__)))
    print('initialized',len(v),'vertices; annotation only, requires native review',flush=True)

def semantic():
    import mediapipe as mp
    protocol=read(OUT/'protocol.json');assert sha(SEMANTIC)==protocol['semantic_sha256'];dest=OUT/'semantics';dest.mkdir(exist_ok=False)
    options=mp.tasks.vision.ImageSegmenterOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(SEMANTIC),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.IMAGE,output_category_mask=False,output_confidence_masks=True)
    records=[];started=time.monotonic()
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as model:
        for item in read(RGB/FRAME/'input.json')['inputs']:
            assert sha(item['path'])==item['sha256'];name=item['camera']['physical_camera'];rgb=np.array(Image.open(item['path']).convert('RGB'))
            masks=model.segment(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)).confidence_masks
            skin=masks[2].numpy_view().copy()+masks[3].numpy_view().copy();label=np.argmax(np.stack([m.numpy_view() for m in masks]),axis=0).astype(np.uint8)
            np.savez_compressed(dest/(name+'.npz'),skin=np.rint(skin.clip(0,1)*255).astype(np.uint8),label=label)
            records.append(dict(camera=name,input_sha256=item['sha256'],output_sha256=sha(dest/(name+'.npz'))))
    save(dest/'complete.json',dict(records=records,seconds=time.monotonic()-started,mediapipe=mp.__version__,delegate='CPU',protocol_sha256=sha(OUT/'protocol.json')))
    print('semantic CPU complete',len(records),time.monotonic()-started,flush=True)

def anchors():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from study_confidence_depth_prior import load_real,support,unproject
    from joint_temporal_texture import HELD_CAMERAS
    if (OUT/'anchors.npz').exists():raise ValueError('Frozen anchors exist')
    initial=np.load(OUT/'initial.npz');proto=read(OUT/'protocol.json');rows,depths,receipt=load_real(Path('/mnt/data/dec5_jaw_measured_depth/analysis'),FRAME)
    assert not ({r['physical_camera'] for r in rows}&HELD_CAMERAS);val=np.array([any(r['physical_camera'].startswith(p) for p in proto['validation_prefixes']) for r in rows]);fitrows=[r for r,k in zip(rows,val) if not k];fitdepths=[d for d,k in zip(depths,val) if not k]
    mesh=o3d.io.read_triangle_mesh(proto['original_mesh']);mesh.compute_triangle_normals();v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles);norm=np.asarray(mesh.triangle_normals);scene=scene_for(v,t)
    modelscene=scene_for(initial['vertices'],initial['triangles']);predictions={r['camera']:r for r in read(RGB/'inference.json')['records'] if r['frame']==FRAME and r['detected']==1}
    points=[];normals=[];cameras=[];classes=[];votesall=[];pixel=[];lm=[];li=[];lc=[];diagnostics=[];folder=OUT/'anchor_review';folder.mkdir()
    for ci,row in enumerate(rows):
        name=row['physical_camera'];m=np.load(OUT/'semantics'/(name+'.npz'));skin=binary_erosion(m['skin']>=230,iterations=5)
        yy,xx=np.mgrid[0:skin.shape[0]:8,0:skin.shape[1]:8];ok=skin[yy,xx];px=xx[ok];py=yy[ok];xy=portrait_to_native(np.column_stack((px,py+CROP[1]))).astype(int);z=depths[ci][xy[:,1],xy[:,0]]
        world=unproject(row,xy[:,0],xy[:,1],z);model=(world-initial['translation'])@initial['rotation']/initial['scale'];domain=(model[:,1]>143)&(model[:,1]<176)&(abs(model[:,0])<12)&(z>0)
        xy=xy[domain];world=world[domain];z=z[domain];px=px[domain];py=py[domain];model=model[domain]
        center=np.asarray(row['transform_matrix'])[:3,3];direction=unproject(row,xy[:,0],xy[:,1],np.ones(len(xy)))-center
        hit=scene.cast_rays(o3d.core.Tensor(np.column_stack((np.broadcast_to(center,direction.shape),direction)).astype(np.float32)));md=hit['t_hit'].numpy();tid=hit['primitive_ids'].numpy();ok=np.isfinite(md)&(abs(md-z)<=.001)
        xy=xy[ok];world=world[ok];tid=tid[ok];px=px[ok];py=py[ok];model=model[ok]
        other,_=support(world,row,fitrows,fitdepths);close=modelscene.compute_closest_points(o3d.core.Tensor(world.astype(np.float32)))['points'].numpy();distance=np.linalg.norm(close-world,axis=1);ok=(other>=3)&(distance<=.006)
        neck=model[:,1]<153;selected=[]
        for kind in [False,True]:
            available=np.flatnonzero(ok&(neck==kind));selected.extend(available[np.linspace(0,len(available)-1,min(len(available),100),dtype=int)])
        selected=np.array(selected,int);points.extend(world[selected]);normals.extend(norm[tid[selected]]);cameras.extend([ci]*len(selected));classes.extend(neck[selected]);votesall.extend(other[selected]);pixel.extend(xy[selected])
        if name in predictions:
            uv=portrait_to_native(predictions[name]['portrait_xy']);lm.extend(uv[LANDMARKS]);li.extend(range(len(LANDMARKS)));lc.extend([ci]*len(LANDMARKS))
        diagnostics.append(dict(camera=name,validation=bool(val[ci]),face=int((~neck[selected]).sum()),neck_underside=int(neck[selected].sum())))
        if any(name.startswith(p) for p in ['G004_A','G004_B','M004_A','M004_B','E004_B','H004_C']):
            im=Image.open(RGB/FRAME/(name+'.png')).convert('RGB');draw=ImageDraw.Draw(im)
            for j in selected:
                x,y=int(px[j]),int(py[j]);draw.ellipse((x-2,y-2,x+2,y+2),fill='cyan' if neck[j] else 'yellow')
            im.save(folder/(name+'.png'))
    np.savez_compressed(OUT/'anchors.npz',points=np.array(points),normals=np.array(normals),camera=np.array(cameras),neck=np.array(classes),other_votes=np.array(votesall),native_xy=np.array(pixel),validation=val,landmark_uv=np.array(lm),landmark_indices=np.array(li),landmark_camera=np.array(lc))
    save(OUT/'anchors.json',dict(cameras=rows,diagnostics=diagnostics,depth_receipt=receipt,anchors_sha256=sha(OUT/'anchors.npz'),protocol_sha256=sha(OUT/'protocol.json'),semantics_receipt_sha256=sha(OUT/'semantics/complete.json')))
    print('anchors',len(points),'neck/underside',int(np.array(classes).sum()),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','semantic','anchors']);a=p.parse_args();globals()[a.action]()
if __name__=='__main__':main()
