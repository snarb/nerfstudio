"""Conservative real-train silhouette support experiment; no generated RGB.

Uses cached DeepLab person probabilities, with a wide native-space dilation.
Only unanimous unsupported triangle vertices are candidates, and only triangles
whose removal reveals no deeper geometry in the target may change target RGB.
This is explicitly experimental semantic geometry trimming, not TSDF evidence.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from PIL import Image,ImageDraw
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights,deeplabv3_resnet50
from joint_temporal_texture import read,sha,atomic_json,cameras,ROOT,exr,display,project
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from central_space_temporal_flythrough import OUTPUT


def trim(output,frame,dilation=20,threshold=.1,grabcut=False):
    request=read(output/'request.json');record=next(r for r in request['inventory'] if r['frame_id']==frame)
    source=output/'frames'/frame;target=output/'silhouette_trim_test'/f'{frame}_d{dilation}';target.mkdir(parents=True,exist_ok=True)
    if (target/'result.json').exists():raise FileExistsError('Preserve reviewed proposal; choose a new experiment')
    rows,_,_=cameras(frame);centers=np.array([r['transform_matrix'] for r in rows])[:,:3,3]
    selected=np.argsort(np.linalg.norm(centers-np.array(record['camera']['transform_matrix'])[:3,3],axis=1))[:6]
    chosen=[rows[i] for i in selected]
    profiles=read(ROOT/'camera_profiles.json');gain_by_name=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']))
    gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    def load(row):
        rgb=np.rint(display(exr(row['file_path'])*np.array(gain_by_name[row['physical_camera']]),gain)*255).clip(0,255).astype(np.uint8)
        return np.rot90(rgb).copy()
    with ThreadPoolExecutor(max_workers=4) as pool:images=list(pool.map(load,chosen))
    weights=DeepLabV3_ResNet50_Weights.DEFAULT;model=deeplabv3_resnet50(weights=weights).cuda().eval()
    preprocessing=weights.transforms();person=weights.meta['categories'].index('person');masks=[]
    panel=Image.new('RGB',(960,1136));draw=ImageDraw.Draw(panel)
    for i,(row,image) in enumerate(zip(chosen,images)):
        tensor=preprocessing(Image.fromarray(image)).unsqueeze(0).cuda()
        with torch.inference_mode():
            prediction=model(tensor)['out'];p=F.interpolate(prediction.softmax(1)[:,person:person+1],size=image.shape[:2],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        # Fill interior unknown regions, then dilate rather than erode fine hair.
        mask=(p>threshold).astype(np.uint8)
        if grabcut:
            # The semantic map provides conservative seeds; real image colors
            # refine the boundary. No RGB is generated or used from an eval view.
            reduced=cv2.resize(image,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
            prob=cv2.resize(p,(reduced.shape[1],reduced.shape[0]))
            seed=np.full(prob.shape,cv2.GC_PR_BGD,np.uint8);seed[prob>threshold]=cv2.GC_PR_FGD;seed[prob<.01]=cv2.GC_BGD
            foreground=cv2.erode((prob>.85).astype(np.uint8),np.ones((15,15),np.uint8))>0
            seed[foreground]=cv2.GC_FGD
            cv2.grabCut(reduced,seed,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),3,cv2.GC_INIT_WITH_MASK)
            refined=np.isin(seed,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
            mask=cv2.resize(refined,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
        from scipy.ndimage import binary_fill_holes
        mask=binary_fill_holes(mask).astype(np.uint8)
        mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation+1,2*dilation+1)))
        masks.append(np.rot90(mask,-1).copy());Image.fromarray(mask*255).save(target/f'mask_{i}.png')
        overlay=image.copy();edge=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0;overlay[edge]=[255,0,255]
        x,y=i%3*320,i//3*568;panel.paste(Image.fromarray(overlay).resize((320,544)),(x,y+24));draw.text((x+3,y+4),row['physical_camera'],fill='white')
    del model;torch.cuda.empty_cache();panel.save(target/'real_train_mask_review.png')
    mesh=o3d.io.read_triangle_mesh(record['mesh']);v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    uv,z=project(v,chosen);outside=np.zeros(len(v),np.uint8);inside=np.zeros(len(v),np.uint8)
    for i,mask in enumerate(masks):
        xy=np.rint(uv[i]).astype(np.int32);valid=(z[i]>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
        hit=np.zeros(len(v),bool);hit[valid]=mask[xy[valid,1],xy[valid,0]]>0
        outside+=valid&~hit;inside+=valid&hit
    # Require >=4 of 6 dilated real silhouettes to reject every triangle vertex.
    candidate=(outside[t]>=4).all(1)
    before,ids,_=camera_depth(scene_for(v,t),record['camera'])
    valid=ids<len(t)
    # Restoring one occluder can expose a backing surface for another candidate.
    # Monotone restoration reaches a conservative fixed point; never delete more.
    restoration_rounds=0
    while True:
        after,_,_=camera_depth(scene_for(v,t[~candidate]),record['camera'])
        removed_hit=valid&candidate[np.minimum(ids,len(t)-1)]
        protected=np.unique(ids[removed_hit&np.isfinite(after)])
        if not len(protected):break
        candidate[protected]=False;restoration_rounds+=1
        if restoration_rounds>100:raise ValueError('Conservative trim did not converge')
    rgb=np.array(Image.open(source/'frame.png').convert('RGB'));trimmed=rgb.copy();trimmed[np.rot90(removed_hit)]=0
    Image.fromarray(trimmed).save(target/'frame.png')
    resultmesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[~candidate]));o3d.io.write_triangle_mesh(str(target/'mesh.ply'),resultmesh)
    boxes={'hair':(230,530,970,1130),'face_hand':(270,660,950,1430),'overview':(0,0,1080,1920)}
    for name,box in boxes.items():
        a,b=Image.fromarray(rgb).crop(box),Image.fromarray(trimmed).crop(box)
        if name=='overview':a=a.resize((432,768));b=b.resize((432,768))
        comparison=Image.new('RGB',(a.width*2,a.height+24));comparison.paste(a,(0,24));comparison.paste(b,(a.width,24))
        ImageDraw.Draw(comparison).text((4,5),'Original | real-train conservative silhouette trim',fill='white');comparison.save(target/f'comparison_{name}.png')
    atomic_json(target/'result.json',{'frame':frame,'source_render_sha256':sha(source/'frame.png'),'source_mesh_sha256':record['mesh_sha256'],
                'source_train_rgb':{r['physical_camera']:sha(r['file_path']) for r in chosen},'heldout_rgb_used':False,
                'source_camera_count':6,'person_probability_threshold':threshold,'grabcut_train_color_boundary_refinement':grabcut,
                'native_dilation_pixels':dilation,'required_outside_votes':4,
                'removed_faces':int(candidate.sum()),'removed_target_pixels':int(removed_hit.sum()),'remaining_face_count':int((~candidate).sum()),
                'conservative_restoration_rounds':restoration_rounds,
                'new_rgb_samples_generated':False,'remaining_pixels_byte_identical':True,'removal_reveals_no_deeper_target_surface':True,
                'semantic_masks_are_not_independent_depth_measurements':True,'visual_status':'requires_actual_review',
                'script_sha256':sha(__file__),'mask_weights_sha256':sha(Path(torch.hub.get_dir())/'checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth'),
                'render_sha256':sha(target/'frame.png'),'mesh_sha256':sha(target/'mesh.ply')})
    print(f'frame={frame} removed_faces={candidate.sum()} removed_pixels={removed_hit.sum()}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--frame',default='001047')
    p.add_argument('--dilation',type=int,default=20);p.add_argument('--threshold',type=float,default=.1);p.add_argument('--grabcut',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(2);trim(a.output,a.frame,a.dilation,a.threshold,a.grabcut)
