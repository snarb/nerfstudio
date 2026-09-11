"""Post-hoc mesh-patch inspector and face-only comparisons; never feeds fitting."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import open3d as o3d
import torch
from PIL import Image, ImageDraw
from joint_temporal_texture import (ROOT, CALIBRATION, read, sha, atomic_json, load_frame,
    geometry_paths, load_parameters, project, sample, apply_response, bounded_warp, display)
from bake_joint_temporal_mesh import camera_depth


def inspect(root,frame):
    """Common tangent patches from train images, not target-space blending."""
    from render_patchmatch_camera_path import normalize_frame
    out=root/'frames'/frame/'patch_inspector';out.mkdir(parents=True,exist_ok=True)
    meshpath,meta=geometry_paths(frame);mesh=o3d.io.read_triangle_mesh(str(meshpath))
    scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    cal=read(CALIBRATION)
    query=normalize_frame(next(r for r in cal['frames'] if r['physical_camera']=='F004_B005_1210O9'),cal,read(meta))
    depth,ids,bary=camera_depth(scene,query)
    data=load_frame(root,frame);rows=data['rows'];device=data['images'].device
    profile,static,residual=load_parameters(root,frame,device)
    gain=read(root/'exposure.json')['fixed_exposure_gain']
    positions={'cheek':(840,475),'forehead':(1090,560),'neck':(630,700),'hand':(560,560),'lipstick':(662,570)}
    v=np.asarray(mesh.vertices);tri=np.asarray(mesh.triangles)
    records=[]
    for label,(x,y) in positions.items():
        if not np.isfinite(depth[y,x]):continue
        face=v[tri[ids[y,x]]];b=bary[y,x];center=face.T@np.array([1-b.sum(),*b])
        tangent=face[1]-face[0];tangent/=np.linalg.norm(tangent)
        normal=np.cross(face[1]-face[0],face[2]-face[0]);normal/=np.linalg.norm(normal)
        other=np.cross(normal,tangent)
        yy,xx=np.mgrid[-16:17,-16:17];spacing=depth[y,x]/query['fl_x']
        points=(center+spacing*(xx[...,None]*tangent+yy[...,None]*other)).reshape(-1,3).astype(np.float32)
        uv,z=project(points,rows);uv=torch.tensor(uv.reshape(62,33,33,2),device=device)
        valid=[];quality=[]
        for row in rows:
            origin=np.asarray(row['transform_matrix'])[:3,3];directions=points-origin
            hit=scene.cast_rays(o3d.core.Tensor(np.concatenate((np.broadcast_to(origin,directions.shape),directions),1).astype(np.float32)))['t_hit'].numpy()
            good=(np.abs(hit-1)*np.linalg.norm(directions,axis=1)<spacing*1.5).reshape(33,33)
            direction=origin-center;direction/=np.linalg.norm(direction)
            valid.append(good);quality.append(good.mean()*abs(normal@direction)**4)
        chosen=np.argsort(quality)[-6:][::-1]
        with torch.inference_mode():
            shift=bounded_warp(static,residual,uv)
            raw=sample(data['images'],uv)
            variants={'fixed exposure':raw,'camera profile':apply_response(raw,profile),
                      'profile + registration':apply_response(sample(data['images'],uv+shift),profile)}
        cell=132;panel=Image.new('RGB',(6*cell,3*(cell+26)),(25,25,25));draw=ImageDraw.Draw(panel)
        for j,(name,colors) in enumerate(variants.items()):
            rgb=display(colors.cpu().numpy().transpose(0,2,3,1),gain)
            for k,c in enumerate(chosen):
                patch=rgb[c].copy();patch[~valid[c]]=0
                image=Image.fromarray(np.rint(np.clip(patch,0,1)*255).astype(np.uint8)).resize((cell,cell),Image.Resampling.NEAREST)
                panel.paste(image,(k*cell,j*(cell+26)+26))
                draw.text((k*cell+3,j*(cell+26)),rows[c]['physical_camera'][:10],fill='white')
                draw.text((k*cell+3,j*(cell+26)+12),name,fill='white')
        path=out/f'{label}.png';panel.save(path)
        records.append({'label':label,'query_pixel':[x,y],'world_center':center.tolist(),'tangent_patch_size':33,
                        'native_pixel_equivalent_spacing':float(spacing),'cameras':[rows[c]['physical_camera'] for c in chosen],
                        'visible_fraction':[float(np.mean(valid[c])) for c in chosen],'image_sha256':sha(path)})
    atomic_json(out/'manifest.json',{'inspection_only':True,'eval_rgb_read':False,
                'geometry_query_camera':'F004_B005_1210O9','invalid_geometry_samples_shown_black':True,
                'patches':records})


def score(root,frame,roi):
    from score_colmap_patchmatch_tsdf_face import (load_display_rgb,load_manual_face_mask,
        masked_display_metrics,LearnedPerceptualImagePatchSimilarity)
    target=root/'frames'/frame/'review'/'F004_B005_1210O9'
    gt=load_display_rgb(target/'gt.png');mask,payload=load_manual_face_mask(roi,target/'gt.png',gt.shape[:2])
    net=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    gt_tensor=torch.tensor(gt.transpose(2,0,1),device='cuda');mask_tensor=torch.tensor(mask,device='cuda')
    results={}
    with torch.inference_mode():
        for variant in ['fixed_exposure','camera_profile','joint']:
            pred=load_display_rgb(target/f'{variant}.png')
            results[variant]=masked_display_metrics(torch.tensor(pred.transpose(2,0,1),device='cuda'),gt_tensor,mask_tensor,net)
            results[variant]['prediction_sha256']=sha(target/f'{variant}.png')
    import cv2
    overlay=np.rint(gt*255).astype(np.uint8)
    overlay[cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0]=[255,0,255]
    Image.fromarray(overlay).save(target/'face_roi_overlay.png')
    atomic_json(target/'face_metrics.json',{'frame':frame,'variants':results,
        'ground_truth_sha256':sha(target/'gt.png'),'roi_sha256':sha(roi),
        'protocol':{'roi':'manual_polygon_on_heldout_gt_only','face_psnr':'selected RGB pixels',
                    'face_ssim_lpips':'tight face bbox with both images zero outside face mask',
                    'lpips_network':'alex','display':'frozen exposure + Reinhard + sRGB',
                    'candidate_surface_mask':False,'no_full_frame_metrics':True,
                    'not_comparable_to_prior_per_image_exposure_metrics':True}})
    print(json.dumps({'frame':frame,'variants':results}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['inspect','score'])
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--frame',default='000973')
    p.add_argument('--face-roi',type=Path)
    a=p.parse_args();torch.set_num_threads(8)
    if a.action=='inspect':inspect(a.output,a.frame)
    else:
        if a.face_roi is None:p.error('score requires --face-roi')
        score(a.output,a.frame,a.face_roi)


if __name__=='__main__':main()
