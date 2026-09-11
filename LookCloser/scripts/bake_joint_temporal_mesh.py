"""Bake a portable, self-contained GLB from jointly calibrated train textures.

The output is a fixed UV texture, independent of the runtime viewing camera.
Raw EXR sources, profiles, and bounded registration are sampled before baking.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import zipfile
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
import torch
import trimesh
import xatlas
from joint_temporal_texture import (
    ROOT,SOURCE,CALIBRATION,HELD_CAMERAS,atomic_json,read,sha,display,save_png,
    cameras,geometry_paths,load_frame,load_parameters,project,sample,bounded_warp,apply_response,
)


def atlas_geometry(mesh_path,output,resolution=4096):
    output.mkdir(parents=True,exist_ok=True)
    path=output/'atlas_geometry.npz'
    if path.exists():
        record=read(output/'atlas_request.json')
        if record['mesh_sha256']!=sha(mesh_path) or record['resolution']!=resolution or 'unwrap_scale' not in record:raise ValueError('Atlas request mismatch')
        if record.get('npz_sha256')!=sha(path):raise ValueError('Atlas cache checksum mismatch')
        return dict(np.load(path))
    mesh=o3d.io.read_triangle_mesh(str(mesh_path));v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles,np.uint32)
    # xatlas has absolute geometric tolerances. TSDF triangles in normalized
    # nerfstudio coordinates can have area <1e-7, falsely becoming degenerates.
    # Only parameterization uses this similarity; exported geometry is unchanged.
    extent=float(np.ptp(v,axis=0).max())
    if not np.isfinite(extent) or extent<=0:raise ValueError('Degenerate mesh')
    unwrap_scale=100./extent
    atlas=xatlas.Atlas();atlas.add_mesh((v-v.mean(0))*unwrap_scale,t)
    pack=xatlas.PackOptions();pack.resolution=resolution;pack.padding=6;pack.bilinear=True
    atlas.generate(pack_options=pack)
    mapping,indices,uv=atlas[0]
    if not np.array_equal(mapping[indices],t):raise ValueError('xatlas changed oriented triangle inventory')
    width,height=atlas.width,atlas.height
    plane=np.column_stack((uv,np.zeros(len(uv),np.float32))).astype(np.float32)
    scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(plane),o3d.core.Tensor(indices.astype(np.uint32)))
    pix=[];faces=[];weights=[]
    for y in range(0,height,128):
        yy,xx=np.mgrid[y:min(y+128,height),0:width]
        origin=np.stack(((xx+.5)/width,1-(yy+.5)/height,np.ones_like(xx)),axis=-1).astype(np.float32)
        rays=np.concatenate((origin,np.broadcast_to([0,0,-1],origin.shape)),axis=-1).astype(np.float32)
        hit=scene.cast_rays(o3d.core.Tensor(rays));ids=hit['primitive_ids'].numpy();b=hit['primitive_uvs'].numpy()
        good=ids!=np.iinfo(np.uint32).max
        pix.append((yy[good]*width+xx[good]).astype(np.int32));faces.append(ids[good])
        weights.append(np.column_stack((1-b[good].sum(1),b[good])).astype(np.float32))
    a={'vertices':v,'triangles':t,'mapping':mapping,'indices':indices,'uv':uv,
       'pixels':np.concatenate(pix),'face_ids':np.concatenate(faces),'barycentric':np.concatenate(weights),
       'width':width,'height':height}
    np.savez_compressed(path,**a)
    atomic_json(output/'atlas_request.json',{'mesh_sha256':sha(mesh_path),'resolution':resolution,
                'width':width,'height':height,'covered_texels':len(a['pixels']),
                'triangles':len(t),'geometry_changed':False,'unwrap_scale':unwrap_scale,
                'npz_sha256':sha(path)})
    return a


def camera_depth(scene, row):
    pose=np.asarray(row['transform_matrix']);ext=np.linalg.inv(pose@np.diag([1.,-1.,-1.,1.])).astype(np.float32)
    k=np.array([[row['fl_x'],0,row['cx']],[0,row['fl_y'],row['cy']],[0,0,1]],np.float32)
    rays=scene.create_rays_pinhole(o3d.core.Tensor(k),o3d.core.Tensor(ext),row['w'],row['h'])
    hit=scene.cast_rays(rays)
    return hit['t_hit'].numpy(),hit['primitive_ids'].numpy(),hit['primitive_uvs'].numpy()


def robust_fusion(colors,weights):
    """Calibrated linear radiance; robust weights suppress isolated highlights."""
    w=weights[:,None]
    log=colors.clamp_min(1e-7).log()
    center=(log*w).sum(0)/w.sum(0).clamp_min(1e-8)
    for _ in range(3):
        distance=(log-center[None]).square().mean(1).sqrt()
        rw=w/(1+(distance[:,None]/.12).square())
        center=(log*rw).sum(0)/rw.sum(0).clamp_min(1e-8)
    rgb=(colors*rw).sum(0)/rw.sum(0).clamp_min(1e-8)
    return rgb


def bake(root,frame,resolution):
    out=root/'frames'/frame;out.mkdir(parents=True,exist_ok=True)
    mesh_path,meta=geometry_paths(frame)
    a=atlas_geometry(mesh_path,out,resolution)
    data=load_frame(root,frame);rows=data['rows'];device=data['images'].device
    profile,static,residual=load_parameters(root,frame,device)
    if sha(mesh_path)!=read(root/'cache'/frame/'request.json')['mesh_sha256']:raise ValueError('Geometry changed after fitting')
    if [r['physical_camera'] for r in rows]!=read(root/'camera_profiles.json')['physical_cameras']:raise ValueError('Texture/profile camera mismatch')
    gain=read(root/'exposure.json')['fixed_exposure_gain']
    scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(a['vertices']),o3d.core.Tensor(a['triangles'].astype(np.uint32)))
    # A full native depth image permits conservative footprint checks after UV shifts.
    depths=[]
    for i,row in enumerate(rows):
        z,_,_=camera_depth(scene,row);depths.append(np.where(np.isfinite(z),z,0))
        if i%16==0:print(f'bake frame={frame} visibility_camera={i+1}/62',flush=True)
    depth=torch.tensor(np.stack(depths)[:,None],device=device)
    h,w=int(a['height']),int(a['width']);size=h*w
    variants={name:np.zeros((size,3),np.float32) for name in ['fixed_exposure','camera_profile','joint']}
    support=np.zeros(size,np.uint8)
    source_map=np.full(size,255,np.uint8)
    v,t=a['vertices'],a['triangles'];tv=v[t]
    normal=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None].clip(1e-12)
    centers=np.array([r['transform_matrix'] for r in rows],np.float32)[:,:3,3]
    start=time.monotonic();count=len(a['pixels'])
    with torch.inference_mode():
        for start_idx in range(0,count,60000):
            end=min(start_idx+60000,count);ids=a['face_ids'][start_idx:end];b=a['barycentric'][start_idx:end]
            points=(tv[ids]*b[...,None]).sum(1)
            uv,z=project(points,rows);uv_t=torch.tensor(uv[:,None],device=device);z_t=torch.tensor(z,device=device)
            shifted=uv_t+bounded_warp(static,residual,uv_t)
            # Surface support and depth discontinuities use original geometry for all variants.
            sampled_z=sample(depth,uv_t)[:,0,0]
            shifted_z=sample(depth,shifted)[:,0,0]
            valid=(z_t>0)&(uv_t[:,0,:,0]>2)&(uv_t[:,0,:,0]<1917)&(uv_t[:,0,:,1]>2)&(uv_t[:,0,:,1]<1077)
            valid&=(sampled_z>0)&((sampled_z-z_t).abs()<z_t*.0015)
            safe=(shifted_z>0)&((shifted_z-z_t).abs()<z_t*.0025)
            # Reject a bilinear footprint which crosses an occluder, even if its mean depth agrees.
            base_uv=uv_t.floor()
            for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
                ztap=sample(depth,base_uv+base_uv.new_tensor([dx,dy]))[:,0,0]
                valid&=(ztap>0)&((ztap-z_t).abs()<z_t*.003)
            base_shift=shifted.floor()
            for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
                ztap=sample(depth,base_shift+base_shift.new_tensor([dx,dy]))[:,0,0]
                safe&=(ztap>0)&((ztap-z_t).abs()<z_t*.003)
            shifted=torch.where(safe[:,None,:,None],shifted,uv_t)
            dirs=centers[:,None]-points[None];length=np.linalg.norm(dirs,axis=-1);dirs/=length[...,None]
            cosine=np.abs((dirs*normal[ids][None]).sum(-1))
            quality=cosine**8/length.clip(.01)**2
            weights=torch.tensor(quality,device=device)*valid
            # Restrict broad angle mixtures to nearby high-quality observations.
            threshold=weights.max(0).values*.12
            weights=torch.where(weights>=threshold[None],weights,0)
            raw=sample(data['images'],uv_t)[:,:,0].clamp_min(0)
            corrected=apply_response(raw[:,:,:,None],profile)[...,0]
            aligned=apply_response(sample(data['images'],shifted),profile)[:,:,0].clamp_min(0)
            pixels=a['pixels'][start_idx:end]
            for name,rgb in [('fixed_exposure',raw),('camera_profile',corrected),('joint',aligned)]:
                fusion=robust_fusion(rgb,weights).T.cpu().numpy()
                variants[name][pixels]=display(fusion,gain)
            support[pixels]=(weights>0).sum(0).clamp(0,254).cpu().numpy().astype(np.uint8)
            selected=weights.argmax(0).cpu().numpy().astype(np.uint8);selected[(weights.sum(0)==0).cpu().numpy()]=255
            source_map[pixels]=selected
            if start_idx//60000%20==0:print(f'bake frame={frame} texels={end}/{count} seconds={time.monotonic()-start:.1f}',flush=True)
    occupied=np.zeros(size,bool);occupied[a['pixels']]=True;occupied=occupied.reshape(h,w)
    # Padding outside UV charts prevents filtering black atlas gutters. Missing
    # observed surface texels remain explicitly unfilled and counted in the audit.
    from scipy.ndimage import distance_transform_edt
    distance,nearest=distance_transform_edt(~occupied,return_indices=True)
    pad=(~occupied)&(distance<=8)
    for name,flat in variants.items():
        rgb=flat.reshape(h,w,3);rgb[pad]=rgb[nearest[0][pad],nearest[1][pad]]
        save_png(out/f'texture_{name}.png',rgb)
    np.savez_compressed(out/'texture_support.npz',support=support.reshape(h,w),source_map=source_map.reshape(h,w))
    atomic_json(out/'bake_result.json',{'frame':frame,'status':'baked_pending_visual_review',
                'mesh_sha256':sha(mesh_path),'parameters_sha256':sha(root/'parameters.npz'),
                'exposure_sha256':sha(root/'exposure.json'),'geometry_changed':False,
                'scripts_sha256':{Path(__file__).name:sha(__file__),
                    'joint_temporal_texture.py':sha(Path(__file__).with_name('joint_temporal_texture.py'))},
                'adaptation_sha256':sha(root/'adaptations'/frame/'residual.npz') if (root/'adaptations'/frame/'result.json').exists() else None,
                'fixed_texture_no_runtime_view_selection':True,'uses_eval_rgb':False,
                'covered_texels':count,'unobserved_texels':int((support[a['pixels']]==0).sum()),
                'texture_sha256':{n:sha(out/f'texture_{n}.png') for n in variants}})
    export_asset(out,a,frame,meta)
    return a


def export_asset(out,a,frame,meta):
    texture=Image.open(out/'texture_joint.png').convert('RGB')
    v=a['vertices'][a['mapping']].copy();v-=a['vertices'].mean(0)
    # Nerfstudio's up is +Z; glTF's up is +Y. This rigid export transform is saved.
    rotation=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32);v=v@rotation.T
    material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture,metallicFactor=0,roughnessFactor=1,doubleSided=True)
    mesh=trimesh.Trimesh(vertices=v,faces=a['indices'],process=False,
        visual=trimesh.visual.TextureVisuals(uv=a['uv'],material=material))
    def unlit(tree):
        tree.setdefault('extensionsUsed',[]).append('KHR_materials_unlit')
        for m in tree.get('materials',[]):m.setdefault('extensions',{})['KHR_materials_unlit']={}
    glb=trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh),tree_postprocessor=unlit)
    glb_path=out/f'dec5_{frame}_joint_calibrated.glb';glb_path.write_bytes(glb)
    # OBJ alongside GLB for applications with older importers.
    obj=trimesh.exchange.obj.export_obj(mesh,include_texture=True,return_texture=True)
    text,assets=obj;(out/'mesh.obj').write_text(text)
    for name,data in assets.items():(out/name).write_bytes(data)
    archive=out/f'dec5_{frame}_joint_calibrated_obj.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out/'mesh.obj','mesh.obj')
        for name in assets:z.write(out/name,name)
    imported=trimesh.load(glb_path,force='scene',process=False)
    loaded=list(imported.geometry.values())[0]
    if len(loaded.faces)!=len(a['triangles']) or not np.isfinite(loaded.vertices).all():raise ValueError('Invalid GLB round-trip')
    if np.max(np.abs(loaded.vertices-v))>1e-6:raise ValueError('GLB geometry round-trip mismatch')
    if not np.allclose(loaded.visual.uv,a['uv'],atol=1e-6):raise ValueError('GLB texture-coordinate round-trip mismatch')
    atomic_json(out/'asset_manifest.json',{'glb':str(glb_path),'glb_sha256':sha(glb_path),
                'obj_zip':str(archive),'obj_zip_sha256':sha(archive),'triangles':len(mesh.faces),
                'embedded_texture_size':list(texture.size),'material':'KHR_materials_unlit',
                'coordinates':'centered normalized reconstruction, Z-up rotated to glTF Y-up',
                'rotation':rotation.tolist(),'original_center':a['vertices'].mean(0).tolist(),
                'geometry_changed':False,'roundtrip_vertices_max_error':float(np.max(np.abs(loaded.vertices-v)))})


def render_review(root,frame):
    from render_patchmatch_camera_path import normalize_frame
    out=root/'frames'/frame;a=dict(np.load(out/'atlas_geometry.npz'));mesh,meta=geometry_paths(frame)
    cal=read(CALIBRATION);metadata=read(meta);gain=read(root/'exposure.json')['fixed_exposure_gain']
    scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(a['vertices']),o3d.core.Tensor(a['triangles'].astype(np.uint32)))
    textures={n:np.asarray(Image.open(out/f'texture_{n}.png').convert('RGB')) for n in ['fixed_exposure','camera_profile','joint']}
    uv=a['uv'][a['indices']];source=read(SOURCE/frame/'transforms.json')
    views=[]
    for camera in sorted(HELD_CAMERAS):
        row=normalize_frame(next(f for f in cal['frames'] if f['physical_camera']==camera),cal,metadata)
        depth,ids,b=camera_depth(scene,row);valid=ids!=np.iinfo(np.uint32).max
        weights=np.column_stack((1-b[valid].sum(1),b[valid]));pixel_uv=(uv[ids[valid]]*weights[...,None]).sum(1)
        target=out/'review'/camera;target.mkdir(parents=True,exist_ok=True)
        renders={}
        for name,texture in textures.items():
            hh,ww=texture.shape[:2]
            u=(pixel_uv[:,0]*ww-.5).astype(np.float32);v=((1-pixel_uv[:,1])*hh-.5).astype(np.float32)
            # OpenCV remap limits dimensions to <32767, so compact chunks are rows.
            color=[]
            for start in range(0,len(u),16000):
                color.append(cv2.remap(texture,u[start:start+16000][None],v[start:start+16000][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0])
            rgb=np.zeros((1080,1920,3),np.uint8);rgb[valid]=np.concatenate(color)
            Image.fromarray(rgb).save(target/f'{name}.png');renders[name]=rgb
        # Held-out RGB read occurs only here, after parameters and textures are finalized.
        original=next(f for f in source['frames'] if f['physical_camera']==camera)
        from joint_temporal_texture import exr
        gt=np.rint(display(exr(SOURCE/frame/original['file_path']),gain)*255).clip(0,255).astype(np.uint8)
        Image.fromarray(gt).save(target/'gt.png')
        crops={'hand_neck':(430,430,1060,970),'face_ear':(650,400,1150,950),'lipstick':(687,540,987,800)}
        if camera=='F004_B005_1210O9':
            crops['ear_native']=(805,315,1040,480) if frame=='000973' else (835,360,1060,520)
            crops['tube_hand_native']=(440,485,760,705)
        # Portrait orientation is for inspection only; calibration/native files
        # and metrics retain the original 1920x1080 pixel coordinates.
        overview=Image.new('RGB',(432*4,768+25));overview_draw=ImageDraw.Draw(overview)
        for i,(label,rgb) in enumerate([('GT',gt),*renders.items()]):
            overview.paste(Image.fromarray(rgb).transpose(Image.Transpose.ROTATE_90).resize((432,768)),(i*432,25))
            overview_draw.text((i*432+4,5),label,fill='white')
        overview.save(target/'overview_upright.png')
        for name,box in crops.items():
            x0,y0,x1,y1=box;ww=x1-x0;hh=y1-y0;panel=Image.new('RGB',(ww*4,hh+25));draw=ImageDraw.Draw(panel)
            for i,(label,rgb) in enumerate([('GT',gt),*renders.items()]):
                panel.paste(Image.fromarray(rgb[y0:y1,x0:x1]),(i*ww,25));draw.text((i*ww+4,5),label,fill='white')
            panel.save(target/f'compare_{name}.png')
        views.append({'camera':camera,'gt_sha256':sha(target/'gt.png'),'joint_sha256':sha(target/'joint.png')})
    atomic_json(out/'review'/'manifest.json',{'heldout_rgb_used_only_for_review':True,'views':views})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['bake','review'])
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--frame',default='000973');p.add_argument('--resolution',type=int,default=4096)
    a=p.parse_args();torch.set_num_threads(8)
    if a.action=='bake':bake(a.output,a.frame,a.resolution)
    else:render_review(a.output,a.frame)


if __name__=='__main__':main()
