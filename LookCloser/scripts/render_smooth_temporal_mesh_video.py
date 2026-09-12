"""Stream 150 real source instants through fixed-profile hard mesh texturing.

No temporal image morphs, no per-image exposure and no new SfM. Camera motion
is defined once in calibration coordinates, then transferred into each mesh's
recorded normalization. Intermediate frame outputs are never complete receipts.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
import torch
from joint_temporal_texture import ROOT,SOURCE,CALIBRATION,GEOMETRY,HELD_CAMERAS,read,sha,atomic_json,cameras,exr,display,project,sample,bounded_warp,apply_response
from render_patchmatch_camera_path import normalize_frame
from smooth_mesh_flythrough import central_path
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from hard_surface_texture import select_surface_sources,gather_hard_rgb
from finalize_local_mesh_repair import verify_hashes

OUTPUT=Path('/mnt/data/lookcloser_dec5_5a3_smooth_temporal_150_v2')
RECIPE={'frames':150,'fps':30,'radius':.3,'source_count':62,'fixed_profiles':True,
        'per_time_registration':False,'static_registration':True,'averages_rgb':False,
        'graph_smoothness':.08,'graph_color_weight':.5,'graph_iterations':2,
        'label_lowpass_kernel':9,'source_depth_relative_tolerance':.0015,'native_footprint_tolerance':.003}


def calibration_pose(normalized,calibration,metadata):
    """Invert pose normalization without incorrectly scaling its rotation."""
    row=deepcopy(normalized);pose=np.array(row['transform_matrix']);pose[:3,3]/=metadata['dataparser_scale']
    transform=np.eye(4);transform[:3]=metadata['dataparser_transform'];applied=np.eye(4)
    if 'applied_transform' in calibration:applied[:3]=calibration['applied_transform']
    row['transform_matrix']=(applied@np.linalg.inv(transform)@pose).tolist()
    return row


def init(output):
    old=read(GEOMETRY.parent/'campaign_request.json');ids=old['ordered_frame_ids']
    actual=sorted(p.name for p in SOURCE.iterdir() if p.is_dir() and len(p.name)==6 and p.name.isdigit())[:150]
    if ids!=actual or len(set(ids))!=150:raise ValueError('Chronological source inventory mismatch')
    rows,mesh,metadata=cameras('000973');mm=o3d.io.read_triangle_mesh(str(mesh));cal=read(CALIBRATION)
    path,report=central_path(rows,np.asarray(mm.vertices).mean(0),count=150,fps=30,radius=.3)
    raw_path=[calibration_pose(p,cal,read(metadata)) for p in path];inventory=[]
    for index,frame in enumerate(ids):
        _,mp,meta=cameras(frame);original=read(GEOMETRY/frame/'result.json')
        if frame!='000973' and sha(mp)!=original['mesh_sha256']:raise ValueError('Unverified reusable mesh')
        inventory.append({'index':index,'frame_id':frame,'mesh':str(mp),'mesh_sha256':sha(mp),
            'metadata':str(meta),'metadata_sha256':sha(meta),'source_transforms_sha256':sha(SOURCE/frame/'transforms.json'),
            'camera':normalize_frame(raw_path[index],cal,read(meta))})
    scripts=['render_smooth_temporal_mesh_video.py','smooth_mesh_flythrough.py','joint_temporal_texture.py',
             'hard_surface_texture.py','bake_joint_temporal_mesh.py','render_patchmatch_camera_path.py','diffusion_mesh_repair.py']
    request={'recipe':RECIPE,'ordered_frame_ids':ids,'inventory':inventory,'camera_path_report':report,
        'calibration_sha256':sha(CALIBRATION),'profiles_sha256':sha(ROOT/'parameters.npz'),
        'exposure_sha256':sha(ROOT/'exposure.json'),'original_source_manifest_sha256':sha(GEOMETRY.parent/'campaign_request.json'),
        'source_rows':old['source_frames'],'script_hashes':{n:sha(Path(__file__).with_name(n)) for n in scripts},
        'uses_heldout_rgb':False,'previous_goal_turn_classification':'progress'}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Immutable temporal request mismatch')
    atomic_json(output/'request.json',request);(output/'frames').mkdir(exist_ok=True)
    config=output/'config';config.mkdir(exist_ok=True)
    for p in [CALIBRATION,ROOT/'parameters.npz',ROOT/'camera_profiles.json',ROOT/'exposure.json']:
        shutil.copyfile(p,config/p.name)
    atomic_json(output/'progress.json',{'stage':'initialized','frames':150,'complete':0})
    print('initialized=150 actual_times=000899..001197 normalization_transferred=True',flush=True)


def verify_request(output):
    r=read(output/'request.json')
    for name,h in r['script_hashes'].items():
        if sha(Path(__file__).with_name(name))!=h:raise ValueError(f'Frozen render code changed: {name}')
    if sha(CALIBRATION)!=r['calibration_sha256'] or sha(ROOT/'parameters.npz')!=r['profiles_sha256'] or sha(ROOT/'exposure.json')!=r['exposure_sha256']:
        raise ValueError('Frozen calibration/exposure changed')
    return r


def load_sources(rows,source_manifest):
    expected={r['physical_camera']:r['sha256'] for r in source_manifest['source_images']}
    def load(row):
        if row['physical_camera'] in HELD_CAMERAS:raise ValueError('Heldout source leak')
        if sha(row['file_path'])!=expected[row['physical_camera']]:raise ValueError('Source EXR checksum mismatch')
        return exr(row['file_path'])
    with ThreadPoolExecutor(max_workers=8) as pool:images=list(pool.map(load,rows))
    return torch.tensor(np.stack(images).transpose(0,3,1,2),device='cuda')


def render_one(output,record,source_manifest):
    start=time.monotonic();frame=record['frame_id'];target=output/'frames'/frame;target.mkdir(exist_ok=True)
    request_sha=sha(output/'request.json')
    if (target/'complete.json').exists():
        receipt=read(target/'complete.json')
        if receipt['request_sha256']!=request_sha:raise ValueError('Frame resume mismatch')
        verify_hashes(target,receipt['hashes']);return read(target/'result.json')
    if sha(record['mesh'])!=record['mesh_sha256'] or sha(record['metadata'])!=record['metadata_sha256']:
        raise ValueError('Reusable geometry changed')
    if sha(SOURCE/frame/'transforms.json')!=record['source_transforms_sha256']:raise ValueError('Source camera inventory changed')
    def status(stage):
        atomic_json(output/'progress.json',{'frame_id':frame,'index':record['index'],'stage':stage,'pid':os.getpid(),
            'elapsed_seconds':time.monotonic()-start,'utc':datetime.now(timezone.utc).isoformat()})
    status('load_sources');rows,_,_=cameras(frame);images=load_sources(rows,source_manifest)
    parameters=np.load(ROOT/'parameters.npz');profile=torch.tensor(parameters['log_gain'],device='cuda')
    static=torch.tensor(parameters['static_warp'],device='cuda');residual=torch.zeros_like(static)
    gain=read(ROOT/'exposure.json')['fixed_exposure_gain'];mesh=o3d.io.read_triangle_mesh(record['mesh'])
    v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles,np.uint32);tv=v[t]
    normal=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None].clip(1e-12)
    scene=scene_for(v,t);centers=np.array([r['transform_matrix'] for r in rows],np.float32)[:,:3,3]
    status('source_raycast')
    with ThreadPoolExecutor(max_workers=4) as pool:
        depth_array=np.stack(list(pool.map(lambda r:camera_depth(scene,r)[0],rows)))
    depth=torch.tensor(np.where(np.isfinite(depth_array),depth_array,0)[:,None],device='cuda');del depth_array
    def query(points):
        uv,z=project(points,rows);q=torch.tensor(uv[:,None],device='cuda');zq=torch.tensor(z,device='cuda')
        d=sample(depth,q)[:,0,0];valid=(zq>0)&(d>0)&((d-zq).abs()<.0015*zq)
        valid&=(q[:,0,:,0]>2)&(q[:,0,:,0]<1917)&(q[:,0,:,1]>2)&(q[:,0,:,1]<1077)
        return q,zq,valid
    status('surface_labels');centroid=tv.mean(1)
    with torch.inference_mode():
        q,z,valid=query(centroid);directions=centers[:,None]-centroid;length=np.linalg.norm(directions,axis=-1);directions/=length[...,None]
        quality=np.abs((directions*normal).sum(-1))**8/length.clip(.01)**2*valid.cpu().numpy()
        quality=np.where(quality>=quality.max(0)*.12,quality,0)
        low=torch.nn.functional.avg_pool2d(images,9,stride=1,padding=4)
        color=display(apply_response(sample(low,q),profile)[:,:,0].permute(0,2,1).cpu().numpy(),gain);del low
    labels,graph=select_surface_sources(color,quality,t,color_weight=.5)
    np.save(target/'face_source_labels.npy',labels)
    status('target_rgb');d,ids,b=camera_depth(scene,record['camera']);hit=np.isfinite(d)
    if hit.mean()<.01:raise ValueError('Empty camera framing')
    pixels=np.flatnonzero(hit);face=ids[hit];weights=np.column_stack((1-b[hit].sum(1),b[hit]))
    rgb=np.zeros((1080*1920,3),np.float32);chosen_all=np.full(1080*1920,255,np.uint8);fallback_count=0
    with torch.inference_mode():
        for s in range(0,len(pixels),60000):
            end=min(s+60000,len(pixels));f=face[s:end];points=(tv[f]*weights[s:end,:,None]).sum(1)
            q,z,valid=query(points);shifted=q+bounded_warp(static,residual,q);sd=sample(depth,shifted)[:,0,0]
            safe=(sd>0)&((sd-z).abs()<.0025*z)
            for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
                tap=sample(depth,q.floor()+q.new_tensor([dx,dy]))[:,0,0]
                valid&=(tap>0)&((tap-z).abs()<.003*z)
                tap=sample(depth,shifted.floor()+q.new_tensor([dx,dy]))[:,0,0]
                safe&=(tap>0)&((tap-z).abs()<.003*z)
            shifted=torch.where(safe[:,None,:,None],shifted,q)
            direction=centers[:,None]-points;length=np.linalg.norm(direction,axis=-1);direction/=length[...,None]
            quality=np.abs((direction*normal[f]).sum(-1))**8/length.clip(.01)**2
            colors=apply_response(sample(images,shifted),profile)[:,:,0].clamp_min(0)
            chosen,source,fallback=gather_hard_rgb(colors,torch.tensor(quality,device='cuda')*valid,torch.tensor(labels[f],device='cuda'))
            rgb[pixels[s:end]]=display(chosen.T.cpu().numpy(),gain);chosen_all[pixels[s:end]]=source.cpu().numpy().astype(np.uint8)
            fallback_count+=int(fallback.sum())
    rgb=rgb.reshape(1080,1920,3);png=np.rint(rgb*255).clip(0,255).astype(np.uint8)
    if not np.isfinite(rgb).all() or (png.max(2)>0).mean()<.01:raise ValueError('Invalid RGB render')
    Image.fromarray(png).save(target/'prediction_native.png');Image.fromarray(np.rot90(png)).save(target/'frame.png',compress_level=1)
    Image.fromarray(chosen_all.reshape(1080,1920)).save(target/'source_ids.png')
    np.savez_compressed(target/'target_depth.npz',depth=np.where(hit,d,0))
    result={'frame_id':frame,'index':record['index'],'mesh_sha256':record['mesh_sha256'],'mesh_path':record['mesh'],
        'render_sha256':sha(target/'frame.png'),'camera':record['camera'],'source_cameras':[r['physical_camera'] for r in rows],
        'fixed_exposure':gain,'mesh_hit_fraction':float(hit.mean()),'rgb_coverage':float((png.max(2)>0).mean()),
        'fallback_pixels':fallback_count,'graph':graph,'elapsed_seconds':time.monotonic()-start,'visual_status':'pending',
        'source_time_frame_count':1,'target_rgb_read':False,'rgb_averaging':False}
    atomic_json(target/'result.json',result)
    atomic_json(target/'complete.json',{'request_sha256':request_sha,'hashes':{p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file() and p.name!='complete.json'}})
    print(f'frame={frame} index={record["index"]} seconds={result["elapsed_seconds"]:.1f} coverage={result["rgb_coverage"]:.4f}',flush=True)
    del images,depth;torch.cuda.empty_cache();return result


def render(output,ids=None):
    request=verify_request(output);source={Path(r['source_dataset']).name:r for r in request['source_rows']}
    if sorted(source)!=request['ordered_frame_ids']:raise ValueError('Source manifest time inventory mismatch')
    for record in request['inventory']:
        if ids is not None and record['frame_id'] not in ids:continue
        render_one(output,record,source[record['frame_id']])
    atomic_json(output/'progress.json',{'stage':'render_request_finished','complete':len(list((output/'frames').glob('*/complete.json'))),'pid':os.getpid()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','render']);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--frames',nargs='+');a=p.parse_args();torch.set_num_threads(8)
    if a.action=='init':init(a.output)
    else:render(a.output,a.frames)


if __name__=='__main__':main()
