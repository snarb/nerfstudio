"""Trace one reviewed target pixel to its actual single train RGB source.

The rectangle/pixel is diagnostic only: it is never a prediction or scoring mask.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras,ROOT,exr,display,project,bounded_warp
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth


def trace(output,frame,pixel):
    directory=output/'frames'/frame;result=read(directory/'result.json');rows,_,_=cameras(frame)
    mesh=o3d.io.read_triangle_mesh(result['mesh_path']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    scene=scene_for(v,t);depth,ids,b=camera_depth(scene,result['camera'])
    px,py=pixel;x,y=1919-py,px;face=int(ids[y,x]);source=int(np.array(Image.open(directory/'source_ids.png'))[y,x])
    if face>=len(t) or source>=62:raise ValueError('Diagnostic pixel has no mesh/source')
    w=np.r_[1-b[y,x].sum(),b[y,x]];point=(v[t[face]]*w[:,None]).sum(0)
    uv,z=project(point[None],rows);q=torch.tensor(uv[:,None]);static=torch.tensor(np.load(ROOT/'parameters.npz')['static_warp'])
    shifted=(q+bounded_warp(static,torch.zeros_like(static),q)).numpy()[:,0]
    row=rows[source];sd,_,_=camera_depth(scene,row);sd=np.where(np.isfinite(sd),sd,0)
    guard=output/'foreground_guard'/frame/'masks.npz'
    if guard.exists():sd[np.load(guard)['masks'][source]==0]=0
    def at(array,coord):return cv2.remap(array,np.asarray(coord,np.float32).reshape(1,1,2),None,cv2.INTER_LINEAR)[0,0]
    qq=shifted[source,0];zz=z[source,0];d=at(sd,qq);safe=d>0 and abs(d-zz)<.0025*zz
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        d=at(sd,np.floor(qq)+[dx,dy]);safe &= d>0 and abs(d-zz)<.003*zz
    coord=qq if safe else uv[source,0]
    profiles=read(ROOT/'camera_profiles.json');response=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']))
    gain=read(ROOT/'exposure.json')['fixed_exposure_gain'];linear=exr(row['file_path'])*np.array(response[row['physical_camera']])
    color=display(at(linear.astype(np.float32),coord),gain)
    rgb=np.rint(display(linear,gain)*255).clip(0,255).astype(np.uint8)
    train=Image.fromarray(np.rot90(rgb));train_xy=[float(coord[1]),float(1919-coord[0])]
    pred=Image.open(directory/'frame.png').convert('RGB');target=output/'pixel_traces'/frame;target.mkdir(parents=True,exist_ok=True)
    panel=Image.new('RGB',(800,424));draw=ImageDraw.Draw(panel)
    for j,(im,xy,label) in enumerate([(pred,pixel,'target artifact pixel'),(train,train_xy,row['physical_camera'])]):
        cx,cy=map(int,np.rint(xy));crop=im.crop((cx-100,cy-100,cx+100,cy+100)).resize((400,400),Image.Resampling.NEAREST)
        marker=ImageDraw.Draw(crop);marker.line((194,200,206,200),fill='red',width=1);marker.line((200,194,200,206),fill='red',width=1)
        panel.paste(crop,(j*400,24));draw.text((j*400+3,4),label,fill='white')
    panel.save(target/'source_trace.png')
    labels=np.load(directory/'face_source_labels.npy')
    atomic_json(target/'trace.json',dict(frame_id=frame,target_pixel_portrait=list(pixel),source_camera=row['physical_camera'],
        source_file=row['file_path'],source_file_sha256=sha(row['file_path']),mesh_sha256=result['mesh_sha256'],face_id=face,
        target_rgb=np.asarray(pred)[py,px].tolist(),reconstructed_source_rgb_255=(color*255).tolist(),
        source_pixel_native=coord.tolist(),source_pixel_portrait=train_xy,registration_applied=bool(safe),
        chosen_source_index=source,preferred_face_source=int(labels[face]),fallback=bool(source!=labels[face]),
        rgb_source_count=1,averaging=False,source_coordinates_opencv_interpolation_precision=True,
        script_sha256=sha(__file__),evidence_sha256=sha(target/'source_trace.png')))
    print(target/'trace.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--frame',required=True)
    p.add_argument('--pixel',type=int,nargs=2,required=True);a=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(2);trace(a.output,a.frame,a.pixel)
