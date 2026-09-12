"""Slow central convex-hull camera loop; render actual static mesh, never image morphs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import subprocess
import os
import time
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from joint_temporal_texture import cameras,sha,read,atomic_json,HELD_CAMERAS
from diffusion_mesh_repair import scene_for,render_atlas,BASE

ANCHORS=['G004_B005_1210FG','I004_B005_1210T5','I004_D005_1210Q7','G004_D005_12107O']


def central_path(rows,subject_center,*,count=360,fps=30,radius=.6):
    if count<30 or fps<=0 or not 0<radius<1/np.sqrt(2):raise ValueError('Invalid smooth loop options')
    by_name={r['physical_camera']:r for r in rows}
    if set(ANCHORS)&HELD_CAMERAS:raise ValueError('Path anchors must be train cameras')
    reference=by_name['H004_C005_1210SZ'];ref=np.array(reference['transform_matrix'])
    corners=np.array([by_name[n]['transform_matrix'] for n in ANCHORS])[:,:3,3]
    def weights(theta):
        signs=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        return (1+radius*np.column_stack((np.cos(theta),np.sin(theta)))@signs.T)/4
    dense_theta=np.linspace(0,2*np.pi,20001);dense_weights=weights(dense_theta);dense_pos=dense_weights@corners
    length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(dense_pos,axis=0),axis=1))]
    if length[-1]<=1e-9:raise ValueError('Degenerate camera trajectory')
    theta=np.interp(np.arange(count)*length[-1]/count,length,dense_theta)
    ww=weights(theta);positions=ww@corners
    # Preserve the reference camera's optical-axis target. The mean of four
    # asymmetric rig centers is NOT the reference center; substituting it here
    # can shift framing completely outside this narrow-FOV calibrated camera.
    focus_depth=float((ref[:3,3]-np.asarray(subject_center))@ref[:3,2])
    if not np.isfinite(focus_depth) or focus_depth<=0:raise ValueError('Subject is behind the reference camera')
    target=ref[:3,3]-ref[:3,2]*focus_depth
    up=ref[:3,1];frames=[]
    for index,pos in enumerate(positions):
        z=pos-target;z/=np.linalg.norm(z);x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x)
        pose=np.eye(4);pose[:3,:3]=np.column_stack((x,y,z));pose[:3,3]=pos
        row=deepcopy(reference);row['transform_matrix']=pose.tolist();row.pop('file_path',None)
        row['physical_camera']=f'smooth_synthetic_{index:05d}';row['convex_weights']=ww[index].tolist();frames.append(row)
    mats=np.array([r['transform_matrix'] for r in frames]);next_mats=np.roll(mats,-1,axis=0)
    speed=np.linalg.norm(next_mats[:,:3,3]-mats[:,:3,3],axis=1)*fps
    angular=Rotation.from_matrix(np.transpose(mats[:,:3,:3],(0,2,1))@next_mats[:,:3,:3]).magnitude()*fps*180/np.pi
    acceleration=np.linalg.norm((np.roll(positions,-1,axis=0)-2*positions+np.roll(positions,1,axis=0))*fps**2,axis=1)
    angles=Rotation.from_matrix(ref[:3,:3].T@mats[:,:3,:3]).magnitude()*180/np.pi
    report={'frames':count,'fps':fps,'duration_seconds':count/fps,'anchors':ANCHORS,'radius':radius,
            'static_source_frame':'000973','mesh_render_per_frame':True,'image_interpolation':False,
            'fixed_intrinsics':True,'continuous_periodic_loop':True,'duplicate_endpoint_frame':False,
            'inside_train_hull':bool((ww>=0).all() and np.allclose(ww.sum(1),1)),
            'minimum_convex_weight':float(ww.min()),'path_length_normalized':float(length[-1]),
            'speed_normalized_min_median_max':np.quantile(speed,[0,.5,1]).tolist(),
            'angular_speed_degrees_per_second_min_median_max':np.quantile(angular,[0,.5,1]).tolist(),
            'acceleration_normalized_max':float(acceleration.max()),'maximum_angle_from_center_degrees':float(angles.max()),
            'loop_seam_step_normalized':float(np.linalg.norm(positions[0]-positions[-1])),
            'loop_seam_angular_speed':float(angular[-1]),'target_world':target.tolist()}
    if not report['inside_train_hull'] or angular.max()>5:raise ValueError('Unsafe or excessively fast camera loop')
    return frames,report


def render_video(asset,output,*,count=360,fps=30,radius=.6,workers=4):
    atlas=dict(np.load(asset/'atlas_geometry.npz'));texture=np.asarray(Image.open(asset/'texture_joint.png').convert('RGB'))
    rows,_,_=cameras('000973');frames,report=central_path(rows,atlas['vertices'].mean(0),count=count,fps=fps,radius=radius)
    output.mkdir(parents=True,exist_ok=True);sequence=output/'frames';sequence.mkdir(exist_ok=True)
    request={'asset':str(asset),'atlas_sha256':sha(asset/'atlas_geometry.npz'),'texture_sha256':sha(asset/'texture_joint.png'),
             'camera_path':frames,'report':report,'script_sha256':sha(__file__),'renderer_sha256':sha(Path(__file__).with_name('diffusion_mesh_repair.py'))}
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Video request mismatch')
    atomic_json(output/'request.json',request);scene=scene_for(atlas['vertices'],atlas['triangles']);start=time.monotonic()
    def checked_render(row):
        rgb,depth,_,_=render_atlas(scene,atlas,texture,row)
        coverage=float(np.isfinite(depth).mean())
        if coverage<.01 or np.count_nonzero(rgb.max(2))<.01*rgb.shape[0]*rgb.shape[1]:
            raise ValueError('Camera framing contains no usable subject surface')
        return rgb,coverage
    # Fail before the batch if a quarter-turn has invalid framing.
    for index in [0,count//4,count//2,3*count//4]:checked_render(frames[index])
    def render_one(item):
        index,row=item;path=sequence/f'{index:05d}.png'
        rgb,coverage=checked_render(row)
        Image.fromarray(np.rot90(rgb)).save(path,compress_level=1)
        return {'index':index,'sha256':sha(path),'bytes':path.stat().st_size,'surface_coverage':coverage}
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(render_one,enumerate(frames)):
            results.append(result)
            if len(results)%30==0:
                atomic_json(output/'progress.json',{'completed':len(results),'total':count,'elapsed_seconds':time.monotonic()-start})
                print(f'video rendered={len(results)}/{count} elapsed={time.monotonic()-start:.1f}',flush=True)
    atomic_json(output/'frame_manifest.json',{'frames':results,'camera_report':report})
    env=dict(os.environ);env['LD_PRELOAD']='/lib/x86_64-linux-gnu/libmpg123.so.0'
    movie=output/'smooth_000973.mp4'
    command=['ffmpeg','-y','-hide_banner','-loglevel','error','-framerate',str(fps),'-i',str(sequence/'%05d.png'),
             '-c:v','libx264','-preset','slow','-crf','18','-pix_fmt','yuv420p','-threads','8','-movflags','+faststart',str(movie)]
    subprocess.run(command,env=env,check=True)
    # Sampled overview sheet complements native-detail review, not replaces it.
    panel=Image.new('RGB',(270*6,480*2+48));draw=ImageDraw.Draw(panel)
    for k,index in enumerate(np.linspace(0,count-1,12,dtype=int)):
        im=Image.open(sequence/f'{index:05d}.png').resize((270,480));x=(k%6)*270;y=(k//6)*504
        panel.paste(im,(x,y+24));draw.text((x+4,y+4),f'{index/fps:.2f}s',fill='white')
    panel.save(output/'contact_sheet.png')
    atomic_json(output/'result.json',{'status':'encoded_pending_visual_review','video':str(movie),'video_sha256':sha(movie),
                'elapsed_seconds':time.monotonic()-start,'camera_report':report,'source_frame_count':1,
                'not_a_150_time_frame_campaign':True,'frame_manifest_sha256':sha(output/'frame_manifest.json')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset',type=Path,default=BASE)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=360)
    p.add_argument('--fps',type=int,default=30);p.add_argument('--radius',type=float,default=.6);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();render_video(a.asset,a.output,count=a.frames,fps=a.fps,radius=a.radius,workers=a.workers)


if __name__=='__main__':main()
