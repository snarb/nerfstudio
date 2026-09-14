"""Static-object camera-control pilots: measured 3x3 / 4x4 grid extents.

Existing immutable atlas, no training, generated RGB, temporal actor motion or
production renderer changes. Periodic cubic spline in a bilinear train hull.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import time
import numpy as np
from PIL import Image,ImageDraw
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from joint_temporal_texture import cameras,read,sha,atomic_json,HELD_CAMERAS,CALIBRATION
from diffusion_mesh_repair import BASE,scene_for,render_atlas
from render_smooth_temporal_mesh_video import calibration_pose
from render_patchmatch_camera_path import normalize_frame

OUTPUT=Path('/mnt/data/dec5_camera_grid_diagnosis_v2')
OLD=Path('/mnt/data/lookcloser_dec5_5a3_central_space_flight_150')


def spline_weights(count,*,endpoint=False):
    theta=np.linspace(0,2*np.pi,9)
    controls=.5+.48*np.column_stack((np.cos(theta),np.sin(theta)))
    controls[-1]=controls[0]
    spline=CubicSpline(theta,controls,bc_type='periodic')
    uv=spline(np.linspace(0,2*np.pi,count,endpoint=endpoint))
    if (uv<0).any() or (uv>1).any():raise ValueError('Spline escaped train hull')
    u,v=uv.T
    return np.column_stack(((1-u)*(1-v),u*(1-v),u*v,(1-u)*v)),uv


def grid_path(rows,center,size,count):
    # 3x3 = G..I / B..D. 4x4 = F..I / A..D: all four corners are train.
    names=['G004_B005','I004_B005','I004_D005','G004_D005'] if size==3 else ['F004_A005','I004_A005','I004_D005','F004_D005']
    by_name={r['physical_camera'][:9]:r for r in rows}
    anchors=[by_name[n] for n in names]
    if {r['physical_camera'] for r in anchors}&HELD_CAMERAS:raise ValueError('Held-out anchor')
    corners=np.array([r['transform_matrix'] for r in anchors])[:,:3,3]
    # Include the closing segment in arc-length integration; video itself has
    # no duplicate endpoint. Otherwise one seam step is 2-3% longer.
    ww,uv=spline_weights(24001,endpoint=True);dense=ww@corners
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
    samples=np.interp(np.arange(count)*distance[-1]/count,distance,np.arange(len(distance)))
    q=np.column_stack([np.interp(samples,np.arange(len(uv)),uv[:,j]) for j in range(2)])
    u,v=q.T;ww=np.column_stack(((1-u)*(1-v),u*(1-v),u*v,(1-u)*v));positions=ww@corners
    extent=np.ptp(q,axis=0)*(size-1)
    if (extent<.95*(size-1)).any():raise ValueError('Insufficient actual grid travel')
    ref=next(r for r in rows if r['physical_camera']=='H004_C005_1210SZ');pose=np.array(ref['transform_matrix'])
    focus=(pose[:3,3]-center)@pose[:3,2];target=pose[:3,3]-pose[:3,2]*focus;up=pose[:3,1]
    frames=[]
    for i,pos in enumerate(positions):
        z=pos-target;z/=np.linalg.norm(z);x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x)
        p=np.eye(4);p[:3,:3]=np.column_stack((x,y,z));p[:3,3]=pos
        row=deepcopy(ref);row.pop('file_path',None);row['transform_matrix']=p.tolist()
        row['physical_camera']=f'grid_{size}_{i:05d}';row['convex_weights']=ww[i].tolist();frames.append(row)
    mats=np.array([r['transform_matrix'] for r in frames]);nxt=np.roll(mats,-1,axis=0)
    speed=np.linalg.norm(nxt[:,:3,3]-mats[:,:3,3],axis=1)*24
    angular=Rotation.from_matrix(mats[:,:3,:3].transpose(0,2,1)@nxt[:,:3,:3]).magnitude()*24*180/np.pi
    if speed.max()/speed.min()>1.002 or angular.max()>3.5:raise ValueError('Nonuniform or fast grid flight')
    return frames,{'grid_size':size,'anchors':[r['physical_camera'] for r in anchors],
        'achieved_grid_interval_extent_xy':extent.tolist(),'minimum_convex_weight':float(ww.min()),
        'angular_speed_deg_s_min_median_max':np.quantile(angular,[0,.5,1]).tolist(),
        'linear_speed_ratio':float(speed.max()/speed.min()),'fps':24,'duration_seconds':count/24}


def run(output,workers=4):
    output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    atlas=dict(np.load(BASE/'atlas_geometry.npz'));texture=np.array(Image.open(BASE/'texture_joint.png').convert('RGB'))
    rows,_,meta=cameras('000973');cal=read(CALIBRATION);old=read(OLD/'request.json')
    oldrows=[normalize_frame(calibration_pose(r['camera'],cal,read(r['metadata'])),cal,read(meta)) for r in old['inventory']]
    w=np.array([r['convex_weights'] for r in oldrows]);grid=w@np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    atomic_json(output/'diagnosis.json',{'old_request_sha256':sha(OLD/'request.json'),
        'old_exact_anchor_grid_extent_xy':np.ptp(grid,axis=0).tolist(),
        'cause':'Contracted four-anchor curve and only 150/360 loop samples; no minimum grid-extent acceptance gate. Fixed look-at and moving actor obscure the already-small travel.',
        'renderer_uses_requested_pose':True,'static_source_frame':'000973',
        'atlas_note':'Existing static hard-texture-v2 atlas, identical in every pilot; not a new temporal reconstruction.',
        'atlas_sha256':sha(BASE/'atlas_geometry.npz'),'texture_sha256':sha(BASE/'texture_joint.png'),
        'script_sha256':sha(__file__)})
    variants={'old':(oldrows,{'fps':30,'duration_seconds':5,'achieved_grid_interval_extent_xy':np.ptp(grid,axis=0).tolist()})}
    for size,count in [(3,480),(4,720)]:variants[f'{size}x{size}']=grid_path(rows,atlas['vertices'].mean(0),size,count)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref=np.array(next(r for r in rows if r['physical_camera']=='H004_C005_1210SZ')['transform_matrix'])
    basis=ref[:3,[1,0]];origin=ref[:3,3]
    fig,ax=plt.subplots(figsize=(9,8))
    for row in rows:
        name=row['physical_camera'];point=(np.array(row['transform_matrix'])[:3,3]-origin)@basis
        ax.scatter(*point,s=12,c='gray')
        if name[0] in 'FGHIJ':ax.annotate(name[:9],point,fontsize=7)
    for name,(path,_) in variants.items():
        xy=(np.array([r['transform_matrix'] for r in path])[:,:3,3]-origin)@basis
        ax.plot(*xy.T,label=name,lw=2)
    ax.set_aspect('equal');ax.grid(alpha=.3);ax.legend();ax.set_title('Actual camera-center travel; static actor')
    ax.set_xlabel('reference portrait horizontal, normalized units');ax.set_ylabel('reference portrait vertical, normalized units')
    fig.tight_layout();fig.savefig(output/'camera_travel.png',dpi=130);plt.close(fig)
    scene=scene_for(atlas['vertices'],atlas['triangles'])
    for name,(path,report) in variants.items():
        root=output/name;root.mkdir(exist_ok=True);sequence=root/'frames';sequence.mkdir(exist_ok=True)
        request={'path':path,'report':report,'diagnosis_sha256':sha(output/'diagnosis.json')}
        if (root/'request.json').exists() and read(root/'request.json')!=request:raise ValueError('Immutable pilot mismatch')
        atomic_json(root/'request.json',request)
        def frame(item):
            i,row=item;row=deepcopy(row)
            for key in ['fl_x','fl_y','cx','cy']:row[key]*=.5
            row['w'],row['h']=960,540
            rgb,depth,_,_=render_atlas(scene,atlas,texture,row)
            if np.isfinite(depth).mean()<.01:raise ValueError('Empty pilot framing')
            destination=sequence/f'{i:05d}.png';Image.fromarray(np.rot90(rgb)).save(destination,compress_level=1)
            return {'index':i,'sha256':sha(destination)}
        results=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(frame,enumerate(path)):
                results.append(result)
                if len(results)%60==0:
                    atomic_json(output/'progress.json',{'variant':name,'completed':len(results),'total':len(path),'pid':os.getpid(),'elapsed_seconds':time.monotonic()-start})
                    print(f'{name} {len(results)}/{len(path)}',flush=True)
        atomic_json(root/'frame_hashes.json',results)
        panel=Image.new('RGB',(540*4,960*2+48));draw=ImageDraw.Draw(panel)
        for j,i in enumerate(np.linspace(0,len(path)-1,8,dtype=int)):
            x,y=j%4*540,j//4*984;panel.paste(Image.open(sequence/f'{i:05d}.png'),(x,y+24));draw.text((x+5,y+4),f'{name} {i/report["fps"]:.2f}s',fill='white')
        panel.save(root/'contact.png')
        close=Image.new('RGB',(760,888));draw=ImageDraw.Draw(close)
        for j,i in enumerate([0,len(path)//4,len(path)//2,3*len(path)//4]):
            x,y=j%2*380,j//2*444;close.paste(Image.open(sequence/f'{i:05d}.png').crop((120,240,500,660)),(x,y+24))
            draw.text((x+4,y+4),f'{name} {i/report["fps"]:.2f}s',fill='white')
        close.save(root/'camera_only_detail.png')
        env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0')
        subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-framerate',str(report['fps']),'-i',str(sequence/'%05d.png'),'-c:v','libx264','-crf','18','-preset','fast','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(root/'video.mp4')],env=env,check=True)
        atomic_json(root/'result.json',{'report':report,'source_time_count':1,'camera_only_control':True,'video_sha256':sha(root/'video.mp4'),'visual_status':'pending'})
    atomic_json(output/'progress.json',{'status':'finished','elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.output,a.workers)
