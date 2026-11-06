"""Source overlays and orthographic reviews of a temporal mask-hull AABB.

These are photographs with geometry overlays, not newly trained NeRF renders.
Occlusion uses the retained hull and the fitted floor, both approximate proxies.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from aabb_visualization import box_edges, camera_in_box_frame, overlay_depth_aware


def voxel_scene(lower,upper):
    import open3d as o3d
    corners,edges=box_edges([[0,0,0],[1,1,1]])
    vertices=(lower[:,None]+corners[None]*(upper-lower)[:,None]).reshape(-1,3)
    faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],
                    [0,4,5],[0,5,1],[2,3,7],[2,7,6],
                    [0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int32)
    triangles=(faces[None]+8*np.arange(len(lower))[:,None,None]).reshape(-1,3)
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(vertices.astype('float32')),
                        o3d.core.Tensor(triangles.astype('uint32')))
    return scene


def depths(scene,camera):
    import open3d as o3d
    c=camera;pose=np.asarray(c['transform_matrix'])
    yy,xx=np.mgrid[:c['h'],:c['w']]
    d=np.stack([(xx+.5-c['cx'])/c['fl_x'],-(yy+.5-c['cy'])/c['fl_y'],-np.ones_like(xx)],-1)@pose[:3,:3].T
    d/=np.linalg.norm(d,axis=-1,keepdims=True)
    o=np.broadcast_to(pose[:3,3],d.shape)
    rays=np.concatenate([o,d],-1).astype('float32')
    hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)['t_hit'].numpy()
    with np.errstate(divide='ignore',invalid='ignore'):
        floor=-o[...,2]/d[...,2]
    floor=np.where(floor>0,floor,np.inf)
    return np.minimum(hit,floor)


def main(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta=json.loads((a.geometry/'common_box.json').read_text())
    bounds=np.asarray(meta['aabb']);T=np.array(meta['world_to_box'])
    corners,edges=box_edges(bounds)
    a.output.mkdir(parents=True,exist_ok=False)
    reports=[];orth=[]
    for f in meta['sample_frames']:
        folder=a.output/str(f);folder.mkdir()
        z=np.load(a.geometry/str(f)/'cells.npz');lo,hi=z['lower'],z['upper']
        center=(lo+hi)/2;orth.append(center)
        scene=voxel_scene(lo,hi)
        for name in a.cameras:
            row=json.loads((a.sources/str(f)/name/'receipt.json').read_text())
            c=camera_in_box_frame(row['camera_row'],T)
            im=Image.open(row['image']).convert('RGB')
            depth=depths(scene,c)
            overlay,receipt=overlay_depth_aware(im,bounds,c,depth,tolerance=.008,dash=16,gap=10)
            ImageDraw.Draw(overlay).text((12,12),f'{f} | +{(f-1660)/120:.1f}s | {name} | SAME FLOOR-ALIGNED BOX | source + hull/floor depth proxy',fill='white',stroke_width=1,stroke_fill='black')
            overlay.save(folder/(name+'_box.png'))
            reports.append(dict(frame=f,camera=name,**receipt))
        print('source overlays',f,flush=True)
    colors=['#2878b5','#5aaf45','#ee9b24','#bd42ad','#e54c43']
    fig,axes=plt.subplots(1,3,figsize=(16,6),constrained_layout=True)
    for ax,(i,j,title) in zip(axes,[(0,1,'Top / floor footprint'),(0,2,'Front / height above floor'),(1,2,'Side / height above floor')]):
        for k,points in enumerate(orth):
            p=points[::max(1,len(points)//3500)]
            ax.scatter(p[:,i],p[:,j],s=2,alpha=.25,color=colors[k],label=f'{k*.5:.1f}s')
        ax.plot([bounds[0,i],bounds[1,i],bounds[1,i],bounds[0,i],bounds[0,i]],
                [bounds[0,j],bounds[0,j],bounds[1,j],bounds[1,j],bounds[0,j]],'r--',lw=2)
        if j==2:ax.axhline(0,color='black',lw=1,label='fitted floor')
        ax.set_aspect('equal');ax.grid(alpha=.25);ax.set_title(title)
        ax.set_xlabel('XYZ'[i]+' (prepared units)');ax.set_ylabel('XYZ'[j]+' (prepared units)')
    axes[0].legend(markerscale=3)
    fig.suptitle('One box for 0-2 seconds | silhouettes reconstructed independently at five times | explicit margin')
    fig.savefig(a.output/'orthographic.png',dpi=160);plt.close(fig)
    # Three-dimensional individual poses, one common floor and common box.
    fig=plt.figure(figsize=(18,7),constrained_layout=True)
    for k,points in enumerate(orth):
        ax=fig.add_subplot(1,5,k+1,projection='3d')
        p=points[::max(1,len(points)//4500)]
        ax.scatter(p[:,0],p[:,1],p[:,2],s=2,c=colors[k],alpha=.3)
        for i,j in edges:
            q=corners[[i,j]];ax.plot(q[:,0],q[:,1],q[:,2],'r--',lw=1)
        xx,yy=np.meshgrid(bounds[:,0],bounds[:,1]);ax.plot_surface(xx,yy,np.zeros_like(xx),alpha=.13,color='gray')
        ax.set(xlim=bounds[:,0],ylim=bounds[:,1],zlim=bounds[:,2],title=f'+{k*.5:.1f}s')
        ax.set_box_aspect(bounds[1]-bounds[0]);ax.view_init(18,-60)
        ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('floor height')
    fig.savefig(a.output/'five_poses_3D.png',dpi=150);plt.close(fig)
    contact=Image.new('RGB',(480*len(a.contact_cameras),280*5),'#151515')
    for j,f in enumerate(meta['sample_frames']):
        for i,c in enumerate(a.contact_cameras):
            im=Image.open(a.output/str(f)/(c+'_box.png'));im.thumbnail((480,253))
            contact.paste(im,(i*480,j*280+24))
            ImageDraw.Draw(contact).text((i*480+5,j*280+5),f'+{j*.5:.1f}s | {c}',fill='white')
    contact.save(a.output/'all_five_times.jpg',quality=95)
    (a.output/'receipt.json').write_text(json.dumps(dict(kind='SOURCE_PHOTOS_WITH_PROXY_OCCLUDED_GEOMETRY',
        no_NeRF_training=True,depth_proxy='retained multi-view hull plus fitted floor plane; other scene geometry not modelled',
        hidden_geometry_certified=False,records=reports),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['sources','geometry','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cameras',nargs='+',default=['HC','DC','KA','HE','MD','CE','NA'])
    p.add_argument('--contact-cameras',nargs='+',default=['HC','HE','MD','KA'])
    main(p.parse_args())
