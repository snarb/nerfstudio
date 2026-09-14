"""Bounded local 3D membrane proposals at head silhouette notches.

Fit inverse depth to the observed boundary, lift it into 3D, require train
foreground support, and keep all existing mesh triangles. This is a disclosed
surface prior, not independently measured depth or RGB inpainting.
"""
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage
from joint_temporal_texture import read,sha,atomic_json,cameras,project
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth

SETTINGS=dict(radius_pixels=24,max_area=12000,boundary_radius=5,max_absolute_depth_rmse=.004,min_head_x=-.03,min_train_silhouettes=2)

def complete(record,output):
    root=output/record['frame_id'];root.mkdir(parents=True,exist_ok=True)
    request=dict(mesh_sha256=record['mesh_sha256'],camera=record['camera'],source_masks=record['source_masks'],settings=SETTINGS,script_sha256=sha(__file__))
    if (root/'complete.json').exists():
        done=read(root/'complete.json')
        if done['request']!=request or sha(root/'mesh.ply')!=done['mesh_sha256']:raise ValueError('Notch resume mismatch')
        return done
    if sha(record['mesh'])!=record['mesh_sha256']:raise ValueError('Changed input geometry')
    m=o3d.io.read_triangle_mesh(record['mesh']);v=np.asarray(m.vertices);t=np.asarray(m.triangles);row=record['camera']
    depth,_,_=camera_depth(scene_for(v,t),row);valid=np.isfinite(depth)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(49,49))
    closed=cv2.morphologyEx(valid.astype(np.uint8),cv2.MORPH_CLOSE,kernel)>0
    proposals=(closed|ndimage.binary_fill_holes(valid))&~valid
    labels,count=ndimage.label(proposals);filled=np.zeros_like(valid);fit=depth.copy();operations=[]
    yy,xx=np.mgrid[:row['h'],:row['w']];pose=np.array(row['transform_matrix'])
    def points_at(xs,ys,z):
        q=np.column_stack(((xs-row['cx'])/row['fl_x']*z,-(ys-row['cy'])/row['fl_y']*z,-z))
        return q@pose[:3,:3].T+pose[:3,3]
    maskspec=record['source_masks'];maskroot=Path(maskspec['root'])
    if sha(maskroot/'masks.npz')!=maskspec['masks_sha256'] or sha(maskroot/'cameras.json')!=maskspec['cameras_sha256']:raise ValueError('Changed train silhouette evidence')
    masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json');rows,_,_=cameras(record['frame_id']);lookup={r['physical_camera']:r for r in rows};rows=[lookup[n] for n in names]
    for label in range(1,count+1):
        hole=labels==label;area=int(hole.sum())
        if area>SETTINGS['max_area'] or area<3:continue
        ring=ndimage.binary_dilation(hole,iterations=SETTINGS['boundary_radius'])&valid
        y,x=np.nonzero(ring)
        if len(x)<12:continue
        z=depth[ring];center=np.array([x.mean(),y.mean()]);a=np.column_stack((x-center[0],y-center[1],np.ones(len(x))))
        coeff=np.linalg.lstsq(a,1/z,rcond=None)[0];pred=1/(a@coeff);rmse=float(np.sqrt(np.mean((pred-z)**2)))
        if rmse>SETTINGS['max_absolute_depth_rmse']:continue
        hy,hx=np.nonzero(hole);hz=1/(np.column_stack((hx-center[0],hy-center[1],np.ones(len(hx))))@coeff)
        points=points_at(hx,hy,hz)
        eligible=np.isfinite(hz)&(hz>0)&(points[:,0]>SETTINGS['min_head_x'])
        if not eligible.any():continue
        uv,zq=project(points,rows);support=np.zeros(len(points),np.uint8)
        for i,mask in enumerate(masks):
            px=np.rint(uv[i]).astype(int);inside=(zq[i]>0)&(px[:,0]>=0)&(px[:,0]<1920)&(px[:,1]>=0)&(px[:,1]<1080)
            support[inside]+=mask[px[inside,1],px[inside,0]]>0
        eligible&=support>=SETTINGS['min_train_silhouettes'];filled[hy[eligible],hx[eligible]]=True;fit[hy[eligible],hx[eligible]]=hz[eligible]
        operations.append(dict(area=area,accepted_pixels=int(eligible.sum()),depth_rmse=rmse))
    domain=ndimage.binary_dilation(filled)&(valid|filled);py,px=np.nonzero(domain)
    points=points_at(px,py,fit[domain]+.00002);index=np.full(valid.shape,-1,int);index[domain]=np.arange(len(points))+len(v)
    aa=index[:-1,:-1];bb=index[:-1,1:];cc=index[1:,:-1];dd=index[1:,1:]
    triangles=[]
    for a,b,c,has in [(aa,bb,cc,filled[:-1,:-1]|filled[:-1,1:]|filled[1:,:-1]),(bb,dd,cc,filled[:-1,1:]|filled[1:,1:]|filled[1:,:-1])]:
        good=(a>=0)&(b>=0)&(c>=0)&has;triangles.append(np.column_stack((a[good],b[good],c[good])))
    added=np.concatenate(triangles);vv=np.concatenate((v,points));tt=np.concatenate((t,added))
    if len(added):
        safe=(vv[added,:,][...,0]>.03*-1).all(1)&(np.ptp(vv[added],axis=1).max(1)<.01)
        added=added[safe];tt=np.concatenate((t,added))
    result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt));result.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(root/'mesh.ply'),result)
    np.savez_compressed(root/'evidence.npz',original_depth=np.where(valid,depth,0),filled=filled,proposed_depth=np.where(filled,fit,0))
    atomic_json(root/'operations.json',dict(components=operations,added_triangles=len(added),all_original_triangles_unchanged=True,
        camera_conditioned_3d_completion=True,train_silhouettes_are_not_depth_measurements=True,uses_target_rgb=False,uses_synthetic_rgb=False))
    done=dict(request=request,mesh_sha256=sha(root/'mesh.ply'),operations_sha256=sha(root/'operations.json'),evidence_sha256=sha(root/'evidence.npz'),added_triangles=len(added))
    atomic_json(root/'complete.json',done);print(f'notches={record["frame_id"]} pixels={filled.sum()} added={len(added)}',flush=True);return done
