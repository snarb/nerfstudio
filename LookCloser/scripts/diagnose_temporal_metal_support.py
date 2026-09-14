"""Train-only material/visibility diagnostic for thin metallic-object fringes.

A neutral-metal color test is a disclosed material prior, not a segmentation or
independent depth observation. This diagnostic does not modify production meshes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras,ROOT,exr,display,project
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth


def diagnose(output,frame):
    root=output/frame;root.mkdir(parents=True,exist_ok=True)
    rows,mesh_path,_=cameras(frame)
    mesh=o3d.io.read_triangle_mesh(str(mesh_path));v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    points=v[t].mean(1);scene=scene_for(v,t);uv,z=project(points,rows)
    response=read(ROOT/'camera_profiles.json');gains=dict(zip(response['physical_cameras'],response['rgb_gain']))
    exposure=read(ROOT/'exposure.json')['fixed_exposure_gain']
    colors=[];visibility=[]
    for i,row in enumerate(rows):
        rgb=display(exr(row['file_path'])*np.array(gains[row['physical_camera']]),exposure).astype(np.float32)
        depth,_,_=camera_depth(scene,row);depth=np.where(np.isfinite(depth),depth,0)
        samples=[];sampled_depth=[]
        for start in range(0,len(points),16000):
            q=uv[i,start:start+16000].reshape(1,-1,2)
            samples.append(cv2.remap(rgb,q,None,cv2.INTER_LINEAR)[0]);sampled_depth.append(cv2.remap(depth,q,None,cv2.INTER_LINEAR)[0])
        color=np.concatenate(samples);d=np.concatenate(sampled_depth)
        visible=(z[i]>0)&(d>0)&(np.abs(d-z[i])<.0015*z[i])
        colors.append(color);visibility.append(visible)
    color=np.stack(colors);visible=np.stack(visibility);maximum=color.max(2)
    neutral=(maximum>.28)&(np.ptp(color,axis=2)<.18*maximum)&visible
    count=neutral.sum(0);total=visible.sum(0)
    selected=(count>=4)&(count>=.3*total)
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[selected]));labels=np.asarray(cloud.cluster_dbscan(eps=.0012,min_points=6))
    original=np.flatnonzero(selected);clusters=[]
    for label in sorted(set(labels)-{-1}):
        faces=original[labels==label];p=points[faces];center=np.median(p,axis=0)
        _,s,vt=np.linalg.svd(p-center,full_matrices=False);axis=vt[0]
        delta=p-center;along=delta@axis;radial=np.linalg.norm(delta-along[:,None]*axis,axis=1)
        clusters.append(dict(cluster=int(label),faces=faces.tolist(),center=center.tolist(),axis=axis.tolist(),
            length=float(np.ptp(along)),radial_median_p90=np.quantile(radial,[.5,.9]).tolist(),singular_values=s.tolist(),
            neutral_view_count_median=float(np.median(count[faces]))))
    np.savez_compressed(root/'evidence.npz',neutral_count=count,visible_count=total,selected=selected,cluster_labels=labels)
    atomic_json(root/'clusters.json',dict(frame_id=frame,mesh_sha256=sha(mesh_path),script_sha256=sha(__file__),
        material_prior=dict(minimum_max_channel=.28,maximum_relative_channel_range=.18,minimum_neutral_views=4,minimum_neutral_fraction=.3),
        candidates=int(selected.sum()),clusters=clusters,source_rgb_hashes={r['physical_camera']:sha(r['file_path']) for r in rows},
        not_independent_depth=True,production_mesh_modified=False))
    ref=next(r for r in rows if r['physical_camera']=='H004_C005_1210SZ');rgb=np.rint(display(exr(ref['file_path'])*np.array(gains[ref['physical_camera']]),exposure)*255).clip(0,255).astype(np.uint8)
    panel=Image.fromarray(rgb);draw=ImageDraw.Draw(panel);palette=['red','lime','cyan','magenta','yellow']
    for i,cluster in enumerate(clusters):
        q,_=project(points[cluster['faces']],[ref]);box=[*q[0].min(0),*q[0].max(0)]
        draw.rectangle(box,outline=palette[i%len(palette)],width=2);draw.text((box[0],box[1]-12),str(i),fill=palette[i%len(palette)])
    panel.transpose(Image.Transpose.ROTATE_90).save(root/'cluster_review.png')
    print(frame,[(c['cluster'],len(c['faces']),round(c['length'],5),c['radial_median_p90']) for c in clusters],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_dynamic_metal_diagnosis'))
    p.add_argument('--frames',nargs='+',default=['000973']);a=p.parse_args();cv2.setNumThreads(2)
    for frame in a.frames:diagnose(a.output,frame)
