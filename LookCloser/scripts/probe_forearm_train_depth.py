"""Real-train evidence for temporal forearm holes; no evaluation RGB input."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import cameras,read,sha,atomic_json,exr,display,ROOT
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth


def references(frame, output):
    parent=Path('/mnt/data/dec5_elevated_camera_dynamic_150')
    row=next(r for r in read(parent/'request.json')['inventory'] if r['frame_id']==frame)
    rows,_,_=cameras(frame)
    centers=np.array([r['transform_matrix'] for r in rows])[:,:3,3]
    target=np.array(row['camera']['transform_matrix'])[:3,3]
    selected=np.argsort(np.linalg.norm(centers-target,axis=1))[:3]
    profile=read(ROOT/'camera_profiles.json');gains=dict(zip(profile['physical_cameras'],profile['rgb_gain']))
    exposure=read(ROOT/'exposure.json')['fixed_exposure_gain']
    mesh=o3d.io.read_triangle_mesh(row['untrimmed_mesh'])
    if sha(row['untrimmed_mesh'])!=row['untrimmed_mesh_sha256']:raise ValueError('Changed original mesh')
    scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    root=output/frame;root.mkdir(parents=True,exist_ok=True)
    panel=Image.new('RGB',(1620,984));draw=ImageDraw.Draw(panel);evidence=[]
    for j,i in enumerate(selected):
        camera=rows[i];rgb=np.rint(display(exr(camera['file_path'])*gains[camera['physical_camera']],exposure)*255).clip(0,255).astype(np.uint8)
        portrait=np.rot90(rgb);path=root/f'train_{j}.png';Image.fromarray(portrait).save(path)
        depth,_,_=camera_depth(scene,camera)
        np.savez_compressed(root/f'train_{j}_mesh_depth.npz',depth=np.where(np.isfinite(depth),depth,0))
        panel.paste(Image.fromarray(portrait).resize((540,960)),(j*540,24));draw.text((j*540+3,4),camera['physical_camera'],fill='white')
        evidence.append(dict(camera=camera,source_sha256=sha(camera['file_path']),image_sha256=sha(path)))
    panel.save(root/'train_references.png')
    atomic_json(root/'references.json',dict(frame=frame,mesh=row['untrimmed_mesh'],mesh_sha256=row['untrimmed_mesh_sha256'],
        metadata=row['metadata'],evidence=evidence,script_sha256=sha(__file__),heldout_rgb_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',nargs='+',default=['001033','001041'])
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_train_probe'))
    a=p.parse_args()
    for frame in a.frames:references(frame,a.output)
