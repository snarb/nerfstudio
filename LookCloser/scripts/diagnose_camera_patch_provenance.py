"""Show original, boundary-prior, and view-conditioned triangles in a shot."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from bake_joint_temporal_mesh import camera_depth
from diffusion_mesh_repair import scene_for

def diagnose(output,frame):
    row=next(r for r in read(output/'request.json')['inventory'] if r['frame_id']==frame)
    head=read(row['head_repair_receipt']);source=o3d.io.read_triangle_mesh(head['request']['source_mesh'])
    boundary=o3d.io.read_triangle_mesh(row['pre_notch_mesh']);mesh=o3d.io.read_triangle_mesh(row['mesh'])
    _,ids,_=camera_depth(scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles)),row['camera'])
    labels=np.zeros(ids.shape,np.uint8);valid=ids!=np.iinfo(np.uint32).max
    labels[valid]=1;labels[valid&(ids>=len(source.triangles))]=2;labels[valid&(ids>=len(boundary.triangles))]=3
    labels=np.rot90(labels);rgb=np.asarray(Image.open(output/'frames'/frame/'frame.png'))
    colors=np.array([[0,0,0],[0,160,0],[0,140,255],[255,40,0]],np.uint8)
    overlay=rgb.copy();selected=labels>=2
    overlay[selected]=(rgb[selected]*.25+colors[labels[selected]]*.75).astype(np.uint8)
    panel=Image.new('RGB',(2160,1950));panel.paste(Image.fromarray(rgb),(0,30));panel.paste(Image.fromarray(overlay),(1080,30))
    ImageDraw.Draw(panel).text((10,5),'Original RGB | BLUE=boundary completion RED=view-conditioned completion',fill='white')
    root=output/'patch_provenance';root.mkdir(exist_ok=True);panel.save(root/f'{frame}.png')
    np.savez_compressed(root/f'{frame}.npz',labels=labels)
    atomic_json(root/f'{frame}.json',dict(frame=frame,mesh_sha256=sha(row['mesh']),camera=row['camera'],
        original_triangles=len(source.triangles),boundary_triangles=len(boundary.triangles)-len(source.triangles),
        camera_conditioned_triangles=len(mesh.triangles)-len(boundary.triangles),
        pixels={str(i):int((labels==i).sum()) for i in range(4)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--frame',required=True)
    a=p.parse_args();diagnose(a.output,a.frame)
