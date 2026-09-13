"""Read-only attribution of visible fringe to graph labels versus pixel fallback.

The reviewed rectangle is diagnostic only, never a segmentation/metric mask or
a permission to remove geometry. Original RGB remains untouched.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from bake_joint_temporal_mesh import camera_depth
from diffusion_mesh_repair import scene_for
from central_space_temporal_flythrough import OUTPUT


def diagnose(output,frame='001047'):
    request=read(output/'request.json');record=next(r for r in request['inventory'] if r['frame_id']==frame)
    directory=output/'frames'/frame;mesh=o3d.io.read_triangle_mesh(record['mesh'])
    v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    depth,ids,_=camera_depth(scene_for(v,t),record['camera'])
    labels=np.load(directory/'face_source_labels.npy');chosen=np.asarray(Image.open(directory/'source_ids.png'))
    valid=(ids<len(t))&(chosen<62);preferred=np.full(ids.shape,-1,np.int32);preferred[ids<len(t)]=labels[ids[ids<len(t)]]
    fallback=valid&(chosen!=preferred)
    rgb=np.array(Image.open(directory/'frame.png').convert('RGB'));fallback=np.rot90(fallback)
    box=(790,770,950,1100);x0,y0,x1,y1=box
    original=Image.fromarray(rgb).crop(box).resize((320,660),Image.Resampling.NEAREST)
    overlay=rgb.copy();overlay[fallback]=[255,0,255]
    marked=Image.fromarray(overlay).crop(box).resize((320,660),Image.Resampling.NEAREST)
    panel=Image.new('RGB',(640,688));panel.paste(original,(0,28));panel.paste(marked,(320,28));draw=ImageDraw.Draw(panel)
    draw.text((4,6),'Original native fringe ROI (2x)',fill='white');draw.text((324,6),'Magenta: fallback pixels only',fill='white')
    target=output/'fringe_diagnostic';target.mkdir(exist_ok=True);panel.save(target/f'{frame}_fallback.png')
    atomic_json(target/f'{frame}_fallback.json',{'frame':frame,'mesh_sha256':record['mesh_sha256'],'render_sha256':sha(directory/'frame.png'),
                'diagnostic_rectangle_portrait':box,'rectangle_pixels':(x1-x0)*(y1-y0),
                'rectangle_fallback_pixels':int(fallback[y0:y1,x0:x1].sum()),'frame_fallback_pixels':int(fallback.sum()),
                'recorded_frame_fallback_pixels':read(directory/'result.json')['fallback_pixels'],
                'does_not_claim_foreground_segmentation':True,'rgb_or_geometry_modified':False,'script_sha256':sha(__file__)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--frame',default='001047');a=p.parse_args();diagnose(a.output,a.frame)
