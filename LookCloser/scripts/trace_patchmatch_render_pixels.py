#!/usr/bin/env python3
"""Trace displayed pixels to fixed train RGB and mesh/raw stereo visibility."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import map_coordinates
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from render_mesh_image_blend import load_depth,fill_small_consistent_depth_holes


def sample(array,u,v):
    return float(map_coordinates(array,[[v],[u]],order=1,mode='constant',cval=0)[0])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('normalized-data','render','mesh-depth-manifest','raw-depth-data','mesh-metadata','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pixel',type=int,nargs=2,action='append',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    payload=json.loads((a.normalized_data/'transforms.json').read_text())
    frames={Path(f['file_path']).resolve():f for f in payload['frames']}
    target=next(f for f in payload['frames'] if f['file_path'] in payload['val_filenames'])
    target_pose=np.asarray(target['transform_matrix'])
    dm=json.loads(a.mesh_depth_manifest.read_text());depths={Path(r['image']).resolve():Path(r['depth']) for r in dm['images']}
    raw=json.loads((a.raw_depth_data/'transforms.json').read_text())
    raw_by_stem={Path(f['file_path']).stem:f for f in raw['frames'] if f.get('depth_file_path')}
    meta=json.loads(a.mesh_metadata.read_text())
    audit=json.loads((a.render/'reprojection_audit.json').read_text())
    def fill(path):return fill_small_consistent_depth_holes(load_depth(path),max_area=1000,boundary_radius=4,max_relative_plane_rmse=.015)[0]
    target_depth=fill(depths[Path(target['file_path']).resolve()])
    source_arrays=[]
    for source in audit['sources']:
        image=Path(source['source_image']);f=frames[image]
        source_arrays.append((source,f,fill(depths[image]),
            load_depth(a.raw_depth_data/raw_by_stem[image.stem]['depth_file_path'])*meta['dataparser_scale']))
    rows=[]
    for index,(x,y) in enumerate(a.pixel):
        z=float(target_depth[y,x]);q=np.array([(x-target['cx'])/target['fl_x']*z,-(y-target['cy'])/target['fl_y']*z,-z])
        world=target_pose[:3,:3]@q+target_pose[:3,3]
        row={'pixel':[x,y],'target_depth':z,'world':world.tolist(),'sources':[]}
        crops=[]
        for source,f,depth,raw_depth in source_arrays:
            pose=np.asarray(f['transform_matrix']);q=(world-pose[:3,3])@pose[:3,:3];projected=-q[2]
            u=f['fl_x']*q[0]/projected+f['cx'];v=-f['fl_y']*q[1]/projected+f['cy']
            observed=sample(depth,u,v);stereo=sample(raw_depth,u,v)
            rank=source['rank'];image=Path(source['source_image'])
            valid=bool(np.asarray(Image.open(a.render/'source_warps'/f'valid_{rank:02d}.png'))[y,x])
            row['sources'].append({'rank':rank,'physical_camera':f['physical_camera'],'uv':[float(u),float(v)],
                  'projected_z':float(projected),'mesh_z':observed,'raw_stereo_z':stereo,'renderer_valid':valid,
                  'mesh_log_error':float(abs(np.log(projected/observed))) if observed>0 else None})
            im=Image.open(image).convert('RGB');cx,cy=round(u),round(v)
            crop=im.crop((cx-100,cy-100,cx+100,cy+100)).rotate(90,expand=True)
            draw=ImageDraw.Draw(crop);draw.ellipse((96,96,104,104),outline='red',width=1)
            pane=Image.new('RGB',(200,226));pane.paste(crop,(0,26))
            ImageDraw.Draw(pane).text((2,3),f'{rank}: {f["physical_camera"][:9]} valid={valid}',fill='white');crops.append(pane)
        canvas=Image.new('RGB',(800,452))
        for j,crop in enumerate(crops):canvas.paste(crop,((j%4)*200,(j//4)*226))
        canvas.save(a.output/f'point_{index:02d}_train_patches.png');rows.append(row)
    atomic_json(a.output/'trace.json',{'uses_eval_rgb':False,'render_audit_sha256':sha256(a.render/'reprojection_audit.json'),
               'script_sha256':sha256(Path(__file__)),'pixels':rows})
    print(json.dumps(rows))


if __name__=='__main__':main()
