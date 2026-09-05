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


def source_identity(prediction,warps,selection,colors):
    """Check selected RGB against individual source warps, allowing PNG rounding."""
    rows=[]
    for rank,(warp,color) in enumerate(zip(warps,colors)):
        chosen=(selection==color).all(-1)
        difference=np.abs(prediction.astype(np.int16)-warp.astype(np.int16))[chosen]
        rows.append({'rank':rank,'pixels':int(chosen.sum()),
                     'max_rgb_difference_8bit':int(difference.max()) if difference.size else None,
                     'nonidentical_pixels':int((difference!=0).any(-1).sum()),
                     'pixels_differing_over_one_lsb':int((difference>1).any(-1).sum())})
    return rows


def seam_depth_statistics(selection,depth,colors,box):
    """Check the primary/first-alternative seam; this does not validate true depth."""
    x0,y0,x1,y1=box
    selected=selection[y0:y1,x0:x1]
    primary=(selected==colors[0]).all(-1);secondary=(selected==colors[1]).all(-1)
    z=depth[y0:y1,x0:x1]
    jumps=[]
    for a,b in [((slice(None),slice(None,-1)),(slice(None),slice(1,None))),
                ((slice(None,-1),slice(None)),(slice(1,None),slice(None)))]:
        boundary=((primary[a]&secondary[b])|(secondary[a]&primary[b]))&(z[a]>0)&(z[b]>0)
        jumps.extend(np.abs(np.log(z[a][boundary]/z[b][boundary])).tolist())
    return {'adjacent_pixel_pairs':len(jumps),'median_abs_log_depth_jump':float(np.median(jumps)) if jumps else None,
            'p90_abs_log_depth_jump':float(np.percentile(jumps,90)) if jumps else None,
            'fraction_under_0_001':float(np.mean(np.array(jumps)<.001)) if jumps else None,
            'interpretation':'Smoothness of this mesh only; not independent ground-truth geometry.'}


def sample(array,u,v):
    return float(map_coordinates(array,[[v],[u]],order=1,mode='constant',cval=0)[0])


def bilinear_footprint(depth,u,v,projected_z,rgb):
    """Audit native taps without smoothing depth across an object boundary."""
    x,y=int(np.floor(u)),int(np.floor(v));fx,fy=u-x,v-y;h,w=depth.shape
    rows=[]
    for dy in (0,1):
        for dx in (0,1):
            xx,yy=x+dx,y+dy;inside=0<=xx<w and 0<=yy<h
            z=float(depth[yy,xx]) if inside else None
            finite=z is not None and np.isfinite(z) and z>0
            rows.append({'xy':[xx,yy],'weight':float((fx if dx else 1-fx)*(fy if dy else 1-fy)),
                         'mesh_depth':z if finite else None,
                         'same_depth_layer':bool(finite and projected_z>0 and abs(np.log(z/projected_z))<=.005),
                         'native_rgb8':list(rgb.getpixel((xx,yy))) if inside else None})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('normalized-data','render','mesh-depth-manifest','raw-depth-data','mesh-metadata','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pixel',type=int,nargs=2,action='append',required=True)
    p.add_argument('--variant',default='seam_cut8')
    p.add_argument('--audit-box',type=int,nargs=4,default=(400,390,800,780))
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
    if audit.get('surface_texture_registration'):
        raise ValueError('This calibrated-projection trace cannot follow adjusted texture UVs')
    offset=float(audit.get('pixel_center_offset',0))
    from mesh_texture_visibility import MeshVisibility
    mesh_path=Path(dm['mesh'])
    if sha256(mesh_path)!=dm['mesh_sha256']:raise ValueError('Raycast mesh hash mismatch')
    if dm.get('parameters',{}).get('mesh_coordinate_scale',1.)!=1.:
        raise ValueError('Point trace requires unscaled mesh coordinates')
    visibility=MeshVisibility(mesh_path)
    def fill(path):return fill_small_consistent_depth_holes(load_depth(path),max_area=1000,boundary_radius=4,max_relative_plane_rmse=.015)[0]
    target_depth=fill(depths[Path(target['file_path']).resolve()])
    source_arrays=[]
    for source in audit['sources']:
        image=Path(source['source_image']);f=frames[image]
        source_arrays.append((source,f,fill(depths[image]),load_depth(depths[image]),
            load_depth(a.raw_depth_data/raw_by_stem[image.stem]['depth_file_path'])*meta['dataparser_scale']))
    rows=[]
    for index,(x,y) in enumerate(a.pixel):
        z=float(target_depth[y,x]);q=np.array([(x+offset-target['cx'])/target['fl_x']*z,-(y+offset-target['cy'])/target['fl_y']*z,-z])
        world=target_pose[:3,:3]@q+target_pose[:3,3]
        point_distance=visibility.scene.compute_distance(visibility.o3d.core.Tensor(world[None].astype(np.float32))).numpy()[0]
        row={'pixel':[x,y],'target_depth':z,'world':world.tolist(),'distance_to_mesh_normalized':float(point_distance),'sources':[]}
        crops=[]
        for source,f,depth,native_depth,raw_depth in source_arrays:
            pose=np.asarray(f['transform_matrix']);q=(world-pose[:3,3])@pose[:3,:3];projected=-q[2]
            u=f['fl_x']*q[0]/projected+f['cx']-offset;v=-f['fl_y']*q[1]/projected+f['cy']-offset
            observed=sample(depth,u,v);stereo=sample(raw_depth,u,v)
            rank=source['rank'];image=Path(source['source_image'])
            valid=bool(np.asarray(Image.open(a.render/'source_warps'/f'valid_{rank:02d}.png'))[y,x])
            exact,exact_stats=visibility.visible(world[None,None],pose[:3,3],np.ones((1,1),bool))
            im=Image.open(image).convert('RGB')
            row['sources'].append({'rank':rank,'physical_camera':f['physical_camera'],'uv':[float(u),float(v)],
                  'projected_z':float(projected),'mesh_z':observed,'raw_stereo_z':stereo,'renderer_valid':valid,
                  'exact_mesh_visible':bool(exact[0,0]),'exact_visibility_stats':exact_stats,
                  'native_bilinear_footprint':bilinear_footprint(native_depth,u,v,projected,im),
                  'mesh_log_error':float(abs(np.log(projected/observed))) if observed>0 else None})
            cx,cy=round(u),round(v)
            crop=im.crop((cx-100,cy-100,cx+100,cy+100)).rotate(90,expand=True)
            draw=ImageDraw.Draw(crop);draw.ellipse((96,96,104,104),outline='red',width=1)
            pane=Image.new('RGB',(200,226));pane.paste(crop,(0,26))
            ImageDraw.Draw(pane).text((2,3),f'{rank}: {f["physical_camera"][:9]} valid={valid}',fill='white');crops.append(pane)
        canvas=Image.new('RGB',(800,452))
        for j,crop in enumerate(crops):canvas.paste(crop,((j%4)*200,(j//4)*226))
        canvas.save(a.output/f'point_{index:02d}_train_patches.png');rows.append(row)
    # Categorical palette from write_source_selection, limited to the audited first eight ranks.
    colors=np.array([[230,25,75],[60,180,75],[255,225,25],[0,130,200],[245,130,48],[145,30,180],[70,240,240],[240,50,230]],np.uint8)
    if len(audit['sources'])>len(colors):raise ValueError('This trace supports at most eight source ranks')
    pred_path=a.render/a.variant/'eval_pred_0000.png';selection_path=a.render/a.variant/'source_selection.png'
    prediction=np.asarray(Image.open(pred_path).convert('RGB'))
    selection=np.asarray(Image.open(selection_path).convert('RGB'))
    warps=[np.asarray(Image.open(a.render/'source_warps'/f'source_{s["rank"]:02d}.png').convert('RGB')) for s in audit['sources']]
    identity=source_identity(prediction,warps,selection,colors)
    geometry=seam_depth_statistics(selection,target_depth,colors,a.audit_box)
    panes=[]
    for rank in range(min(3,len(warps))):
        valid=np.asarray(Image.open(a.render/'source_warps'/f'valid_{rank:02d}.png'))>0
        visible_only=Image.fromarray(np.where(valid[...,None],warps[rank],0)).crop(a.audit_box).rotate(90,expand=True)
        pane=Image.new('RGB',(visible_only.width,visible_only.height+28));pane.paste(visible_only,(0,28))
        ImageDraw.Draw(pane).text((5,6),f'Source {rank}; black = camera cannot see surface',fill='white');panes.append(pane)
    canvas=Image.new('RGB',(sum(p.width for p in panes),max(p.height for p in panes)))
    xpos=0
    for pane in panes:canvas.paste(pane,(xpos,0));xpos+=pane.width
    canvas.save(a.output/'visible_train_sources.png')
    hashed=[a.render/'reprojection_audit.json',pred_path,selection_path,a.normalized_data/'transforms.json',a.mesh_depth_manifest,mesh_path]
    hashed.extend(sorted((a.render/'source_warps').glob('*.png')))
    atomic_json(a.output/'trace.json',{'uses_eval_rgb':False,'render_audit_sha256':sha256(a.render/'reprojection_audit.json'),
               'script_sha256':sha256(Path(__file__)),'pixel_center_offset':offset,'pixels':rows,
               'source_identity':identity,'audit_box_xyxy':a.audit_box,'seam_depth_statistics':geometry,
               'input_hashes':{str(path):sha256(path) for path in hashed},
               'raw_depth_caveat':'Bilinear samples can mix invalid zeros or separate depth layers; not independent visibility evidence.'})
    print(json.dumps({'pixels':len(rows),'source_identity':identity,'seam_depth_statistics':geometry}))


if __name__=='__main__':main()
