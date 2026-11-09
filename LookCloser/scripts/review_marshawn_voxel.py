#!/usr/bin/env python3
"""Calibrated surface-witness and voxel review for a prepared-world grid.

SIFT witnesses provide sparse geometric evidence, not watertight surface truth.
Camera crops establish frustum overlap, not visibility through other surfaces.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from marshawn_occupancy_audit import load_grid, dilate_one, cell_record
from audit_marshawn_geometry import project_points


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def classify(grid,boxes,points):
 if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Finite N×3 witness positions required')
 shape=np.asarray(grid.shape[1:]);levels=[];admitted=np.zeros(len(points),bool);containing=np.zeros(len(points),bool)
 for level,box in enumerate(boxes):
  rel=(points-box[:3])/(box[3:]-box[:3]);inside=((rel>=0)&(rel<1)).all(1)
  ids=np.flatnonzero(inside);indices=np.floor(rel[inside]*shape).astype(int)
  occupied=grid[(level,*indices.T)] if len(indices) else np.zeros(0,bool)
  admitted[ids]|=occupied;containing|=inside
  evidence=np.zeros(grid.shape[1:],np.uint8)
  if len(indices):evidence[tuple(indices.T)]=np.where(occupied,1,2)
  misses=ids[~occupied]
  levels.append(dict(level=level,inside=inside,ids=ids,indices=indices,occupied=occupied,evidence=evidence,misses=misses))
 return levels,np.flatnonzero(containing&~admitted),np.flatnonzero(~containing)

def witness_atlas(grid,levels,out):
 pages=[];scale=2
 for record in levels:
  level=record['level'];evidence=record['evidence']
  for axis in range(3):
   other=[i for i in range(3) if i!=axis];w,h=[grid.shape[i+1]*scale for i in other]
   for start in range(0,grid.shape[axis+1],16):
    sheet=Image.new('RGB',(4*(w+8),4*(h+25)),'#242424');draw=ImageDraw.Draw(sheet)
    for local,k in enumerate(range(start,min(start+16,grid.shape[axis+1]))):
     value=np.take(grid[level],k,axis=axis).T;ev=np.take(evidence,k,axis=axis).T
     rgb=np.repeat((value.astype(np.uint8)*160)[...,None],3,axis=2);rgb[ev==1]=[0,255,80];rgb[ev==2]=[255,50,50]
     im=Image.fromarray(rgb).resize((w,h),Image.Resampling.NEAREST);x=local%4*(w+8);y=local//4*(h+25);sheet.paste(im,(x,y+20));draw.text((x+2,y+2),f'{"XYZ"[axis]}={k}',fill='white')
    name=f'witness_l{level}_{"xyz"[axis]}_{start:04d}.png';sheet.save(out/name);pages.append(name)
 return pages

def corners(bounds):
 lo=np.asarray(bounds[:3]);hi=np.asarray(bounds[3:]);return np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])

def camera_crop(row,cell,data,out):
 points=corners(cell['bounds']);uv,front=project_points(row,points)
 if not front.any():return dict(camera=row['physical_camera'],status='behind_camera')
 if not front.all():return dict(camera=row['physical_camera'],status='crosses_camera_plane_requires_ray_review')
 lo=np.floor(uv.min(0)-.5).astype(int);hi=np.ceil(uv.max(0)-.5).astype(int)
 w,h=row['w'],row['h']
 if hi[0]<0 or hi[1]<0 or lo[0]>=w or lo[1]>=h:return dict(camera=row['physical_camera'],status='outside_raster')
 lo=np.maximum(lo-12,0);hi=np.minimum(hi+13,[w,h]);region=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]
 im=Image.open(data/row['file_path']).convert('RGB').crop(region);draw=ImageDraw.Draw(im);p=uv-.5-lo
 for k in range(8):
  for bit in [1,2,4]:
   n=k^bit
   if n>k:draw.line([tuple(p[k]),tuple(p[n])],fill='red',width=2)
 im.thumbnail((640,640));name=row['physical_camera']+'.jpg';im.save(out/name)
 return dict(camera=row['physical_camera'],status='in_frustum_occlusion_unknown',crop=name,source_rectangle=region,projected_corners_pixel_centers=uv.tolist())

def run(args):
 meta=json.loads((args.data/'transforms.json').read_text());grid,boxes=load_grid(args.grid)
 if args.dilation_radius:grid=dilate_one(grid)
 with np.load(args.witnesses,allow_pickle=False) as z:points=z['xyz'].copy();camera_ids=z['cameras'].tolist()
 audit_path=args.witnesses.parent/'audit.json'
 if audit_path.exists():
  audit=json.loads(audit_path.read_text())
  if audit.get('transforms_sha256')!=sha(args.data/'transforms.json'):raise ValueError('Witnesses refer to different prepared cameras/coordinates')
 levels,misses,outside=classify(grid,boxes,points)
 args.output.mkdir(parents=True,exist_ok=False)
 np.savez_compressed(args.output/'witness_cells.npz',xyz=points,misses_all_containing_cascades=misses,outside_all_cascades=outside,**{f'level{r["level"]}_misses':r['misses'] for r in levels})
 pages=witness_atlas(grid,levels,args.output);rows=[]
 for r in levels:
  # One row per unique surface-witness cell; preserve every point ID in NPZ.
  inside_ids=r['ids'];ix=r['indices'];unique,counts=np.unique(ix,axis=0,return_counts=True)
  false_empty=unique[~grid[(r['level'],*unique.T)]] if len(unique) else np.empty((0,3),int)
  records=[]
  for index in false_empty:
   ids=inside_ids[(ix==index).all(1)];center=np.mean(points[ids],axis=0);projections=[]
   for row in meta['frames']:
    uv,front=project_points(row,center[None]);u,v=uv[0]
    if front[0] and 0<=u<row['w'] and 0<=v<row['h']:
     projections.append(dict(camera=row['physical_camera'],pixel_center=[float(u),float(v)],occlusion='unknown'))
   records.append(dict(index_xyz=index.tolist(),witness_ids=ids.tolist(),witness_mean_xyz=center.tolist(),in_frustum_projections=projections))
  rows.append(dict(level=r['level'],inside_witnesses=len(inside_ids),outside_witnesses=int((~r['inside']).sum()),surface_witness_cells=len(unique),skipped_witnesses=len(r['misses']),skipped_witness_cells=len(false_empty),false_empty_candidates=records))
 query_rows=[]
 for query in args.cell or []:
  level,*index=query;cell=cell_record(grid,boxes,level,index);folder=args.output/f'cell_{level}_{index[0]}_{index[1]}_{index[2]}';folder.mkdir();crops=[camera_crop(row,cell,args.data,folder) for row in meta['frames']]
  (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Calibrated cell projections</h1><p>Frustum overlap only; occlusion unknown.</p>'+''.join(f'<h2>{html.escape(c["camera"])}</h2><p>{c["status"]}</p>'+ (f'<img src="{c["crop"]}">' if 'crop' in c else '') for c in crops))
  query_rows.append(dict(query=query,cell=cell,cameras=crops,path=str(folder.name)+'/index.html'))
 summary=dict(grid=str(args.grid),grid_sha256=sha(args.grid),transforms_sha256=sha(args.data/'transforms.json'),witnesses_sha256=sha(args.witnesses),script_sha256=sha(__file__),coordinate_contract='Grid and witnesses must both be in unchanged prepared_world coordinates; no camera renormalization is applied here.',dilation_radius=args.dilation_radius,shape=list(grid.shape),aabbs=boxes.tolist(),witnesses=len(points),train_cameras=camera_ids,misses_all_containing_cascades=len(misses),outside_all_cascades=len(outside),levels=rows,queries=query_rows,pages=pages,limitations=['Sparse triangulated witnesses are evidence, not full scene ground truth.','Outside-box witnesses can intentionally belong to excluded scene geometry.','Individual-cascade misses can be admitted by another cascade; combined-cascade count still does not emulate runtime cascade selection.','No false-occupied assertion is derived from density or missing witnesses.','In-frustum projections do not prove line-of-sight visibility.'])
 (args.output/'review.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Surface witnesses and occupancy</h1><p>Grey: occupied. Black: skipped. Green: witness admitted. Red: witness skipped. No witness means unknown, not empty.</p><a href="review.json">Complete report and projections</a>'+''.join(f'<p><a href="{q["path"]}">Cell {q["query"]}</a></p>' for q in query_rows)+''.join(f'<a href="{p}"><img loading="lazy" style="max-width:100%" src="{p}"></a>' for p in pages))
 print(json.dumps({k:summary[k] for k in ['witnesses','misses_all_containing_cascades','outside_all_cascades','shape']}))

def self_test():
 g=np.zeros((2,2,2,2),bool);g[1,0,0,0]=True;b=np.array([[0,0,0,1,1,1],[0,0,0,2,2,2]],float)
 p=np.array([[.25,.25,.25],[.75,.75,.75],[1.5,1.5,1.5],[2.,0,0]])
 levels,miss,outside=classify(g,b,p)
 assert levels[0]['misses'].tolist()==[0,1] and levels[1]['misses'].tolist()==[2]
 assert miss.tolist()==[2] and outside.tolist()==[3]
 row=dict(transform_matrix=np.eye(4).tolist(),fl_x=100.,fl_y=100.,cx=50.,cy=50.)
 uv,front=project_points(row,np.array([[0,0,-2],[1,0,-2]]))
 assert front.all() and np.allclose(uv,[[50,50],[100,50]])
 assert corners([0,0,0,1,2,3]).shape==(8,3)
 print('Synthetic cascade admission, half-open bounds, camera projection and voxel corners passed')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grid',type=Path);p.add_argument('--data',type=Path);p.add_argument('--witnesses',type=Path);p.add_argument('--output',type=Path);p.add_argument('--cell',type=int,nargs=4,action='append',metavar=('LEVEL','X','Y','Z'));p.add_argument('--dilation-radius',type=int,choices=[0,1],default=0);p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:self_test()
 else:
  if any(getattr(a,k) is None for k in ['grid','data','witnesses','output']):p.error('--grid --data --witnesses --output required')
  run(a)
