#!/usr/bin/env python3
"""Read-only geometric/photometric map QC, including frozen train RGB regions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def read_array(path):
 with path.open('rb') as f:
  header=b''
  while header.count(b'&')<3:
   value=f.read(1)
   if not value:raise ValueError(f'Truncated header: {path}')
   header+=value
  width,height,channels=map(int,header[:-1].split(b'&'))
  array=np.fromfile(f,np.float32)
 if array.size!=width*height*channels:raise ValueError(f'Wrong array size: {path}')
 return array.reshape((width,height,channels),order='F').transpose(1,0,2).squeeze()

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def read_consistency_counts(path):
 with path.open('rb') as f:
  header=b''
  while header.count(b'&')<3:
   token=f.read(1)
   if not token:raise ValueError(f'Truncated consistency header: {path}')
   header+=token
  width,height,channels=map(int,header[:-1].split(b'&'))
  records=np.fromfile(f,np.int32)
 counts=np.zeros((height,width),np.int16);offset=0
 while offset<len(records):
  if offset+3>len(records):raise ValueError('Truncated consistency record')
  # The pinned 3.13/5509fffe file stores column, row, count. Validate this
  # against both image bounds and geometric valid-depth support below.
  x,y,n=map(int,records[offset:offset+3])
  if not(0<=y<height and 0<=x<width and 0<=n<=61) or offset+3+n>len(records):raise ValueError('Invalid consistency graph record')
  counts[y,x]=n;offset+=3+n
 return counts

def stats(photo,geom,mask):
 p=np.isfinite(photo)&(photo>0);g=np.isfinite(geom)&(geom>0)
 both=p&g&mask
 result=dict(pixels=int(mask.sum()),photometric_valid=int((p&mask).sum()),geometric_valid=int((g&mask).sum()),photo_valid_geo_invalid=int((p&~g&mask).sum()),photo_invalid_geo_valid=int((~p&g&mask).sum()),geometric_valid_fraction=float(g[mask].mean()))
 if both.any():result['median_photo_geo_relative_depth_change']=float(np.median(np.abs(photo[both]-geom[both])/geom[both]))
 if (g&mask).any():result['geometric_depth_p05_p50_p95']=np.percentile(geom[g&mask],[5,50,95]).tolist()
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',type=Path,required=True);parser.add_argument('--regions',type=Path,required=True);args=parser.parse_args()
 region=json.loads(args.regions.read_text());root=args.arm/'dense/stereo/depth_maps'
 rows=[];seen=set();region_masks={}
 for name,mask in region['masks'].items():
  path=args.regions.parent/mask['file']
  if sha(path)!=mask['sha256']:raise ValueError('Evaluation raster-mask hash mismatch')
  region_masks[name]=np.array(Image.open(path),bool)
 for path in sorted(root.glob('**/*.geometric.bin')):
  name=str(path.relative_to(root))[:-len('.geometric.bin')];photo=read_array(root/(name+'.photometric.bin'));geom=read_array(path)
  if geom.shape!=(1080,1920) or photo.shape!=geom.shape:raise ValueError(f'Unexpected shape: {name}')
  if not np.any(np.isfinite(geom)&(geom>0)):raise ValueError(f'Empty geometric depth map: {name}')
  digest=sha(path)
  if digest in seen:raise ValueError(f'Duplicated geometric depth map: {name}')
  seen.add(digest);row=dict(file_path=name,sha256=digest,whole_image_qc=stats(photo,geom,np.ones(geom.shape,bool)))
  if name==region['file_path']:
   row['local_regions']={r:stats(photo,geom,m) for r,m in region_masks.items()}
   graph=args.arm/'dense/stereo/consistency_graphs'/(name+'.geometric.bin')
   if graph.exists():
    support=read_consistency_counts(graph)
    valid_depth=np.isfinite(geom)&(geom>0)
    row['consistency_graph_valid_depth_agreement']=float(np.mean((support>0)==valid_depth))
    if row['consistency_graph_valid_depth_agreement']<.999:raise ValueError('Consistency graph coordinates do not match geometric valid depth')
    for r,m in region_masks.items():
     valid=m&np.isfinite(geom)&(geom>0);values=support[valid]
     row['local_regions'][r]['consistency_graph_observation_histogram']=np.bincount(values,minlength=62).tolist()
     row['local_regions'][r]['consistency_graph_mean_observations_valid_depth']=float(values.mean()) if len(values) else None
    row['consistency_graph_sha256']=sha(graph)
   # Compact native map retained for matched evidence; other dense data stays remote.
   np.savez_compressed(args.arm/'review_train_maps.npz',photometric=photo,geometric=geom)
  rows.append(row)
 if len(rows)!=62:raise ValueError(f'Expected62maps, found{len(rows)}')
 cfg=(args.arm/'dense/stereo/patch-match.cfg').read_text().splitlines();source_counts=[len(x.split(',')) for x in cfg[1::2]]
 if len(cfg)!=124 or len(set(source_counts))!=1:raise ValueError('Inconsistent source config')
 local=[r for r in rows if 'local_regions' in r]
 if len(local)!=1:raise ValueError('No unique matched diagnostic train map')
 result=dict(reference_count=62,sources_per_reference=source_counts[0],map_shape=[1080,1920],regions_sha256=sha(args.regions),geometric_coverage_mean=float(np.mean([r['whole_image_qc']['geometric_valid_fraction'] for r in rows])),geometric_coverage_min=float(np.min([r['whole_image_qc']['geometric_valid_fraction'] for r in rows])),local=local[0],rows=rows,interpretation='photo_valid_geo_invalid measures lost support between passes, including geometric refinement and filtering; it does not identify a particular rejection reason. Whole-image fractions are integrity QC only, not rendered image-quality metrics.')
 (args.arm/'depth_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['reference_count','sources_per_reference','geometric_coverage_mean','local']}))

if __name__=='__main__':main()
