#!/usr/bin/env python3
"""Connected camera-path domain from an existing train-normalized model box."""
from pathlib import Path
import argparse,json,itertools
import cv2,numpy as np
from dec5_hair import project
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['grid','cameras','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--padding',type=int,default=64);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 receipt=json.load(open(a.grid.parent/'complete.json'))
 if receipt['uses_validation_rgb'] or receipt['uses_test_rgb'] or sha(a.grid)!=receipt['votes_sha256']:raise ValueError('Verified train-derived normalization required')
 data=np.load(a.grid);bounds=data['bounds'];corners=np.array(list(itertools.product(*zip(bounds[0],bounds[1]))));world=(corners-data['translation'])@data['rotation']/data['scale'];rows=[]
 for f in json.load(open(a.cameras))['frames']:
  uv,z=project(world,f)
  if (z<=0).any() or not np.isfinite(uv).all():raise ValueError('Model box behind camera; use a proper clipped frustum')
  hull=cv2.convexHull(np.rint(uv).astype(np.int32));mask=np.zeros((f['h'],f['w']),np.uint8);cv2.fillConvexPoly(mask,hull,1)
  if a.padding:mask=cv2.dilate(mask,np.ones((2*a.padding+1,2*a.padding+1),np.uint8))
  if not mask.any():raise ValueError('Model bound misses camera')
  path=a.output/(f['physical_camera']+'.npz');np.savez_compressed(path,roi=mask.astype(bool));rows.append(dict(camera=f['physical_camera'],npz_sha256=sha(path),pixels=int(mask.sum())))
 atomic_json(a.output/'complete.json',dict(records=rows,camera_manifest_sha256=sha(a.cameras),normalization_receipt_sha256=sha(a.grid.parent/'complete.json'),method='project eight corners of the train-derived bounded LookCloser model box; fill convex hull and add fixed pixel margin',padding_pixels=a.padding,uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__)))
 print('Created model-bound path domain',len(rows))
if __name__=='__main__':main()
