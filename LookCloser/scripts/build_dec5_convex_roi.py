#!/usr/bin/env python3
"""Repair a fragmented train-projected evaluation domain without target RGB."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['parent','cameras','output']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);parent=json.load(open(a.parent/'complete.json'));by={r['camera']:r for r in parent['records']};rows=[]
 if parent.get('uses_validation_rgb') or parent.get('uses_test_rgb'):raise ValueError('Train-derived parent required')
 for f in json.load(open(a.cameras))['frames']:
  name=f['physical_camera'];path=a.parent/(name+'.npz');digest=by[name].get('npz_sha256',by[name].get('sha256'))
  if sha(path)!=digest:raise ValueError('Changed ROI parent')
  old=np.load(path)['roi'].astype(bool)
  if old.shape!=(f['h'],f['w']) or not old.any():raise ValueError('Invalid fixed ROI raster')
  contours,_=cv2.findContours(old.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);points=np.concatenate(contours);hull=cv2.convexHull(points);mask=np.zeros(old.shape,np.uint8);cv2.fillConvexPoly(mask,hull,1);roi=mask.astype(bool)
  if (old&~roi).any():raise ValueError('Domain repair removed parent pixels')
  out=a.output/(name+'.npz');np.savez_compressed(out,roi=roi);rows.append(dict(camera=name,npz_sha256=sha(out),parent_pixels=int(old.sum()),pixels=int(roi.sum()),added_pixels=int((roi&~old).sum())))
 atomic_json(a.output/'complete.json',dict(records=rows,parent_receipt_sha256=sha(a.parent/'complete.json'),camera_manifest_sha256=sha(a.cameras),method='filled convex hull of every nonzero parent ROI pixel; fixed generic rule, no target RGB or semantic labels',domain_version='convex_hull_v2',uses_validation_rgb=False,uses_test_rgb=False,validation_revealed_fragmented_old_domain=True,implementation_sha256=sha(__file__)))
 print('Created train-derived convex ROI',len(rows))
if __name__=='__main__':main()
