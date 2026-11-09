#!/usr/bin/env python3
"""Apply a frozen hair-domain geometry/RGB override; preserve parent elsewhere."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['parent','candidate','domain','output']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);parent=json.load(open(a.parent/'complete.json'));candidate=json.load(open(a.candidate/'complete.json'));domain=json.load(open(a.domain/'complete.json'));p_by={r['camera']:r for r in parent['records']};d_by={r['camera']:r for r in domain['records']}
 if parent['source_rgb']!=candidate['source_rgb'] or parent['fallback_receipt_sha256']!=candidate['fallback_receipt_sha256']:raise ValueError('Requires identical source RGB/order and fallback')
 records=[]
 for row in candidate['records']:
  name=row['camera'];pp=a.parent/(name+'.npz');cp=a.candidate/(name+'.npz');dp=a.domain/(name+'.npz')
  if sha(pp)!=p_by[name]['npz_sha256'] or sha(cp)!=row['npz_sha256'] or sha(dp)!=d_by[name]['sha256']:raise ValueError('Changed override inputs')
  before=np.load(pp);after=np.load(cp);mask=np.load(dp)['domain'];np.testing.assert_array_equal(before['roi'],after['roi'])
  if set(before.files)!=set(after.files):raise ValueError('Override buffer mismatch')
  arrays={}
  for key in before.files:
   original=before[key];new=after[key]
   if original.shape!=new.shape or original.shape[:2]!=mask.shape:raise ValueError('Override raster mismatch')
   value=original.copy();value[mask]=new[mask]
   if not np.array_equal(value[~mask],original[~mask]):raise ValueError('Outside-domain buffer changed')
   arrays[key]=value
  np.savez_compressed(a.output/(name+'.npz'),**arrays);image=cv2.imread(str(a.parent/(name+'.png')));new_image=cv2.imread(str(a.candidate/(name+'.png')))
  if sha(a.parent/(name+'.png'))!=p_by[name]['png_sha256'] or sha(a.candidate/(name+'.png'))!=row['png_sha256']:raise ValueError('Changed RGB input')
  image[mask]=new_image[mask];cv2.imwrite(str(a.output/(name+'.png')),image);cv2.imwrite(str(a.output/(name+'_review.jpg')),cv2.resize(np.rot90(image),(360,640),interpolation=cv2.INTER_AREA));records.append(dict(camera=name,npz_sha256=sha(a.output/(name+'.npz')),png_sha256=sha(a.output/(name+'.png')),override_pixels=int(mask.sum())))
 atomic_json(a.output/'complete.json',dict(records=records,parent_receipt_sha256=sha(a.parent/'complete.json'),hair_candidate_receipt_sha256=sha(a.candidate/'complete.json'),domain_receipt_sha256=sha(a.domain/'complete.json'),method='frozen per-pixel hair geometry override; exact parent buffers and RGB elsewhere',source_rgb=parent['source_rgb'],fallback_receipt_sha256=parent['fallback_receipt_sha256'],uses_validation_rgb=False,uses_test_rgb=False,artifact_kind='head ROI; two field sources, not one globally reconstructed surface'))
if __name__=='__main__':main()
