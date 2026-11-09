#!/usr/bin/env python3
"""Pack a sealed background-only prediction as a common renderer fallback."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from run_dec5_hair_campaign import atomic_json,sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['background','roi','cameras','output']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);d=json.load(open(a.background/'complete.json'));request=json.load(open(a.background/'request.json'));poses=json.load(open(a.cameras));by={r['camera']:r for r in d['records']}
 if d['camera_manifest_sha256']!=sha(a.cameras) or request['uses_validation_rgb'] or request['uses_test_rgb']:raise ValueError('Unsealed or fitted target background')
 records=[]
 for f in poses['frames']:
  name=f['physical_camera'];path=a.background/(name+'.png')
  if sha(path)!=by[name]['png_sha256']:raise ValueError('Changed background RGB')
  image=cv2.imread(str(path))[...,::-1].astype(np.float32)/255;roi=np.load(a.roi/(name+'.npz'))['roi']
  if image.shape!=(f['h'],f['w'],3) or roi.shape!=image.shape[:2]:raise ValueError('Background/ROI raster mismatch')
  out=a.output/(name+'.npz');np.savez_compressed(out,rgb_display=image,roi=roi);records.append(dict(camera=name,sha256=sha(out)))
 atomic_json(a.output/'complete.json',dict(records=records,camera_manifest_sha256=sha(a.cameras),source_background_receipt_sha256=sha(a.background/'complete.json'),checkpoint_sha256=d['checkpoint_sha256'],method='train-only frozen ray-space BG; no foreground colors',uses_validation_rgb=False,uses_test_rgb=False,artifact_kind='background-only common fallback, not foreground reconstruction'))
if __name__=='__main__':main()
