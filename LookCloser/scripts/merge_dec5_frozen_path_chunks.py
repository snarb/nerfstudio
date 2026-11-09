#!/usr/bin/env python3
"""Validate and combine disjoint frozen-path receipts without refitting/rendering."""
from pathlib import Path
import argparse,json,os
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--cameras',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--chunks',type=int,default=4);p.add_argument('--depth-key',default='median');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if not a.arm or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.arm):raise ValueError('Simple arm name required')
 prediction_name='prediction' if a.depth_key=='median' else 'prediction_'+a.depth_key
 expected=[f['physical_camera'] for f in json.load(open(a.cameras))['frames']];records={};chunks=[];common=None
 for k in range(a.chunks):
  root=a.campaign/'artifacts'/('path_'+a.arm+'_'+str(k));control=a.campaign/('path_'+a.arm+'_'+str(k)+'_control');result=json.load(open(control/'result.json'))
  if result['returncode'] or result['oom_evidence']:raise ValueError('Chunk controller failed')
  done=json.load(open(root/'complete.json'));receipt=json.load(open(root/prediction_name/'complete.json'));depth=json.load(open(root/'depth/complete.json'));bg=json.load(open(root/'fallback/complete.json'))
  if sha(root/prediction_name/'complete.json')!=done.get('prediction_receipts',{}).get(a.depth_key,done.get('prediction_receipt_sha256')):raise ValueError('Changed chunk receipt')
  if not depth['frozen_state_unchanged'] or done['fitting'] or receipt['uses_validation_rgb'] or receipt['uses_test_rgb']:raise ValueError('Frozen train-only provenance required')
  invariants={key:receipt.get(key,'median') if key=='depth_key' else receipt[key] for key in ['depth_key','checkpoint_sha256','source_receipt_sha256','source_rgb_dataset_sha256','source_rgb','source_count','source_selection']};invariants['background_checkpoint_sha256']=bg['checkpoint_sha256']
  if common is None:common=invariants
  elif common!=invariants:raise ValueError('Chunks disagree on field/source/fallback identity')
  for record in receipt['records']:
   name=record['camera']
   if name in records:raise ValueError('Duplicate camera')
   for extension,key in [('npz','npz_sha256'),('png','png_sha256')]:
    if sha(root/prediction_name/(name+'.'+extension))!=record[key]:raise ValueError('Changed render payload')
   records[name]=(record,root/prediction_name)
  chunks.append(dict(path=str(root),complete_sha256=sha(root/'complete.json'),depth_receipt_sha256=sha(root/'depth/complete.json'),fallback_receipt_sha256=sha(root/'fallback/complete.json')))
 if set(records)!=set(expected):raise ValueError('Incomplete path inventory')
 a.output.mkdir(parents=True,exist_ok=False)
 for name in expected:
  record,source=records[name]
  for suffix in ['.npz','.png','_review.jpg']:
   if (source/(name+suffix)).exists():os.link(source/(name+suffix),a.output/(name+suffix))
 atomic_json(a.output/'complete.json',dict(records=[records[name][0] for name in expected],chunks=chunks,**common,camera_manifest_sha256=sha(a.cameras),uses_validation_rgb=False,uses_test_rgb=False,artifact_kind='merged disjoint frozen head-ROI path; no fitting',source='per-chunk provenance retained; immutable payload SHA256 checked'))
 print('Merged verified frozen path',a.arm,len(expected))
if __name__=='__main__':main()
