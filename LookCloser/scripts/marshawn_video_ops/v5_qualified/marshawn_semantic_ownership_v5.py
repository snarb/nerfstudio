"""Retain native ownership uncertainty in DATA without changing RGB confidence."""
from pathlib import Path
import argparse,json,time
import numpy as np
from prepare_marshawn_tuning import sha,write
from marshawn_actor_coordinates import raw_mask_coverage_to_HD
from marshawn_union_resolution_v5 import array_sha

def apply(root,frame):
 f=root/'frames'/f'{frame:06d}';data=f/'data'
 if(f/'dispatch_ready.json').exists():raise ValueError('Cannot change dispatched DATA')
 ready=json.loads((data/'ready.json').read_text());records=[]
 for row in ready['records']:
  camera=row['camera'];source=json.loads((root/'sources'/str(frame)/camera/'receipt.json').read_text());native=f/'native_masks'/f'{camera}.npz'
  with np.load(native)as z:u=z['actor_ownership_uncertain'].copy();qualified_rgb=z['qualified_actor_rgb'].copy()if'qualified_actor_rgb'in z.files else np.zeros_like(u)
  with np.load(row['semantic'])as z:arrays={k:z[k].copy()for k in z.files}
  old_union=arrays['black']|arrays['grey'];old_weights={k:arrays[k].copy()for k in ['W32','W64','unknown','source_ineligible','core','band','nominal','valid']};ties=arrays['black']&arrays['grey'];arrays['black'][ties]=False
  arrays['actor_ownership_uncertain']=((raw_mask_coverage_to_HD(u.astype('float32'),source['camera_row'])>1e-6)|ties)&arrays['valid']&~arrays['source_ineligible']
  arrays['qualified_actor_rgb']=(raw_mask_coverage_to_HD(qualified_rgb.astype('float32'),source['camera_row'])>1e-6)&arrays['valid']&~arrays['source_ineligible']
  arrays['source_unknown']=arrays['unknown']&~arrays['qualified_actor_rgb']
  if(arrays['qualified_actor_rgb']&(arrays['W64']<=0)).any():raise ValueError('Qualified current hair lost RGB confidence '+camera)
  assert np.array_equal(old_union,arrays['black']|arrays['grey']);assert not(arrays['black']&arrays['grey']).any()
  for k,v in old_weights.items():assert np.array_equal(v,arrays[k]),(camera,k)
  if(arrays['actor_ownership_uncertain']&(arrays['W64']<=0)).any():raise ValueError('Ownership uncertainty became zero RGB confidence '+camera)
  semantic=Path(row['semantic']);temp=semantic.with_suffix('.npz.ownershiptmp')
  with temp.open('wb')as stream:np.savez_compressed(stream,**arrays)
  temp.replace(semantic);row['semantic_sha256']=sha(semantic)
  for actor in ['black','grey']:
   y,x=np.where(arrays[actor])
   if len(x):row['crops'][actor]=[max(0,int(x.min())-32),max(0,int(y.min())-32),min(arrays[actor].shape[1],int(x.max())+33),min(arrays[actor].shape[0],int(y.max())+33)]
  records.append(dict(camera=camera,native_ownership_uncertain_sha256=array_sha(u),native_archive_sha256=sha(native),HD_uncertain_pixels=int(arrays['actor_ownership_uncertain'].sum()),qualified_current_hair_HD_pixels=int(arrays['qualified_actor_rgb'].sum()),source_UNKNOWN_overridden_only_in_qualified_hair=int((arrays['unknown']&arrays['qualified_actor_rgb']).sum()),geometry_UNKNOWN_unchanged=True,HD_ties_resolved_to_grey=int(ties.sum()),eligible_uncertain_all_W64_positive=True,RGB_W32_W64_core_band_nominal_unknown_veto_valid_unchanged=True,semantic_sha256=sha(semantic)))
 ready['ownership_uncertainty_policy']='Separate identity-only uncertainty; never geometry/source UNKNOWN or RGB/loss/hybrid zero. HD actor ties choose grey without changing union.';write(data/'ready.json',ready);receipt=data/'semantic_ownership_v5.json';write(receipt,dict(frame=frame,cameras=len(records),records=records,no_RGB_or_frequency_changes=True,created=time.time()));prepared=json.loads((data/'prepared_ready.json').read_text());prepared['semantic_ownership_binding_sha256']=sha(receipt);write(data/'prepared_ready.json',prepared)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',type=int,required=True);a=p.parse_args();apply(a.root,a.frame)
