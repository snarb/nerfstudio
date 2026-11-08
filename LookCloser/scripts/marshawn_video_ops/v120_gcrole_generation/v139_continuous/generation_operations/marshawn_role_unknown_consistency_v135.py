"""Unproved role pixels cannot re-enter FG through another proposal route."""
from pathlib import Path
import json,time,shutil
import numpy as np
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from marshawn_union_guard_v53 import validate_union
from crew_guard_plan_v53 import candidates

def masks(root,frame,semantic_qualification,preapplication_root=None):
 root=Path(root);q=json.loads(Path(semantic_qualification).read_text());out={};proofs={}
 for row in q['records']:
  n=row['NN_pair']
  if not n.get('ROLE_UNKNOWN_only'):continue
  cam=row['camera'];p=Path(n['role_unknown_current_NN_archive']);assert sha(p)==n['role_unknown_current_NN_archive_sha256']
  with np.load(p)as z:r=z['role_unknown'];x0,y0,x1,y1=z['crop_xyxy']
  out.setdefault(cam,np.zeros((4320,8192),bool))[y0:y1,x0:x1]|=r;proofs.setdefault(cam,[]).append(dict(archive=str(p),sha256=sha(p),NN_pair=n))
 for cam,R in out.items():
  native=Path(preapplication_root)/(cam+'.npz')if preapplication_root else root/'frames'/f'{frame:06d}'/'native_masks'/(cam+'.npz');assert sha(native)==q['input_native_sha256'][cam]
  with np.load(native)as z:assert not(R&(z['black']|z['grey'])).any()
  cp,meta=candidates(root,frame,cam)
  with np.load(cp)as z:assert not(R&z['masks'][[x for x in meta['selected'].values()if x is not None]].any(0)).any()
 return out,proofs

def filter_combined(combined,base_actor,R):
 assert not(R&base_actor).any(),'ROLEUNKNOWN may never erase an existing actor'
 result=combined&~R;assert not(result&R).any()
 return result,dict(other_routes_attempted_ROLEUNKNOWN_FG=int((combined&R).sum()),ROLEUNKNOWN_after_all_routes_FG=0,preexisting_actor_pixels_preserved=True)

def apply_visibility(root,frame,unknown_masks,proofs):
 root=Path(root);f=root/'frames'/f'{frame:06d}';records=[]
 for cam,R in unknown_masks.items():
  if not R.any():continue
  p=f/'native_masks'/(cam+'.npz');mp=p.with_suffix('.json');meta=json.loads(mp.read_text());u_path=f/meta['union_resolution']['receipt'];u=json.loads(u_path.read_text());before=sha(p)
  O=f/'role_unknown_visibility_inputs_v135'/cam;O.mkdir(parents=True,exist_ok=True);done=O/'complete.json'
  if done.exists():
   r=json.loads(done.read_text());assert r['new_native_sha256']==sha(p);validate_union(root,frame,cam);records.append(r);continue
  backup=O/'before';backup.mkdir(exist_ok=True)
  for src,name in[(p,'native.npz'),(mp,'native.json'),(u_path,'union.json')]:
   if not(backup/name).exists():shutil.copy2(src,backup/name)
  meta=json.loads((backup/'native.json').read_text());u=json.loads((backup/'union.json').read_text());before=sha(backup/'native.npz')
  with np.load(backup/'native.npz')as z:old={k:z[k].copy()for k in z.files}
  assert not(R&(old['black']|old['grey']|old.get('qualified_actor_rgb',False))).any(),'Unknown role reintroduced by another FG route'
  pre=Path(u['pre_overlap_archive']);assert sha(pre)==u['pre_overlap_archive_sha256']
  with np.load(pre)as z:raw_pre=z['pre_black']|z['pre_grey']
  after={k:v.copy()for k,v in old.items()};after['unknown']|=R;soft=R&~raw_pre;after['source_ineligible'][soft]=False
  # No label/ownership/current actor or independently retained NN PRE is erased.
  for k in['black','grey','actor_ownership_uncertain','qualified_actor_rgb']:
   if k in old:assert np.array_equal(after[k],old[k])
  target=(after['black']|after['grey'])&~after['source_ineligible'];assert np.array_equal(target,(old['black']|old['grey'])&~old['source_ineligible'])
  O=f/'role_unknown_visibility_inputs_v135'/cam;O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/'CURRENT_NN_role_UNKNOWN.npz',role_unknown=R,soft_eligible=soft)
  tmp=p.with_suffix('.npz.role_unknown_tmp')
  with tmp.open('wb')as out:np.savez_compressed(out,**after)
  tmp.replace(p);u['new_native_sha256']=sha(p);write(u_path,u);meta['archive_sha256']=sha(p);meta['union_resolution']['receipt_sha256']=sha(u_path);receipt=O/'receipt.json';r=dict(frame=frame,camera=cam,kind='CURRENT_NN_ROLE_UNCERTAINTY_NO_FOREGROUND_OR_BACKGROUND_CERTIFICATE',source_receipt_sha256=sha(root/'sources'/str(frame)/cam/'receipt.json'),before_native_sha256=before,new_native_sha256=sha(p),scope_array_sha256=array_sha(R),scope_pixels=int(R.sum()),scope_archive=str(O/'CURRENT_NN_role_UNKNOWN.npz'),scope_archive_sha256=sha(O/'CURRENT_NN_role_UNKNOWN.npz'),soft_eligible_current_refined_PRE0_pixels=int(soft.sum()),existing_veto_retained_on_current_NN_PRE=int((R&raw_pre).sum()),physical_UNKNOWN_OR_current_NN_only=True,all_existing_actor_H_U_preserved=True,no_hard_RGB_weights_assigned=True,unchanged_frozen32_64_falloff_required=True,NN_role_proofs=proofs[cam],created=time.time());write(receipt,r);meta['ROLE_UNKNOWN_visibility']=dict(receipt=str(receipt.relative_to(f)),receipt_sha256=sha(receipt));write(mp,meta);validate_union(root,frame,cam);write(done,r);records.append(r)
 return records
