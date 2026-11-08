"""Fresh native semantic-pair UNION recovery. Prior complete union proposes only."""
from pathlib import Path
import argparse,json,sys,shutil,time,os,fcntl
import cv2,numpy as np
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from marshawn_union_guard_v53 import validate_union
import marshawn_current_parts_v89 as core

def load(p):return json.loads(Path(p).read_text())
def atomic_npz(path,**arrays):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.npz.parttmp')
 with tmp.open('wb')as out:np.savez_compressed(out,**arrays)
 tmp.replace(p)
def output(frame):return V/'operations/current_native_parts_v89'/f'frame{frame:06d}'
def qualify(frame,bindings):
 O=output(frame);O.mkdir(parents=True,exist_ok=True);rows=[];masks={};inputs={}
 for row in load(bindings)['records']:
  cam=row['camera'];native=V/'frames'/f'{frame:06d}'/'native_masks'/(cam+'.npz');inputs.setdefault(cam,sha(native))
  for q in core.qualify_tile(frame,row):
   x0,y0,x1,y1=row['ROI'];masks.setdefault(cam,np.zeros((4320,8192),bool))[y0:y1,x0:x1]|=q['mask'];rows.append(dict(camera=cam,ROI=row['ROI'],NN_pair={k:v for k,v in q.items()if k!='mask'},spec=row['spec'],spec_sha256=sha(Path(row['spec'])),candidates=row['candidates'],NN_receipt_sha256=sha(Path(row['candidates'])/'receipt.json'),proposal=row['proposal'],proposal_sha256=row['proposal_sha256']))
 for cam,H in masks.items():
  with np.load(V/'frames'/f'{frame:06d}'/'native_masks'/(cam+'.npz'))as z:H&=~(z['black']|z['grey'])
  atomic_npz(O/(cam+'_qualified_new_union.npz'),qualified_new_union=H)
 write(O/'qualification.json',dict(kind='CURRENT_NATIVE_SEMANTIC_PAIR_UNION_QUALIFICATION_V89',frame=frame,records=rows,input_native_sha256=inputs,qualified_archives={cam:dict(path=str(O/(cam+'_qualified_new_union.npz')),sha256=sha(O/(cam+'_qualified_new_union.npz')),array_sha256=array_sha(H),pixels=int(H.sum()))for cam,H in masks.items()},bindings=str(bindings),bindings_sha256=sha(Path(bindings)),scientific_constraints=dict(scores_min=.7,different_prompt_pair_IoU_min=.85,prior_union_identity_and_ROI_only=True,current_NN_component64_bound=True,known_current_CREW_core_excluded=True,union_ownership_not_claimed=True),created=time.time()));return O

def apply_one(frame,cam,O):
 f=V/'frames'/f'{frame:06d}';q=load(O/'qualification.json');record=q['qualified_archives'][cam];p=f/'native_masks'/(cam+'.npz');mp=p.with_suffix('.json');d=O/'applications'/cam;done=d/'complete.json'
 if done.exists():
  r=load(done);assert sha(p)==r['new_native_sha256'];validate_union(V,frame,cam);return r
 assert not(f/'native_semantic_ready.json').exists()and not(f/'dispatch_ready.json').exists(),'Published inputs immutable'
 assert sha(p)==q['input_native_sha256'][cam];archive=Path(record['path']);assert sha(archive)==record['sha256']
 with np.load(archive)as z:delta=z['qualified_new_union'].copy()
 assert array_sha(delta)==record['array_sha256'];meta=load(mp);assert meta['archive_sha256']==sha(p)
 with np.load(p)as z:old={k:z[k].copy()for k in z.files}
 assert not(delta&(old['black']|old['grey'])).any()
 d.mkdir(parents=True,exist_ok=True)
 for src in [p,mp,f/meta['union_resolution']['receipt']]:
  dest=d/('old_'+src.name);shutil.copy2(src,dest);assert sha(dest)==sha(src)
 prior=f/'qualified_hair_inputs'/cam
 if prior.exists():shutil.copytree(prior,d/'prior_qualified_RGB',dirs_exist_ok=True)
 after={k:x.copy()for k,x in old.items()};after['black']|=delta;after['actor_ownership_uncertain']|=delta;after['qualified_actor_rgb']=old.get('qualified_actor_rgb',np.zeros_like(delta))|delta;after['source_ineligible']&=~delta
 for k in ['grey','unknown']:assert np.array_equal(old[k],after[k])
 assert not(old['black']&~after['black']).any()and not(after['black']&after['grey']).any();assert np.array_equal(old['source_ineligible']&~delta,after['source_ineligible'])
 prior.mkdir(parents=True,exist_ok=True);atomic_npz(prior/'qualified_hair_native.npz',qualified_actor_rgb=after['qualified_actor_rgb']);atomic_npz(p,**after)
 proof=dict(kind='QUALIFIED_CURRENT_NATIVE_SEMANTIC_PAIR_TWO_ACTOR_UNION_V89',frame=frame,camera=cam,source_receipt_sha256=sha(V/'sources'/str(frame)/cam/'receipt.json'),qualification=str(O/'qualification.json'),qualification_sha256=sha(O/'qualification.json'),qualified_native_delta_archive=str(archive),qualified_native_delta_archive_sha256=sha(archive),native_hair_archive_sha256=sha(prior/'qualified_hair_native.npz'),native_hair_mask_sha256=array_sha(after['qualified_actor_rgb']),prior_same_frame_qualified_RGB=str(d/'prior_qualified_RGB'),new_native_sha256=sha(p),added_actor_union_pixels=int(delta.sum()),source_veto_trim_pixels=int((old['source_ineligible']&delta).sum()),new_labels_BLACK_are_convention_ownership_uncertain=True,all_old_actor_pixels_and_ownership_preserved=True,geometry_UNKNOWN_unchanged=True,no_previous_bitmap_as_foreground=True,created=time.time());write(prior/'application.json',proof)
 up=f/meta['union_resolution']['receipt'];u=load(up);target=(after['black']|after['grey'])&~after['source_ineligible'];u.update(new_native_sha256=sha(p),eligible_pre_union_sha256=array_sha(target),eligible_post_union_sha256=array_sha(target),ownership_uncertain_sha256=array_sha(after['actor_ownership_uncertain']),pre_union_eligible_pixels=int(target.sum()),post_union_eligible_pixels=int(target.sum()),qualified_current_actor_additions=[dict(kind='fresh_native_semantic_pair_union_and_prior_qualified_RGB',receipt=str((prior/'application.json').relative_to(f)),receipt_sha256=sha(prior/'application.json'))],qualified_actor_rgb_sha256=array_sha(after['qualified_actor_rgb']));write(up,u);meta.update(archive_sha256=sha(p));meta['union_resolution'].update(receipt_sha256=sha(up),eligible_pre_union_sha256=array_sha(target));meta['qualified_hair']=dict(receipt=str((prior/'application.json').relative_to(f)),receipt_sha256=sha(prior/'application.json'),mask_sha256=array_sha(after['qualified_actor_rgb']));write(mp,meta);validate_union(V,frame,cam);write(done,proof);return proof

def main(frame,bindings,apply=False):
 O=output(frame)
 if not(O/'qualification.json').exists():qualify(frame,bindings)
 if apply:
  rows=[apply_one(frame,cam,O)for cam,q in load(O/'qualification.json')['qualified_archives'].items()if q['pixels']>0];write(O/'applications_complete.json',dict(frame=frame,records=rows,qualified_current_NN_only=True,all_union_conservation_pass=True))
 return O
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--frame',type=int,required=True);p.add_argument('--bindings',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args();lock=open(V/'operations/source_factory_v74.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX) if a.apply else None;main(a.frame,a.bindings,a.apply)
