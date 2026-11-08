from pathlib import Path
import json,shutil,time
import cv2,numpy as np
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from crew_guard_plan_v53 import candidates

def apply(V,frame,cam,instance,components,approval):
 V=Path(V);f=V/'frames'/f'{frame:06d}';p=f/'native_masks'/(cam+'.npz');mp=p.with_suffix('.json');m=json.loads(mp.read_text());cp,_=candidates(V,frame,cam)
 with np.load(cp)as z:mask=z['masks'][instance]
 _,labels=cv2.connectedComponents(mask.astype('uint8'),8);part=np.isin(labels,components)
 with np.load(p)as z:old={k:z[k].copy()for k in z.files}
 after={k:x.copy()for k,x in old.items()};new=part&~old['black']&~old['grey'];after['black']|=new;after['actor_ownership_uncertain']|=new;after['source_ineligible']&=~part;after['qualified_actor_rgb']=old.get('qualified_actor_rgb',np.zeros_like(part))|part
 d=f/'qualified_union_parts_v79'/cam;d.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d/'old_native.npz');shutil.copy2(mp,d/'old_native.json');
 if(f/'qualified_hair_inputs'/cam).exists():shutil.copytree(f/'qualified_hair_inputs'/cam,d/'old_qualified_RGB',dirs_exist_ok=True)
 else:(d/'old_qualified_RGB').mkdir(exist_ok=True)
 assert np.array_equal(after['unknown'],old['unknown'])and not(old['black']&~after['black']).any()and not(old['grey']&~after['grey']).any()
 h=f/'qualified_hair_inputs'/cam;h.mkdir(parents=True,exist_ok=True);np.savez_compressed(h/'qualified_hair_native.npz',qualified_actor_rgb=after['qualified_actor_rgb']);tmp=p.with_suffix('.npz.parttmp')
 with tmp.open('wb')as stream:np.savez_compressed(stream,**after)
 tmp.replace(p);proof=dict(frame=frame,camera=cam,kind='ROOT_QUALIFIED_CURRENT_TWO_ACTOR_UNION_COMPONENTS',source_receipt_sha256=sha(V/'sources'/str(frame)/cam/'receipt.json'),candidate_archive_sha256=sha(cp),instance=instance,native_components=components,ROOT_approval=str(approval),ROOT_approval_sha256=sha(approval),native_hair_archive_sha256=sha(h/'qualified_hair_native.npz'),native_hair_mask_sha256=array_sha(after['qualified_actor_rgb']),new_native_sha256=sha(p),added_actor_union_pixels=int(new.sum()),prior_same_frame_qualified_scope=str((d/'old_qualified_RGB').relative_to(f)),source_only_current_NN_pixels=True,existing_BLACK_GREY_ownership_preserved=True,new_labels_BLACK_are_convention_ownership_uncertain=True,skin_between_actors_NOT_claimed_identified=True,geometry_UNKNOWN_unchanged=True,created=time.time());write(h/'application.json',proof)
 up=f/m['union_resolution']['receipt'];u=json.loads(up.read_text());shutil.copy2(up,d/'old_union_application.json');target=(after['black']|after['grey'])&~after['source_ineligible'];u.update(new_native_sha256=sha(p),eligible_pre_union_sha256=array_sha(target),eligible_post_union_sha256=array_sha(target),ownership_uncertain_sha256=array_sha(after['actor_ownership_uncertain']),pre_union_eligible_pixels=int(target.sum()),post_union_eligible_pixels=int(target.sum()),qualified_current_actor_additions=[dict(kind='current_source_qualified_union_parts_and_prior_hair',receipt=str((h/'application.json').relative_to(f)),receipt_sha256=sha(h/'application.json'))],qualified_actor_rgb_sha256=array_sha(after['qualified_actor_rgb']));write(up,u);m.update(archive_sha256=sha(p));m['union_resolution'].update(receipt_sha256=sha(up),eligible_pre_union_sha256=array_sha(target));m['qualified_hair']=dict(receipt=str((h/'application.json').relative_to(f)),receipt_sha256=sha(h/'application.json'),mask_sha256=array_sha(after['qualified_actor_rgb']));write(mp,m);return proof
