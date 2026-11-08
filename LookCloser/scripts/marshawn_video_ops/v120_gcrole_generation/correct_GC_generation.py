from pathlib import Path
import sys,os,json,time,hashlib,fcntl,shutil
import cv2,numpy as np
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from marshawn_union_guard_v53 import validate_union

def atomic_npz(p,**d):
 t=p.with_suffix('.npz.tmp')
 with t.open('wb')as f:np.savez_compressed(f,**d)
 os.replace(t,p)
def load(p):return json.loads(Path(p).read_text())
def correction(frame):
 cam='GC';f=G/'frames'/f'{frame:06d}';origin=V/'frames'/f'{frame:06d}';O=f/'generation_corrections/GC_visible_CREW_hand';done=O/'complete.json'
 if done.exists():
  r=load(done);assert sha(f/'native_masks/GC.npz')==r['new_native_sha256'];return r
 assert not(f/'native_semantic_ready.json').exists();O.mkdir(parents=True,exist_ok=True)
 qpath=V/'operations/current_native_parts_v105'/f'frame{frame:06d}'/'qualification.json';qual=load(qpath);rows=[r for r in qual['records']if r['camera']=='GC'and r['NN_pair']['family']=='hand'and r['NN_pair']['new_actor_union_pixels']>0];assert len(rows)==1
 row=rows[0];q=row['NN_pair'];x0,y0,x1,y1=row['ROI'];ms=[]
 for a in[q['a'],q['b']]:
  assert sha(a['archive'])==a['archive_sha256']
  with np.load(a['archive'])as z:ms.append(z['masks'][a['index']])
 _,labels=cv2.connectedComponents((ms[0]&ms[1]).astype('uint8'),8);part=labels==q['NN_intersection_component']
 # Reproduce the exact proposal-bounded scope used by original qualification.
 from crew_guard_plan_v53 import candidates
 cp,meta=candidates(V,frame,cam)
 with np.load(cp)as z:masks=z['masks']
 bindings=load(V/'operations/current_native_parts_v105'/f'frame{frame:06d}'/'bindings.json')['records'];b=next(r for r in bindings if r['camera']==cam and r['ROI']==row['ROI']);allowed=np.zeros_like(part)
 for pr in b['components']:
  _,lab=cv2.connectedComponents(masks[pr['instance']].astype('uint8'),8);allowed|=lab[y0:y1,x0:x1]==pr['native_component']
 allowed=cv2.distanceTransform((~allowed).astype('uint8'),cv2.DIST_L2,5)<=64;part&=allowed
 bad=np.zeros((4320,8192),bool);bad[y0:y1,x0:x1]=part
 prep=V/'operations/ordered_source_stage_v107'/f'frame{frame:06d}'/'applications/GC/old_GC.npz'
 with np.load(prep)as z:pre={k:z[k].copy()for k in z.files}
 with np.load(origin/'native_masks/GC.npz')as z:old={k:z[k].copy()for k in z.files}
 assert int(part.sum())==q['new_actor_union_pixels']and not(bad&(pre['black']|pre['grey'])).any()
 assert bad[bad].all()and old['black'][bad].all()and old['qualified_actor_rgb'][bad].all()
 assert not(old['grey']&bad).any()
 p=f/'native_masks/GC.npz';mp=p.with_suffix('.json');metadata=load(mp);up=f/metadata['union_resolution']['receipt'];u=load(up)
 before=O/'before';before.mkdir(exist_ok=True)
 for src in[p,mp,up,f/'qualified_hair_inputs/GC/application.json',f/'qualified_hair_inputs/GC/qualified_hair_native.npz']:
  shutil.copy2(src,before/(src.parent.name+'_'+src.name));assert sha(src)==sha(before/(src.parent.name+'_'+src.name))
 after={k:v.copy()for k,v in old.items()}
 for k in['black','actor_ownership_uncertain','qualified_actor_rgb']:after[k][bad]=pre[k][bad]if k in pre else False
 after['source_ineligible'][bad]=pre['source_ineligible'][bad]
 assert np.array_equal(after['grey'],old['grey'])and np.array_equal(after['unknown'],old['unknown'])
 for k in old:
  assert np.array_equal(after[k][~bad],old[k][~bad])if old[k].shape==bad.shape else np.array_equal(after[k],old[k]),k
 assert not((old['black']&~bad)&~after['black']).any()
 atomic_npz(O/'retracted_wrong_human_NN_component.npz',wrong_CREW_actor_part=bad)
 Hdir=f/'qualified_hair_inputs/GC';atomic_npz(Hdir/'qualified_hair_native.npz',qualified_actor_rgb=after['qualified_actor_rgb']);atomic_npz(p,**after)
 proof=dict(kind='CURRENT_NATIVE_HUMAN_ROLE_CORRECTION_NEW_SOURCE_GENERATION_V1',frame=frame,camera=cam,generation='gcrole_g1',source_receipt_sha256=sha(G/'sources'/str(frame)/cam/'receipt.json'),ROOT_scope='Parent authorized corrected new source generations1695/1700/1705 after direct native photo review of seated CREW hand; anatomy class and prior flow were not target identity.',origin_source_root=str(V),origin_published_source_sha256=sha(origin/'native_semantic_ready.json'),origin_native_sha256=sha(origin/'native_masks/GC.npz'),original_SEM_qualification=str(qpath),original_SEM_qualification_sha256=sha(qpath),original_SEM_record=row,immutable_pre_SEM_native=str(prep),immutable_pre_SEM_native_sha256=sha(prep),retracted_mask_archive=str(O/'retracted_wrong_human_NN_component.npz'),retracted_mask_archive_sha256=sha(O/'retracted_wrong_human_NN_component.npz'),retracted_mask_array_sha256=array_sha(bad),retracted_pixels=int(bad.sum()),native_hair_archive_sha256=sha(Hdir/'qualified_hair_native.npz'),native_hair_mask_sha256=array_sha(after['qualified_actor_rgb']),new_native_sha256=sha(p),all_unrelated_actor_pixels_preserved=True,geometry_UNKNOWN_array_unchanged=True,source_veto_restored_from_immutable_preSEM_exact_scope=True,original_published_generation_not_modified=True,retained_qualified_H_is_same_frame_original_current_NN_evidence=True,no_NN_inference_or_training_claim=True,created=time.time());write(Hdir/'application.json',proof)
 target=(after['black']|after['grey'])&~after['source_ineligible'];u.update(new_native_sha256=sha(p),eligible_pre_union_sha256=array_sha(target),eligible_post_union_sha256=array_sha(target),ownership_uncertain_sha256=array_sha(after['actor_ownership_uncertain']),pre_union_eligible_pixels=int(target.sum()),post_union_eligible_pixels=int(target.sum()),qualified_current_actor_additions=[dict(kind='retained_current_NN_actor_evidence_after_wrong_human_retraction',receipt='qualified_hair_inputs/GC/application.json',receipt_sha256=sha(Hdir/'application.json'))],qualified_actor_rgb_sha256=array_sha(after['qualified_actor_rgb']),source_generation_role_correction=dict(path=str(O/'complete.json'),origin_native_sha256=sha(origin/'native_masks/GC.npz')));write(up,u)
 metadata['archive_sha256']=sha(p);metadata['union_resolution'].update(receipt_sha256=sha(up),eligible_pre_union_sha256=array_sha(target),actor_ownership_uncertain_pixels=int(after['actor_ownership_uncertain'].sum()));metadata['qualified_hair']=dict(receipt='qualified_hair_inputs/GC/application.json',receipt_sha256=sha(Hdir/'application.json'),mask_sha256=array_sha(after['qualified_actor_rgb']));metadata['source_generation']=dict(generation='gcrole_g1',original_inference_receipts_not_rewritten=True,role_correction_receipt='generation_corrections/GC_visible_CREW_hand/complete.json');write(mp,metadata)
 validate_union(G,frame,cam);proof.update(new_native_metadata_sha256=sha(mp),new_union_receipt_sha256=sha(up),union_conservation_pass=True);write(done,proof)
 print(frame,'CORRECTED',int(bad.sum()),sha(p),flush=True);return proof
if __name__=='__main__':
 cv2.setNumThreads(1);lock=open(G/'operations/source_generation_writer.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
 rows=[correction(f)for f in[1695,1700,1705]];write(G/'operations/finite_GC_role_corrections_complete.json',dict(kind='NEW_SOURCE_GENERATION_GC_CURRENT_ROLE_CORRECTIONS_V1',status='pass',records=rows,no_source_marker_or_model_published=True,created=time.time()))
