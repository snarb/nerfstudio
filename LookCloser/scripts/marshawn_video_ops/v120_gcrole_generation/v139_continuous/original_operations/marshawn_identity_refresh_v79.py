"""Refresh cached CURRENT NN identity against latest qualified 2D reference."""
from pathlib import Path
import json,shutil,time
import cv2,numpy as np
from prepare_marshawn_tuning import sha,write
from marshawn_video_prep_masks import select,small
from marshawn_video_prep_mask_components import qualify
from crew_guard_plan_v53 import candidates
from marshawn_union_resolution_v5 import array_sha

def refresh(V,frame):
 V=Path(V);f=V/'frames'/f'{frame:06d}';out=f/'identity_refresh_inputs';out.mkdir(exist_ok=True);rows=[]
 if(f/'native_semantic_ready.json').exists():return
 previous=frame-5
 for cam in json.loads((V/'source_inventory.json').read_text())['cameras']:
  p=f/'native_masks'/(cam+'.npz');mp=p.with_suffix('.json');m=json.loads(mp.read_text());ref=V/'frames'/f'{previous:06d}'/'native_masks'/(cam+'.npz')
  if m['identity_reference_frame']==previous and m['identity_reference_sha256']==sha(ref):continue
  cp,cm=candidates(V,frame,cam)
  with np.load(ref)as z:reference={a:small(z[a]&~z['actor_ownership_uncertain'])for a in ['black','grey']}
  with np.load(cp)as z:masks=z['masks']
  ids,stats=select(masks,reference)
  if any(x is None for x in ids.values()):raise RuntimeError('Latest-reference current identity stillUNKNOWN '+cam)
  oldids={a:q['instance']for a,q in m['identity'].items()};d=out/cam;d.mkdir(exist_ok=True)
  for q in [p,mp,cp.with_suffix('.json')]:
   dest=d/('old_'+q.name)
   if not dest.exists():shutil.copy2(q,dest)
  proof=dict(frame=frame,camera=cam,reference_frame=previous,reference_native_sha256=sha(ref),candidate_archive_sha256=sha(cp),old_selected=oldids,selected=ids,stats=stats,selector_sha256=sha(__import__('marshawn_video_prep_masks').__file__),same_CURRENT_NN_only=True,old_native_sha256=sha(p),old_metadata_sha256=sha(mp),native_boundary_refinement_reference_preserved=m['identity_reference_frame'],created=time.time())
  if ids!=oldids:
   with np.load(p)as z:old={k:z[k].copy()for k in z.files}
   raw={a:masks[k].copy()for a,k in ids.items()};nominal=raw['black']|raw['grey'];others=[];ambiguous=[]
   for k,mask in enumerate(masks):
    if k in ids.values():continue
    (ambiguous if (mask&nominal).sum()/max(int(mask.sum()),1)>.1 else others).append(k)
   veto=masks[others].any(0)if others else np.zeros_like(nominal);unknown=veto.copy()
   if ambiguous:unknown|=masks[ambiguous].any(0)&~nominal
   cleaned={};components={}
   for a in raw:cleaned[a],discarded,components[a]=qualify(raw[a],reference[a]);unknown|=discarded
   H=old.get('qualified_actor_rgb',np.zeros_like(nominal));old_union=old['black']|old['grey'];fresh={a:(cleaned[a]if ids[a]!=oldids[a]else np.zeros_like(nominal))for a in ['black','grey']};fresh_union=fresh['black']|fresh['grey'];veto=(veto|old['source_ineligible'])&~old_union&~H&~fresh_union;unknown|=veto
   # Current source-qualified manual CREW, when present, keeps priority.
   crew=f/'crew_inputs'/(cam+'_bald_helper.npz')
   if crew.exists():
    with np.load(crew)as z:x0,y0,x1,y1=z['crop_xyxy'];veto[y0:y1,x0:x1]|=z['mask']&~H[y0:y1,x0:x1]
   pb=old['black']|fresh['black'];pg=old['grey']|fresh['grey'];eligible=(pb|pg|H)&~veto
   after={k:x.copy()for k,x in old.items()};after['black']=old['black']|(fresh['black']&~old['grey']&~veto);after['grey']=old['grey']|(fresh['grey']&~after['black']&~veto);after['source_ineligible']=veto;after['unknown']=unknown;after['actor_ownership_uncertain']=(old['actor_ownership_uncertain']|(pb&pg))&eligible
   assert not(old['black']&~after['black']).any()and not(old['grey']&~after['grey']).any();assert np.array_equal((after['black']|after['grey'])&~veto,eligible)
   pre=d/'current_identity_pre_base.npz';np.savez_compressed(pre,pre_black=pb,pre_grey=pg,source_ineligible=veto,unknown=unknown)
   tmp=p.with_suffix('.npz.refresh_tmp')
   with tmp.open('wb')as stream:np.savez_compressed(stream,**after)
   tmp.replace(p);up=f/m['union_resolution']['receipt'];u=json.loads(up.read_text());shutil.copy2(up,d/'old_union_application.json');u.update(pre_overlap_archive=str(pre),pre_overlap_archive_sha256=sha(pre),pre_input_kind='CURRENT_CACHED_GLOBAL_NN_WITH_LATEST_QUALIFIED_IDENTITY_PLUS_PRESERVED_CURRENT_NATIVE_FOREGROUND',new_native_sha256=sha(p),eligible_pre_union_sha256=array_sha(eligible),eligible_post_union_sha256=array_sha(eligible),ownership_uncertain_sha256=array_sha(after['actor_ownership_uncertain']),pre_union_eligible_pixels=int(eligible.sum()),post_union_eligible_pixels=int(eligible.sum()),current_identity_refresh_proof=str((d/'refresh.json').relative_to(f)),historical_original_refine_replay_claim=False);write(up,u);m['union_resolution'].update(receipt_sha256=sha(up),eligible_pre_union_sha256=array_sha(eligible));proof.update(new_native_sha256=sha(p),added_current_actor_pixels=int((eligible&~old_union).sum()),components=components,failed_identity_ALL_UNKNOWN_recomputed_from_CURRENT_candidates=bool(any(x is None for x in oldids.values())),original_current_actor_union_preserved=True,original_H_preserved=True,qualified_CREW_priority=True)
  else:proof['new_native_sha256']=sha(p)
  write(d/'refresh.json',proof);m.update(archive_sha256=sha(p),identity=stats,identity_reference_frame=previous,identity_reference_sha256=sha(ref));m['identity_refresh']=dict(receipt=str((d/'refresh.json').relative_to(f)),receipt_sha256=sha(d/'refresh.json'),original_refinement_reference_frame=proof['native_boundary_refinement_reference_preserved']);write(mp,m)
  cm.update(initial_observer_selected=cm.get('initial_observer_selected',cm.get('selected')),selected=ids,latest_current_identity_refresh=dict(receipt=str((d/'refresh.json').relative_to(f)),receipt_sha256=sha(d/'refresh.json'),reference_frame=previous,reference_native_sha256=sha(ref)));write(cp.with_suffix('.json'),cm);rows.append(proof)
 write(out/'all66_latest_reference_ready.json',dict(frame=frame,reference_frame=previous,records=rows,all66_latest_identity_provenance=True,no_NN_redo_or_model_state=True));return rows
