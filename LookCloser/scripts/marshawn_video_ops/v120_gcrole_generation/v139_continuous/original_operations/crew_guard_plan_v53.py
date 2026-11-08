"""Read-only ALL-person identity/novel-overlap guard. UNKNOWN blocks publication."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from prepare_marshawn_tuning import sha,write
from marshawn_union_guard_v53 import validate_union
from marshawn_actor_conflict_v51 import previous_actor_prediction,crew_component_conflicts,qualified_current_crew_core
BASE=Path('/fsx/tmp/marshawn_001660_hd/two_actor_isolation/confidence_native_hair_v2/manifest.json')

def small(m):return cv2.resize(np.asarray(m,np.uint8),(512,270),interpolation=cv2.INTER_AREA)>0

def prior_actors(root,frame,camera):
 if frame==1665:
  row=next(r for r in json.loads(BASE.read_text())['rows']if r['camera']==camera)
  if sha(row['raw_archive'])!=row['raw_archive_sha256']:raise ValueError('Baseline actor identity source changed')
  with np.load(row['raw_archive']) as z:
   x0,y0,x1,y1=z['crop_xyxy'];result={}
   for a in ['black','grey']:
    m=np.zeros((4320,8192),bool);m[y0:y1,x0:x1]=z[a];result[a]=small(m)
  return result,row['raw_archive_sha256']
 p=root/'frames'/f'{frame-5:06d}'/'native_masks'/f'{camera}.npz'
 with np.load(p) as z:return {a:small(z[a])for a in ['black','grey']},sha(p)

def candidates(root,frame,camera):
 froot=root/'frames'/f'{frame:06d}'
 for folder in ['person_candidates_sg','person_candidates']:
  p=froot/folder/(camera+'.npz')
  if p.exists():
   meta=json.loads(p.with_suffix('.json').read_text())
   if sha(p)!=meta['archive_sha256']:raise ValueError('Current-frame person candidates changed')
   original=meta.get('original_source_receipt_sha256',meta['source_receipt_sha256'])
   if original!=sha(root/'sources'/str(frame)/camera/'receipt.json'):raise ValueError('Candidate source frame changed')
   return p,meta
 p=root/'operations'/f'crew{frame}'/camera/'candidates.npz'
 if p.exists():
  meta=json.loads(p.with_name('receipt.json').read_text());assert sha(p)==meta['candidates_sha256'];return p,meta
 raise FileNotFoundError('Current-frame ALL-person candidates missing '+camera)

def run(root,frame,cameras,diagnostic_partial=False):
 out=root/'frames'/f'{frame:06d}'/'crew_guard';out.mkdir(exist_ok=True);reports=[]
 for camera in cameras:
  if frame>=1675:validate_union(root,frame,camera)
  native=root/'frames'/f'{frame:06d}'/'native_masks'/f'{camera}.npz';record=json.loads(native.with_suffix('.json').read_text())
  if record['archive_sha256']!=sha(native):raise ValueError('Native actor masks changed '+camera)
  with np.load(native) as z:current={a:small(z[a])for a in ['black','grey']};veto=small(z['source_ineligible']);qualified_actor_rgb=z['qualified_actor_rgb'].copy()if'qualified_actor_rgb'in z.files else np.zeros_like(z['black'])
  union=current['black']|current['grey'];prior,prior_sha=prior_actors(root,frame,camera);prior_union=prior['black']|prior['grey'];safe=cv2.dilate(prior_union.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))>0
  current_candidate_path,current_candidate_meta=candidates(root,frame,camera)
  qualified_actor=np.zeros((270,512),bool);qualified_instances={};fragment_proofs=[]
  for fragment in record.get('actor_fragments',[]):
   proofpath=root/'frames'/f'{frame:06d}'/fragment['receipt_path']
   if sha(proofpath)!=fragment['receipt_sha256']:raise ValueError('Qualified current actor fragment proof changed')
   proof=json.loads(proofpath.read_text());assert proof['frame']==frame and proof['camera']==camera and proof['all_existing_actor_pixels_preserved'] and proof['only_current_NN_pixels_added']
   artifact=root/'frames'/f'{frame:06d}'/'actor_fragment_inputs'/f'{camera}_person{proof["person_instance"]}_component{proof["component"]}.npz'
   if proof['candidate_archive_sha256']!=sha(current_candidate_path)or proof['source_receipt_sha256']!=sha(root/'sources'/str(frame)/camera/'receipt.json'):raise ValueError('Current fragment source binding changed')
   with np.load(artifact)as z:qualified_component=z['qualified_component'].copy()
   if proof.get('qualified_component_archive_sha256'):
    if sha(artifact)!=proof['qualified_component_archive_sha256']:raise ValueError('Current fragment artifact changed')
   else:
    # Initial root-reviewed legacy receipts predate artifact hashes: verify
    # exact connected-component equality against the hashed current NN output.
    with np.load(current_candidate_path)as z:nn=z['masks'][proof['person_instance']]
    _,labels=cv2.connectedComponents(nn.astype('uint8'),connectivity=8)
    if not np.array_equal(qualified_component,labels==proof['component']):raise ValueError('Legacy component differs from exact current NN pixels')
   qualified_actor|=small(qualified_component)
   qualified_instances[proof['person_instance']]=proof['actor'];fragment_proofs.append(dict(receipt_path=str(proofpath),receipt_sha256=sha(proofpath),person_instance=proof['person_instance'],component=proof['component']))
  path,meta=current_candidate_path,current_candidate_meta;selected=meta.get('selected',{a:record['identity'][a]['instance']for a in ['black','grey']});people=[];unknown=[]
  previous=root/'frames'/f'{frame-5:06d}'/'crew_guard'/f'{camera}_people.json';old=json.loads(previous.read_text())['people']if previous.exists()else[]
  old_masks=[];claimed_ids=set()
  if old:
   prevpath,_=candidates(root,frame-5,camera)
   with np.load(prevpath) as prevz:old_masks=[small(m)for m in prevz['masks']]
   previous_native=root/'frames'/f'{frame-5:06d}'/'native_masks'/f'{camera}.npz'
   if previous_native.exists():
    with np.load(previous_native)as previous_arrays:
     if'qualified_actor_rgb'in previous_arrays.files:
      previous_hair=small(previous_arrays['qualified_actor_rgb'])
      for oldrow in old:
       if oldrow['status'].startswith(('verified_other_person','verified_crew')):old_masks[oldrow['instance']]&=~previous_hair
   qualified_previous=root/'frames'/f'{frame-5:06d}'/'crew_inputs'/(camera+'_bald_helper.npz')
   if qualified_previous.exists():
    with np.load(qualified_previous)as z:
     x0,y0,x1,y1=z['crop_xyxy'];qm=np.zeros((4320,8192),bool);qm[y0:y1,x0:x1]=z['mask'];qualified_small=small(qm)
    for oldrow in old:
     if oldrow['stable_id']==camera+':bald_helper':old_masks[oldrow['instance']]=qualified_small
  current_crew=np.zeros((270,512),bool);crewinput=root/'frames'/f'{frame:06d}'/'crew_inputs'/(camera+'_bald_helper.npz')
  if crewinput.exists():
   with np.load(crewinput)as z:
    x0,y0,x1,y1=z['crop_xyxy'];full=np.zeros((4320,8192),bool);full[y0:y1,x0:x1]=z['mask'];current_crew=small(full)

  with np.load(path) as z:
   masks=z['masks'].copy();raw_person_hair_overlap={}
   if qualified_actor_rgb.any():
    for instance in range(len(masks)):
     if instance not in selected.values():raw_person_hair_overlap[instance]=int((masks[instance]&qualified_actor_rgb).sum());masks[instance]&=~qualified_actor_rgb
   for k,m in enumerate(masks):
    s=small(m);area=int(s.sum());overlap=float((s&union).sum()/max(area,1));novel=s&~safe;unqualified=novel&~veto&~qualified_actor;novel_n=int(novel.sum());remaining=int(unqualified.sum());actor=next((a for a,v in selected.items()if v==k),None)
    if actor:status='fighter_'+actor;stable=status
    elif overlap<.1:status='verified_other_person';stable=f'{camera}:crew_first{frame}_instance{k}'
    elif remaining>16 and remaining/max(area,1)>.05:status='UNKNOWN_person_overlap';stable=f'{camera}:unresolved_instance{k}';unknown.append(k)
    else:status='same_actor_fragment_or_qualified_crew_overlap';stable=f'{camera}:actor_fragment{k}'
    if k in qualified_instances:
     if k in unknown:unknown.remove(k)
     status='qualified_current_'+qualified_instances[k]+'_actor_fragment';stable='fighter_'+qualified_instances[k]
    previous_match=None
    if old_masks and actor is None:
     scores=np.asarray([(s&m).sum()/max((s|m).sum(),1)for m in old_masks]);order=np.argsort(scores)[::-1]
     for oldk in order:
      oldrow=next((v for v in old if v['instance']==int(oldk)),None)
      if oldrow and scores[oldk]>=.15 and oldrow['stable_id'] not in claimed_ids and not oldrow['status'].startswith('fighter_'):
       stable=oldrow['stable_id'];claimed_ids.add(stable);previous_match=dict(instance=int(oldk),IoU=float(scores[oldk]),status=oldrow['status']);break
    crew_coverage=float((s&current_crew).sum()/max(int(current_crew.sum()),1));crew_iou=float((s&current_crew).sum()/max(int((s|current_crew).sum()),1))
    if actor is None and crew_coverage>.7 and crew_iou>.1:
     status='verified_crew_with_qualified_current_roi';stable=camera+':bald_helper'
    elif actor is None and k not in qualified_instances and previous_match and previous_match['status'].startswith(('verified_crew','verified_other_person')) and int((s&union&~veto&~qualified_actor).sum())>16:
     status='NEEDS_CURRENT_NATIVE_CREW_ROI';unknown.append(k)if k not in unknown else None
    ys,xs=np.where(s);bbox=None if not len(xs)else[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    people.append(dict(instance=k,stable_id=stable,status=status,previous_identity_match=previous_match,current_qualified_crew_coverage=crew_coverage,current_qualified_crew_IoU=crew_iou,area_coarse=area,overlap_current_actor_fraction=overlap,novel_beyond128_native_prior_pixels=novel_n*256,unqualified_novel_pixels=remaining*256,bbox_coarse=bbox))
  crew_checks=[];flow_proof=None;flow_prediction=None
  with np.load(native)as native_arrays:current_native_union=native_arrays['black']|native_arrays['grey'];current_native_veto=native_arrays['source_ineligible'].copy()
  qualified_crew_native=np.zeros_like(current_native_union)
  qualified_crew_record=record.get('crew_guard',{})
  if crewinput.exists()and qualified_crew_record.get('status')=='qualified_current_frame_visible_crew_veto':
   qp=root/'frames'/f'{frame:06d}'/qualified_crew_record['receipt_path']
   if sha(qp)!=qualified_crew_record['receipt_sha256']:raise ValueError('Fresh NN crew qualification changed')
   with np.load(crewinput)as crew_arrays:
    qx0,qy0,qx1,qy1=crew_arrays['crop_xyxy'];qualified_crew_native[qy0:qy1,qx0:qx1]=crew_arrays['mask']
  crew_rows=[person for person in people if person['status'].startswith(('verified_other_person','verified_crew','NEEDS_CURRENT_NATIVE_CREW'))]
  if crew_rows:
   actor_ids=[v for v in selected.values()if v is not None];pre_actor=masks[actor_ids].any(0)if actor_ids else np.zeros((4320,8192),bool)
   pre_actor&=~qualified_crew_native  # Only fresh source-qualified NN crew ROI overrides a bad actor proposal; generic veto never does.
   if any(int((small(masks[person['instance']])&safe).sum())>8 for person in crew_rows):flow_prediction,flow_proof=previous_actor_prediction(root,frame,camera)
   for person in crew_rows:
    previous_match=person.get('previous_identity_match');previous_mask=old_masks[previous_match['instance']]if previous_match and old_masks else np.zeros((270,512),bool);current_nn_core,core_proof=qualified_current_crew_core(masks[person['instance']],previous_mask,previous_match)
    duplicate_proofs=[];this_mask=small(masks[person['instance']])
    for other_person in crew_rows:
     if other_person['instance']==person['instance']:continue
     other_mask=small(masks[other_person['instance']]);intersection=int((this_mask&other_mask).sum());iou=intersection/max(int((this_mask|other_mask).sum()),1);coverage=intersection/max(min(int(this_mask.sum()),int(other_mask.sum())),1)
     is_near_full_duplicate=iou>=.9 and coverage>=.98
     is_hair_disambiguated_head_subset=qualified_actor_rgb.any()and raw_person_hair_overlap.get(person['instance'],0)>0 and intersection/max(int(this_mask.sum()),1)>=.98
     if not(is_near_full_duplicate or is_hair_disambiguated_head_subset):continue
     other_match=other_person.get('previous_identity_match')
     if not other_match or not old_masks:continue
     other_core,other_proof=qualified_current_crew_core(masks[other_person['instance']],old_masks[other_match['instance']],other_match)
     if other_proof['qualified']:
      if not is_near_full_duplicate:
       yy,xx=np.where(this_mask);head=this_mask&(np.indices(this_mask.shape)[0]<=np.percentile(yy,20));other_core512=cv2.resize(other_core.astype('uint8'),(512,270),interpolation=cv2.INTER_NEAREST)>0
       if int((head&other_core512).sum())/max(int(head.sum()),1)<.8:continue
      current_nn_core|=other_core&cv2.resize(this_mask.astype('uint8'),(1024,540),interpolation=cv2.INTER_NEAREST).astype(bool);duplicate_proofs.append(dict(current_instance=other_person['instance'],current_IoU=iou,current_smaller_mask_coverage=coverage,qualified_actor_hair_disambiguated_head_subset=bool(is_hair_disambiguated_head_subset),qualified_core=other_proof));person['stable_id']=min(person['stable_id'],other_person['stable_id'])
    core_proof['near_identical_current_CREW_duplicates']=duplicate_proofs
    components=crew_component_conflicts(masks[person['instance']],pre_actor,flow_prediction,current_actor_union=current_native_union,effective_veto=current_native_veto,qualified_crew_roi=qualified_crew_native,qualified_current_nn_crew_core=current_nn_core);crew_checks.append(dict(instance=person['instance'],components=components,current_NN_crew_head_connected_core=core_proof))
    if any(c['independent_actor_conflict']for c in components):
     if person['instance']not in unknown:unknown.append(person['instance'])
     person['status']='UNKNOWN_CREW_COMPONENT_CONFLICT_WITH_INDEPENDENT_ACTOR_SUPPORT'
  residual=union&~safe&~veto&~qualified_actor;residual_n=int(residual.sum());large_new=residual_n>32 and residual_n/max(int(union.sum()),1)>.03
  report=dict(independent_crew_component_checks=crew_checks,flowed_previous_actor_evidence=flow_proof,frame=frame,camera=camera,status='unknown'if unknown or large_new else'pass',unknown_person_instances=unknown,actor_novel_unqualified_pixels=residual_n*256,actor_novelty_trigger=large_new,people=people,current_candidate_archive_sha256=sha(path),current_native_archive_sha256=sha(native),prior_actor_source_sha256=prior_sha,qualified_current_actor_fragment_proofs=fragment_proofs,policy='Previous labels guide identity/novelty only; no prior bitmap subtracted. New current-frame person overlap outside128native identity allowance must be explained by current verifiedcrew veto; otherwise publication UNKNOWN')
  write(out/(camera+'_people.json'),report);reports.append(report);print(camera,report['status'],unknown,residual_n,flush=True)
 result=dict(frame=frame,cameras=len(reports),status='pass'if all(r['status']=='pass'for r in reports)else'unknown',reports=[dict(camera=r['camera'],status=r['status'],unknown=r['unknown_person_instances'],actor_novelty=r['actor_novelty_trigger'])for r in reports],no_NeRF_state_loaded=True,diagnostic_partial=diagnostic_partial)
 write(out/'plan_receipt.json',result)
 if len(reports)==66 and result['status']=='pass'and not diagnostic_partial:write(out/'identity_novelty_pass.json',result)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--cameras',nargs='+',required=True);p.add_argument('--diagnostic-partial',action='store_true');a=p.parse_args();run(a.root,a.frame,a.cameras,a.diagnostic_partial)
