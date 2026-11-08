"""Read-only ALL-person identity/novel-overlap guard. UNKNOWN blocks publication."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from prepare_marshawn_tuning import sha,write
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
  native=root/'frames'/f'{frame:06d}'/'native_masks'/f'{camera}.npz';record=json.loads(native.with_suffix('.json').read_text())
  if record['archive_sha256']!=sha(native):raise ValueError('Native actor masks changed '+camera)
  with np.load(native) as z:current={a:small(z[a])for a in ['black','grey']};veto=small(z['source_ineligible'])
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
   masks=z['masks']
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
  residual=union&~safe&~veto&~qualified_actor;residual_n=int(residual.sum());large_new=residual_n>32 and residual_n/max(int(union.sum()),1)>.03
  report=dict(frame=frame,camera=camera,status='unknown'if unknown or large_new else'pass',unknown_person_instances=unknown,actor_novel_unqualified_pixels=residual_n*256,actor_novelty_trigger=large_new,people=people,current_candidate_archive_sha256=sha(path),current_native_archive_sha256=sha(native),prior_actor_source_sha256=prior_sha,qualified_current_actor_fragment_proofs=fragment_proofs,policy='Previous labels guide identity/novelty only; no prior bitmap subtracted. New current-frame person overlap outside128native identity allowance must be explained by current verifiedcrew veto; otherwise publication UNKNOWN')
  write(out/(camera+'_people.json'),report);reports.append(report);print(camera,report['status'],unknown,residual_n,flush=True)
 result=dict(frame=frame,cameras=len(reports),status='pass'if all(r['status']=='pass'for r in reports)else'unknown',reports=[dict(camera=r['camera'],status=r['status'],unknown=r['unknown_person_instances'],actor_novelty=r['actor_novelty_trigger'])for r in reports],no_NeRF_state_loaded=True,diagnostic_partial=diagnostic_partial)
 write(out/'plan_receipt.json',result)
 if len(reports)==66 and result['status']=='pass'and not diagnostic_partial:write(out/'identity_novelty_pass.json',result)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--cameras',nargs='+',required=True);p.add_argument('--diagnostic-partial',action='store_true');a=p.parse_args();run(a.root,a.frame,a.cameras,a.diagnostic_partial)
