"""Advance current-source crew identity references without modifying actor pixels.

Only a fresh current person already excluded by the canonical source veto may
advance automatically. Any overlap with either fighter remains UNKNOWN.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from crew_guard_plan import candidates,small
from prepare_marshawn_tuning import sha,write

def advance(root,frame):
 f=root/'frames'/f'{frame:06d}';previous=f.parent/f'{frame-5:06d}'/'crew_inputs';out=f/'crew_inputs';out.mkdir(exist_ok=True);reports=[]
 for old in sorted(previous.glob('*_bald_helper.npz')):
  camera=old.name.split('_')[0];target=out/old.name
  if target.exists():continue
  with np.load(old)as z:
   x0,y0,x1,y1=z['crop_xyxy'];ref=np.zeros((4320,8192),bool);ref[y0:y1,x0:x1]=z['mask'];ref=small(ref)
  archive,meta=candidates(root,frame,camera)
  native=f/'native_masks'/f'{camera}.npz';record=json.loads(native.with_suffix('.json').read_text());assert sha(native)==record['archive_sha256']
  with np.load(native)as z:actors=z['black']|z['grey'];veto=z['source_ineligible']
  with np.load(archive)as z:
   scores=np.array([(ref&small(m)).sum()/max((ref|small(m)).sum(),1)for m in z['masks']]);instance=int(scores.argmax());mask=z['masks'][instance].astype(bool)
  overlap=int((mask&actors).sum());coverage=float((mask&veto).sum()/max(mask.sum(),1));qualified=bool(scores[instance]>=.25 and overlap==0 and coverage>=.999)
  receipt=dict(frame=frame,camera=camera,previous_identity_reference_sha256=sha(old),current_candidate_archive_sha256=sha(archive),current_native_sha256=sha(native),source_receipt_sha256=sha(root/'sources'/str(frame)/camera/'receipt.json'),instance=instance,identity_IoU=float(scores[instance]),current_actor_overlap_native=overlap,canonical_veto_coverage=coverage,status='CURRENT_CREW_ALREADY_FULLY_VETOED'if qualified else'UNKNOWN_CURRENT_CREW_REQUIRES_NATIVE_QUALIFICATION',no_actor_or_veto_pixels_changed=True,previous_pixels_used_only_for_identity=True,created=time.time())
  if qualified:
   y,x=np.where(mask);bbox=np.array([x.min(),y.min(),x.max()+1,y.max()+1]);x0,y0,x1,y1=bbox;np.savez_compressed(target,mask=mask[y0:y1,x0:x1],crop_xyxy=bbox)
   receipt['current_identity_reference_sha256']=sha(target)
  write(out/(camera+'_automatic_reference_receipt.json'),receipt);reports.append(receipt)
 write(out/'reference_advance_receipt.json',dict(frame=frame,records=reports,created=time.time()))
 return reports
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',type=int,required=True);a=p.parse_args();print(json.dumps(advance(a.root,a.frame)))
