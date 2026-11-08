"""Independent current/pre-veto and flowed actor evidence for CREW conflicts.

Flow is identity/visibility evidence only. It never supplies final foreground
pixels or erases a current source label. Material conflicts require fresh NN QA.
"""
from pathlib import Path
import json
import cv2,numpy as np
from PIL import Image
from prepare_marshawn_tuning import sha
SIZE=(1024,540)
def small(m):return cv2.resize(np.asarray(m,np.uint8),SIZE,interpolation=cv2.INTER_AREA)>0

def previous_actor_prediction(root,frame,camera):
 root=Path(root);prev=frame-5;p=root/'frames'/f'{prev:06d}'/'native_masks'/f'{camera}.npz'
 if not p.exists():return None,None
 with np.load(p)as z:actors=small(z['black']|z['grey'])
 actors=cv2.erode(actors.astype('uint8'),np.ones((3,3),np.uint8))>0
 images=[];receipts=[]
 for f in [prev,frame]:
  receipt=root/'sources'/str(f)/camera/'receipt.json'
  if not receipt.exists():return None,None
  source=json.loads(receipt.read_text());im=np.array(Image.open(source['raw_native_image']).convert('L').resize(SIZE,Image.Resampling.BILINEAR));images.append(im);receipts.append(sha(receipt))
 before,current=images;forward=cv2.calcOpticalFlowFarneback(before,current,None,.5,3,21,3,7,1.5,0);back=cv2.calcOpticalFlowFarneback(current,before,None,.5,3,21,3,7,1.5,0)
 yy,xx=np.indices(before.shape,dtype=np.float32);tx=xx+forward[...,0];ty=yy+forward[...,1];bx=cv2.remap(back[...,0],tx,ty,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);by=cv2.remap(back[...,1],tx,ty,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);fb=np.hypot(forward[...,0]+bx,forward[...,1]+by);photo=cv2.remap(current,tx,ty,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);good=actors&(fb<1.5)&(np.abs(photo.astype('float32')-before.astype('float32'))<35)&(tx>=0)&(tx<SIZE[0])&(ty>=0)&(ty<SIZE[1]);prediction=np.zeros_like(actors);prediction[np.rint(ty[good]).astype(int).clip(0,SIZE[1]-1),np.rint(tx[good]).astype(int).clip(0,SIZE[0]-1)]=True
 return prediction,dict(previous_frame=prev,previous_native_sha256=sha(p),source_receipt_sha256=receipts,flow_shape=list(before.shape),FB_max_pixels=1.5,photometric_max_uint8=35,valid_tracked_actor_core_pixels=int(good.sum()),previous_labels_are_guard_only_not_final_pixels=True)

def crew_component_conflicts(person_mask,pre_actor_union,flowed_actor_union,current_actor_union=None,effective_veto=None,qualified_crew_roi=None,qualified_current_nn_crew_core=None):
 crew=small(person_mask);current=small(current_actor_union)if current_actor_union is not None else np.zeros_like(crew);veto=small(effective_veto)if effective_veto is not None else np.zeros_like(crew);qualified=small(qualified_crew_roi)if qualified_crew_roi is not None else np.zeros_like(crew);nn_core=small(qualified_current_nn_crew_core)if qualified_current_nn_crew_core is not None else np.zeros_like(crew);pre=cv2.erode(small(pre_actor_union).astype('uint8'),np.ones((3,3),np.uint8))>0;n,labels,stats,_=cv2.connectedComponentsWithStats(crew.astype('uint8'),connectivity=8);rows=[]
 for k in range(1,n):
  if stats[k,4]<8:continue
  c=labels==k;current_n=int((c&pre).sum());previous=int((c&flowed_actor_union).sum())if flowed_actor_union is not None else 0;area=int(c.sum());raw_conflict=(current_n>=16 and current_n/max(area,1)>=.02)or(previous>=16 and previous/max(area,1)>=.02);evidence=c&(pre|(flowed_actor_union if flowed_actor_union is not None else False));actor_explained=evidence&current&~veto&(flowed_actor_union if flowed_actor_union is not None else False);crew_explained=evidence&qualified&veto;unresolved=(c&pre)|(c&(flowed_actor_union if flowed_actor_union is not None else False)&~nn_core&~crew_explained);unresolved_n=int(unresolved.sum());conflict=raw_conflict and unresolved_n>=16 and unresolved_n/max(area,1)>=.02
  rows.append(dict(component=k,pixels_coarse=area,PRE_veto_current_actor_core_overlap=current_n,current_NN_and_flow_actor_agreement_not_a_CREW_waiver=int(actor_explained.sum()),qualified_current_crew_ROI_explained=int(crew_explained.sum()),unresolved_independent_actor_conflict_pixels=unresolved_n,flowed_previous_actor_core_overlap=previous,independent_actor_conflict=conflict,flow_only_known_current_crew_core_warning=bool(previous>=16 and current_n==0 and not conflict),current_NN_crew_core_flow_explained=int((c&nn_core&(flowed_actor_union if flowed_actor_union is not None else False)).sum()),bbox_coarse=stats[k,:4].tolist()))
 return rows


def qualified_current_crew_core(person_mask,previous_crew_mask,identity_match):
 """Current NN head-connected core within established crew scope; guard only."""
 crew=small(person_mask)
 proof=dict(qualified=False,policy='FLOW-only warning; CURRENT PRE-veto actor-core conflict never waived',envelope_native_pixels=128)
 if not identity_match or not identity_match.get('status','').startswith(('verified_other_person','verified_crew'))or identity_match.get('IoU',0)<.25:return np.zeros_like(crew),proof
 previous=np.asarray(previous_crew_mask,bool)
 if previous.shape!=crew.shape:previous=cv2.resize(previous.astype('uint8'),SIZE,interpolation=cv2.INTER_NEAREST)>0
 envelope=cv2.dilate(previous.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)))>0
 yy,xx=np.where(crew)
 if not len(yy):return np.zeros_like(crew),proof
 head=crew&(np.indices(crew.shape)[0]<=np.percentile(yy,20));head_coverage=float((head&envelope).sum()/max(head.sum(),1));proof.update(known_clean_crew_IoU=identity_match['IoU'],current_upper_head_anchor_pixels=int(head.sum()),head_anchor_previous_crew_envelope_coverage=head_coverage)
 mutual_head_seed=head&cv2.erode(previous.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool);seed_pixels=int(mutual_head_seed.sum());proof['current_NN_head_previous_clean_CREW_core_seed_pixels']=seed_pixels
 if seed_pixels<max(8,int(head.sum()*.1)):return np.zeros_like(crew),proof
 n,labels,stats,_=cv2.connectedComponentsWithStats(crew.astype('uint8'),connectivity=8);out=np.zeros_like(crew);accepted=[]
 for component in range(1,n):
  c=labels==component;anchor=int((c&mutual_head_seed).sum());prior=int((c&previous).sum())
  if anchor>=max(8,int(mutual_head_seed.sum()*.1))and prior/max(int(c.sum()),1)>=.1:out|=c&envelope;accepted.append(component)
 proof.update(qualified=bool(accepted),head_connected_components=accepted,qualified_current_core_pixels=int(out.sum()),disconnected_or_lower_new_components_excluded=True,previous_pixels_never_final_foreground=True)
 return out,proof
