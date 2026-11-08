"""Current predicted components may extend semantic evidence; flow never emits FG."""
from pathlib import Path
import sys,json,time,itertools
import numpy as np,cv2
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from crew_guard_plan_v53 import candidates
import marshawn_union_tracking_proposals_v1 as tracker
from marshawn_current_parts_v96 import crew_core

def decide(semantic_fraction,residual_flow_fraction,crew_overlap,crew_available,component_IoU,RAW_residual_deep32_pixels,residual_pixels):
 semantic=residual_pixels>0 and crew_available and semantic_fraction>=.6 and residual_flow_fraction>=.6 and crew_overlap==0
 agreement=residual_pixels>0 and crew_available and component_IoU>=.9 and residual_flow_fraction>=.3 and RAW_residual_deep32_pixels>=128 and crew_overlap==0
 return dict(semantic_component_path=bool(semantic),component_agreement_path=bool(agreement),accepted=bool(semantic or agreement))

def decide_containment(containment,residual_flow_fraction,crew_overlap,crew_available,RAW_residual_deep32_pixels,residual_pixels,qualified_prior_union):
 return bool(residual_pixels>0 and qualified_prior_union and containment>=.99 and residual_flow_fraction>=.8 and RAW_residual_deep32_pixels>=128 and crew_available and crew_overlap==0)

def qualified_prior_union(frame,cam,guide):
 # Validate the ACTOR union that generated reliable flow, never an UNKNOWN/ROI proxy.
 proof=dict(qualified=False,UNKNOWN_or_envelope_NEVER_actor_support=True)
 try:
  previous=frame-5;f=V/'frames'/f'{previous:06d}';p=f/'native_masks'/(cam+'.npz');marker=f/'native_semantic_ready.json';m=__import__('json').loads(marker.read_text());g=__import__('json').loads(guide.with_suffix('.json').read_text());b=g['binding']
  assert m['native_masks'][cam]['archive_sha256']==sha(p)==b['previous_native_sha256']
  assert b['previous_native_metadata_sha256']==sha(p.with_suffix('.json'))and b['previous']==previous and b['frame']==frame and b['camera']==cam
  assert b['code_sha256']==sha(tracker.__file__)=='065a3ef88a6a3b72a904b4d1764761b7f7b65e7e64f950c5b68c8484e75f6f6f'
  assert b['current_RAW']['receipt_sha256']==sha(V/'sources'/str(frame)/cam/'receipt.json')and b['previous_RAW']['receipt_sha256']==sha(V/'sources'/str(previous)/cam/'receipt.json')
  assert sha(guide)==g['archive_sha256']
  with np.load(p)as z:expected=tracker.pool((z['black']|z['grey'])&~z['source_ineligible'])
  with np.load(guide)as z:
   assert np.array_equal(expected,z['complete_union'])and array_sha(z['complete_union'])==g['array_sha256']['complete_union']and array_sha(z['flowed_complete_union'])==g['array_sha256']['flowed_complete_union']
  proof.update(qualified=True,previous_source_marker_sha256=sha(marker),previous_actor_native_sha256=sha(p),guidance_archive_sha256=sha(guide),complete_actor_union_sha256=array_sha(expected),flowed_actor_union_sha256=g['array_sha256']['flowed_complete_union'],current_RAW_receipt_sha256=b['current_RAW']['receipt_sha256'],ownership_uncertain_actor_union_retained=True)
 except(Exception)as error:proof['reason']=str(error)
 return proof

def propose(frame,cam,semantic_archives,output,current_native=None):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);f=V/'frames'/f'{frame:06d}';native=Path(current_native)if current_native else f/'native_masks'/(cam+'.npz');cp,meta=candidates(V,frame,cam);report=tracker.propose(V,frame,cam,V/'operations/component_closure_v117/proposals'/f'frame{frame:06d}',current_native=native);guide=V/'operations/component_closure_v117/proposals'/f'frame{frame:06d}'/(cam+'_channels.npz')
 with np.load(cp)as z:masks=z['masks']
 selected=[i for i in meta['selected'].values()if i is not None];RAW=masks[selected].any(0);deep=cv2.distanceTransform(RAW.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)>32
 with np.load(guide)as z:flow=cv2.resize(z['flowed_complete_union'].astype('uint8'),(8192,4320),interpolation=cv2.INTER_NEAREST)>0
 prior=qualified_prior_union(frame,cam,guide)
 with np.load(native)as z:oldA=z['black']|z['grey']
 S=np.zeros_like(oldA);semantic=[]
 for p in semantic_archives:
  p=Path(p)
  with np.load(p)as z:S|=z['qualified_new_union']
  semantic.append(dict(path=str(p),sha256=sha(p)))
 crew,crewproof=crew_core(frame,cam,masks);areas=masks.sum((1,2));out=np.zeros_like(oldA);records=[]
 for row in report['records']:
  if row['native_pixels']<256:continue
  i=row['instance'];x0,y0,x1,y1=row['bbox_native_xyxy'];_,lab=cv2.connectedComponents(masks[i].astype('uint8'),8);part=lab[y0:y1,x0:x1]==row['native_component'];area=int(part.sum());sf=float((part&S[y0:y1,x0:x1]).sum()/area);R=part&~oldA[y0:y1,x0:x1];rn=int(R.sum());ff=float((R&flow[y0:y1,x0:x1]).sum()/max(rn,1));dc=int((R&deep[y0:y1,x0:x1]).sum());cc=int((part&crew[y0:y1,x0:x1]).sum());best=None
  for j in range(len(masks)):
   if i==j:continue
   inter=int((part&masks[j,y0:y1,x0:x1]).sum());iou=inter/max(1,area+int(areas[j])-inter)
   if best is None or iou>best['IoU']:best=dict(instance=j,IoU=iou,intersection=inter,partner_native_pixels=int(areas[j]))
  decision=decide(sf,ff,cc,bool(crewproof),best['IoU']if best else 0,dc,rn);semantic_path=decision['semantic_component_path'];agreement_path=decision['component_agreement_path'];accepted=decision['accepted']
  peer_selected=[j for j in selected if j!=i]
  peer_RAW=masks[peer_selected,y0:y1,x0:x1].any(0)if peer_selected else np.zeros_like(part)
  containment=float((part&peer_RAW).sum()/max(area,1));containment_path=decide_containment(containment,ff,cc,bool(crewproof),int((R&deep[y0:y1,x0:x1]&peer_RAW).sum()),rn,prior['qualified'])
  if accepted:out[y0:y1,x0:x1]|=part
  elif containment_path:out[y0:y1,x0:x1]|=part&peer_RAW
  accepted=bool(accepted or containment_path)
  records.append(dict(instance=i,native_component=row['native_component'],native_pixels=area,residual_new_union_pixels=rn,reliable_residual_flow_pixels=int((R&flow[y0:y1,x0:x1]).sum()),independent_CURRENT_CREW_context_available=bool(crewproof),bbox=row['bbox_native_xyxy'],fresh_qualified_semantic_fraction=sf,reliable_complete_union_flow_fraction=ff,CURRENT_RAW_selected_actor_deep32_pixels=dc,independent_CURRENT_CREW_core_overlap=cc,component_agreement=best,semantic_component_path=semantic_path,component_agreement_path=agreement_path,CURRENT_selected_peer_RAW_containment=containment,CURRENT_selected_peer_instance_ids=peer_selected,RAW_containment_path=containment_path,containment_emits_only_CURRENT_NN_intersection_RAW=True,accepted=accepted))
 archive=output/(cam+'_current_component_closure.npz');np.savez_compressed(archive,qualified_current_component_union=out);proof=dict(kind='CURRENT_NN_COMPONENT_CLOSURE_WITH_STRICT_RAW_CONTAINMENT_V117',frame=frame,camera=cam,records=records,source_receipt_sha256=sha(V/'sources'/str(frame)/cam/'receipt.json'),current_candidate_archive_sha256=sha(cp),current_native_sha256=sha(native),CURRENT_CREW_qualification_native_sha256=sha(f/'native_masks'/(cam+'.npz')),guidance_archive_sha256=sha(guide),semantic_archives=semantic,crew_proof=crewproof,qualified_prior_actor_union=prior,qualified_archive=str(archive),qualified_archive_sha256=sha(archive),qualified_union_sha256=array_sha(out),qualified_native_pixels=int(out.sum()),no_source_or_field_modified=True,previous_pixels_only_identity_and_ROI=True,duplicate_current_person_instances_NOT_independent_semantic_models=True,created=time.time());write(output/(cam+'_proof.json'),proof);return proof
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--frame',type=int,required=True);p.add_argument('--camera',required=True);p.add_argument('--semantic',nargs='*',default=[]);p.add_argument('--current-native',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=propose(a.frame,a.camera,a.semantic,a.output,a.current_native);print(a.camera,r['qualified_native_pixels'],[(x['instance'],x['native_component'],x['semantic_component_path'],x['component_agreement_path'])for x in r['records']if x['accepted']],flush=True)
