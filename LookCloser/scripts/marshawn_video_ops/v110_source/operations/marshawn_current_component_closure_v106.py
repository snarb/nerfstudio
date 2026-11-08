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

def propose(frame,cam,semantic_archives,output,current_native=None):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);f=V/'frames'/f'{frame:06d}';native=Path(current_native)if current_native else f/'native_masks'/(cam+'.npz');cp,meta=candidates(V,frame,cam);report=tracker.propose(V,frame,cam,V/'operations/component_closure_v101/proposals'/f'frame{frame:06d}',current_native=native);guide=V/'operations/component_closure_v101/proposals'/f'frame{frame:06d}'/(cam+'_channels.npz')
 with np.load(cp)as z:masks=z['masks']
 selected=[i for i in meta['selected'].values()if i is not None];RAW=masks[selected].any(0);deep=cv2.distanceTransform(RAW.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)>32
 with np.load(guide)as z:flow=cv2.resize(z['flowed_complete_union'].astype('uint8'),(8192,4320),interpolation=cv2.INTER_NEAREST)>0
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
  if accepted:out[y0:y1,x0:x1]|=part
  records.append(dict(instance=i,native_component=row['native_component'],native_pixels=area,residual_new_union_pixels=rn,reliable_residual_flow_pixels=int((R&flow[y0:y1,x0:x1]).sum()),independent_CURRENT_CREW_context_available=bool(crewproof),bbox=row['bbox_native_xyxy'],fresh_qualified_semantic_fraction=sf,reliable_complete_union_flow_fraction=ff,CURRENT_RAW_selected_actor_deep32_pixels=dc,independent_CURRENT_CREW_core_overlap=cc,component_agreement=best,semantic_component_path=semantic_path,component_agreement_path=agreement_path,accepted=accepted))
 archive=output/(cam+'_current_component_closure.npz');np.savez_compressed(archive,qualified_current_component_union=out);proof=dict(kind='CURRENT_NN_COMPONENT_CLOSURE_V106_ORDERED_PREAPPLICATION',frame=frame,camera=cam,records=records,source_receipt_sha256=sha(V/'sources'/str(frame)/cam/'receipt.json'),current_candidate_archive_sha256=sha(cp),current_native_sha256=sha(native),CURRENT_CREW_qualification_native_sha256=sha(f/'native_masks'/(cam+'.npz')),guidance_archive_sha256=sha(guide),semantic_archives=semantic,crew_proof=crewproof,qualified_archive=str(archive),qualified_archive_sha256=sha(archive),qualified_union_sha256=array_sha(out),qualified_native_pixels=int(out.sum()),no_source_or_field_modified=True,previous_pixels_only_identity_and_ROI=True,duplicate_current_person_instances_NOT_independent_semantic_models=True,created=time.time());write(output/(cam+'_proof.json'),proof);return proof
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--frame',type=int,required=True);p.add_argument('--camera',required=True);p.add_argument('--semantic',nargs='*',default=[]);p.add_argument('--current-native',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=propose(a.frame,a.camera,a.semantic,a.output,a.current_native);print(a.camera,r['qualified_native_pixels'],[(x['instance'],x['native_component'],x['semantic_component_path'],x['component_agreement_path'])for x in r['records']if x['accepted']],flush=True)
