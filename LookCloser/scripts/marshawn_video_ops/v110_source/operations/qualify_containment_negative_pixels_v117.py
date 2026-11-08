from pathlib import Path
import json,sys,numpy as np,cv2,time
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations';sys.path[:0]=[str(O),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_current_component_closure_v117 import decide_containment
r=json.loads((O/'closure_negative_controls_readonly_v1/report.json').read_text());rows=[]
for item in r.get('records',r.get('results',[])):
 cp=Path(item['candidate_archive']);assert sha(cp)==item['candidate_archive_sha256'];meta=json.loads(cp.with_suffix('.json').read_text());selected=[i for i in meta['selected'].values()if i is not None and i!=item['instance']]
 with np.load(cp)as z:masks=z['masks'];M=masks[item['instance']];peer=masks[selected].any(0)
 k=7 if item['camera']=='NB'else 1;_,labels=cv2.connectedComponents(M.astype('uint8'),8);C=labels==k;case=next(x for x in item['components']if x['native_component']==k);contain=float((C&peer).sum()/max(int(C.sum()),1));assert int(C.sum())==case['native_pixels'];decision=decide_containment(contain,case['residual_R_reliable_complete_union_flow_native_fraction'],case['independent_current_clean_CREW_native_overlap'],case['independent_current_clean_CREW_available'],case['residual_RAW_selected_PRE_deep32_pixels'],case['residual_R_pixels'],True);assert not decision
 rows.append(dict(camera=item['camera'],frame=item['frame'],candidate_archive_sha256=sha(cp),candidate_metadata_sha256=sha(cp.with_suffix('.json')),instance=item['instance'],component=k,native_component_pixels=int(C.sum()),distinct_selected_target_ids=selected,actual_component_peer_RAW_containment=contain,rejected=True,actual_metrics_source_sha256=sha(O/'closure_negative_controls_readonly_v1/report.json')))
write(O/'containment_v117_qualification/actual_negative_component_containment.json',dict(status='pass',records=rows,no_source_or_model_mutation=True,created=time.time()));print(rows)
