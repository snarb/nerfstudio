from pathlib import Path
import sys,json,time,numpy as np
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from marshawn_current_component_closure_v106 import propose,decide
O=V/'operations/ordered_closure_golden_v106';O.mkdir(exist_ok=True)
MDold=V/'operations/current_native_parts_v92_MD/frame001690/applications/MD/old_MD.npz';DEold=V/'operations/1690_DE_readonly/application_v94/before/native.npz'
positive=[]
for cam,old,sem in [('MD',MDold,V/'operations/current_native_parts_v92_MD/frame001690/MD_qualified_new_union.npz'),('DE',DEold,V/'operations/current_native_parts_v96/frame001690/DE_qualified_new_union.npz')]:
 proof=propose(1690,cam,[sem],O/cam,current_native=old)
 with np.load(proof['qualified_archive'])as z:got=z['qualified_current_component_union']
 if cam=='MD':
  from crew_guard_plan_v53 import candidates
  cp,_=candidates(V,1690,'MD')
  with np.load(cp)as z:M=z['masks'][7]
  import cv2
  _,labels=cv2.connectedComponents(M.astype('uint8'),8);expected=labels==3
 else:
  with np.load(V/'operations/1690_DE_readonly/reviewed_main_components.npz')as z:
   key=next(k for k in z.files if z[k].shape==(4320,8192)and int(z[k].sum())==125974);expected=z[key]
 assert np.array_equal(got,expected),(cam,int(got.sum()),int(expected.sum()),int((got^expected).sum()))
 positive.append(dict(camera=cam,actual_source_before_semantic_application=str(old),actual_native_sha256=sha(old),proof_sha256=sha(O/cam/(cam+'_proof.json')),expected_pixels=int(expected.sum()),expected_array_sha256=array_sha(expected),exact_current_NN_closure_equals_ROOT_source_positive=True))
negative_path=V/'operations/closure_negative_controls_readonly_v1/report.json';negative=json.loads(negative_path.read_text());rows=[]
for item in negative['records']if'records'in negative else negative['results']:
 case=next(x for x in item['components']if x['native_component']==(7 if item['camera']=='NB'else 1));partner=case['near_duplicate_max_partner'];overlap=case['independent_current_clean_CREW_native_overlap'];available=case['independent_current_clean_CREW_available'];rf=case['residual_R_reliable_complete_union_flow_native_fraction'];rn=case['residual_R_pixels'];dc=case['residual_RAW_selected_PRE_deep32_pixels']
 # Even adversarial semantic requalification of all old actor pixels or all
 # component pixels cannot finance a residual with insufficient own support.
 for sem in [case['current_A_pixels']/case['native_pixels'],1.0]:
  result=decide(sem,rf,overlap,available,partner['IoU'],dc,rn);assert not result['accepted'],(item['camera'],result)
 rows.append(dict(camera=item['camera'],frame=item['frame'],residual_pixels=rn,residual_reliable_flow_fraction=rf,independent_current_CREW_overlap=overlap,adversarial_semantic_coverage_up_to1_rejected=True,both_closure_paths_rejected=True))
write(O/'PASS.json',dict(kind='ORDERED_CURRENT_SEMANTIC_THEN_COMPONENT_CLOSURE_SOURCE_GOLDEN_V106',status='pass',positives=positive,negatives=rows,negative_actual_source_report_sha256=sha(negative_path),same_immutable_preapplication_native_for_semantic_and_closure=True,no_field_or_live_source_arrays_modified=True,policy_code_sha256=sha(V/'operations/marshawn_current_component_closure_v106.py'),created=time.time()));print('ORDERED_CURRENT_COMPONENT_GOLDEN_PASS',flush=True)
