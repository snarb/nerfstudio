from pathlib import Path
import sys,json,time,ast
import numpy as np,cv2
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations';sys.path[:0]=[str(O),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from marshawn_current_component_closure_v118 import propose,decide,decide_containment,emitted_containment_evidence

def main():
 cv2.setNumThreads(1)
 # Existing scientific decision function is byte/AST-identical.
 def fun(p):return next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='decide')
 assert ast.dump(fun(O/'marshawn_current_component_closure_v106.py'))==ast.dump(fun(O/'marshawn_current_component_closure_v118.py'))
 Q=O/'containment_v118_qualification';Q.mkdir(exist_ok=True);started=time.time();positives=[]
 fixtures=[('EE',1700,O/'1700_EE2_readonly/application_v116/before/native.npz',[],8504,'ada19e6ca9759dd5e19fe95e40148ba23802b1da10ec42e5380f0b23ad549a93'),('MD',1690,O/'current_native_parts_v92_MD/frame001690/applications/MD/old_MD.npz',[O/'current_native_parts_v92_MD/frame001690/MD_qualified_new_union.npz'],11269,'99b9b5b8dd43140c16baf742e72ef6addc577a2f095c61522270d99c4408118c'),('DE',1690,O/'1690_DE_readonly/application_v94/before/native.npz',[O/'current_native_parts_v96/frame001690/DE_qualified_new_union.npz'],125974,'852a3758cd255406cdadef7f5fc5755aa6912730e7545e4ccd1da887fa841fd9')]
 for camera,frame,native,sem,pixels,expected in fixtures:
  before=sha(V/'frames'/f'{frame:06d}'/'native_masks'/(camera+'.npz'));r=propose(frame,camera,sem,Q/(camera+str(frame)),current_native=native)
  with np.load(r['qualified_archive'])as z:mask=z['qualified_current_component_union']
  assert int(mask.sum())==pixels and array_sha(mask)==expected,(camera,int(mask.sum()),array_sha(mask),[(x['instance'],x['native_component'],x['RAW_containment_path'])for x in r['records']if x['accepted']])
  assert sha(V/'frames'/f'{frame:06d}'/'native_masks'/(camera+'.npz'))==before
  positives.append(dict(camera=camera,frame=frame,preapplication_native_sha256=sha(native),proof_sha256=sha(Q/(camera+str(frame))/(camera+'_proof.json')),pixels=pixels,array_sha256=expected,exact=True,accepted=[dict(instance=x['instance'],component=x['native_component'],old_semantic=x['semantic_component_path'],old_duplicate=x['component_agreement_path'],new_containment=x['RAW_containment_path'])for x in r['records']if x['accepted']],qualified_prior_union=r['qualified_prior_actor_union']))
 # Actual frozen negative residuals, including adversarial semantic financing.
 negative_path=O/'closure_negative_controls_readonly_v1/report.json';negative=json.loads(negative_path.read_text());negatives=[]
 for item in negative.get('records',negative.get('results',[])):
  case=next(x for x in item['components']if x['native_component']==(7 if item['camera']=='NB'else 1));rf=case['residual_R_reliable_complete_union_flow_native_fraction'];cc=case['independent_current_clean_CREW_native_overlap'];available=case['independent_current_clean_CREW_available'];dc=case['residual_RAW_selected_PRE_deep32_pixels'];rn=case['residual_R_pixels']
  for semantic in [case['current_A_pixels']/case['native_pixels'],1.0]:assert not decide(semantic,rf,cc,available,case['near_duplicate_max_partner']['IoU'],dc,rn)['accepted']
  # Even granting impossible100% RAW containment cannot rescue these actual badR.
  assert not decide_containment(1.0,rf,cc,available,dc,rn,True)
  negatives.append(dict(camera=item['camera'],frame=item['frame'],actual_source_negative_report_sha256=sha(negative_path),residual_pixels=rn,residual_flow_fraction=rf,current_CREW_overlap=cc,old_and_new_paths_rejected_even_adversarial_semantics_and_containment=True))
 R=np.ones((100,100),bool);peer=np.ones_like(R);peer[0]=False;flow=np.zeros_like(R);flow[:80]=True;e=emitted_containment_evidence(R,peer,flow,np.ones_like(R));assert flow.sum()/R.sum()==.8 and e['reliable_flow_fraction']<.8;assert not decide_containment(.99,e['reliable_flow_fraction'],0,True,e['RAW_deep32_pixels'],e['pixels'],True),'Removed1percent cannot finance weaker emitted flow'
 assert not decide_containment(1.,1.,0,True,10000,10000,False),'UNKNOWN/envelope proxy cannot finance a foreground union'
 write(Q/'PASS.json',dict(status='pass',source_candidate_sha256=sha(O/'marshawn_current_component_closure_v118.py'),positives=positives,negatives=negatives,UNKNOWN_and_dilated_proxy_rejected=True,whole_point8_emitted_below_point8_rejected=True,old_decide_AST_exact=True,no_live_source_or_field_arrays_modified=True,activation_authorized=False,seconds=time.time()-started));print('CONTAINMENT118_PRIVATE_GOLDENS_PASS',flush=True)
if __name__=='__main__':main()
