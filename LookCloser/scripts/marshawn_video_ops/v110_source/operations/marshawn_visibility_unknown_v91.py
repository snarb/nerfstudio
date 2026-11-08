"""Explicit source visibility uncertainty; never relabel flow as ground truth."""
from pathlib import Path
import json,time
import numpy as np
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha

def resolve(root,frame,report,policy):
 root=Path(root);f=root/'frames'/f'{frame:06d}';camera=report['camera'];p=f/'native_masks'/(camera+'.npz');source=root/'sources'/str(frame)/camera/'receipt.json'
 assert report['frame']==frame and report['current_native_archive_sha256']==sha(p)
 if not policy.get('authorized')or not policy.get('PRE_current_actor_core_conflict_always_hard_fail'):raise ValueError('Visibility policy not authorized')
 with np.load(p)as z:
  actors=z['black']|z['grey'];veto=z['source_ineligible'];unknown=z['unknown'];H=z['qualified_actor_rgb']if'qualified_actor_rgb'in z.files else np.zeros_like(actors)
 from crew_guard_plan_v53 import candidates
 cp,meta=candidates(root,frame,camera);assert sha(cp)==report['current_candidate_archive_sha256']
 with np.load(cp)as z:people=z['masks']
 selected=meta.get('selected',{});selected_ids=[x for x in selected.values()if x is not None];native_PRE=people[selected_ids].any(0)if selected_ids else np.zeros_like(actors)
 original_PRE=native_PRE.copy();exception=f/'qualified_nonactor_inputs'/(camera+'_PRE78')
 if(exception/'proof.json').exists():
  q=json.loads((exception/'proof.json').read_text());assert q['frame']==frame and q['camera']==camera and q['source_receipt_sha256']==sha(source)and q['candidate_archive_sha256']==sha(cp)
  with np.load(exception/'mask.npz')as z:P=z['qualified_NONACTOR_current_RAW_PRE']
  assert sha(exception/'mask.npz')==q['mask_archive_sha256']and array_sha(P)==q['mask_array_sha256']and int(P.sum())==78
  assert not(P&actors).any()and not(P&H).any()and not(P&~original_PRE).any();native_PRE&=~P
 # Explicit current-manual UNRESOLVED proposal exception, not NONACTOR.
 # Current refined PRE and final actor/H are absent; physical UNKNOWN and all
 # source exclusions stay byte-identical. Generic current PRE hard fail remains.
 uncertain_exception=f/'qualified_unresolved_PRE_inputs'/(camera+'_person5')
 if(uncertain_exception/'proof.json').exists():
  q=json.loads((uncertain_exception/'proof.json').read_text());assert q['frame']==frame and q['camera']==camera and q['kind']=='ROOT_SCOPED_UNRESOLVED_RAW_SELECTED_PRE_GUARD_EXCEPTION'
  assert q['source_receipt_sha256']==sha(source)and q['candidate_archive_sha256']==sha(cp)and q['native_archive_sha256']==sha(p)
  with np.load(uncertain_exception/'mask.npz')as z:P=z['unresolved_RAW_selected_PRE']
  assert sha(uncertain_exception/'mask.npz')==q['mask_archive_sha256']and array_sha(P)==q['mask_array_sha256']and int(P.sum())==3511
  assert not(P&actors).any()and not(P&H).any()and not(P&~original_PRE).any()and veto[P].all()and unknown[P].all()
  up=f/json.loads(p.with_suffix('.json').read_text())['union_resolution']['receipt'];ur=json.loads(up.read_text());prepath=Path(ur['pre_overlap_archive']);assert sha(prepath)==q['refined_PRE_archive_sha256']
  with np.load(prepath)as z:refined=(z['pre_black']|z['pre_grey'])if'pre_black'in z.files else(z['black']|z['grey'])
  assert not(P&refined).any();assert sha(q['ROOT_approval_path'])==q['ROOT_approval_sha256'];assert q['projected_scope_W32_W64_zero_before_guard']and q['no_DATA_or_native_arrays_changed']
  native_PRE&=~P
 remaining=[];resolutions=[]
 for instance in report['unknown_person_instances']:
  person=next(x for x in report['people']if x['instance']==instance);check=next((x for x in report['independent_crew_component_checks']if x['instance']==instance),None);conflicts=[]if check is None else[c for c in check['components']if c['independent_actor_conflict']]
  # Novel actor/identity or actual current PRE conflict is NOT a flow fallback.
  allowed=person['status']=='UNKNOWN_CREW_COMPONENT_CONFLICT_WITH_INDEPENDENT_ACTOR_SUPPORT'and bool(conflicts)
  previous=person.get('previous_identity_match')or{};role=previous.get('status','');independent_CREW=role.startswith(('verified_other_person','verified_crew'))or person.get('current_qualified_crew_coverage',0)>.7
  known_actor_role=('actor_fragment'in role or role.startswith('fighter_')or instance in selected_ids)and not independent_CREW
  A=people[instance]&H
  if known_actor_role:A|=people[instance]&actors
  scope=people[instance]&~A;unlabeled=scope&~actors
  actor_overlap=int((scope&actors).sum());bad_veto=int((unlabeled&~veto).sum());bad_unknown=int((unlabeled&~unknown).sum());exact_PRE=int((unlabeled&native_PRE).sum())
  soft_RGB_branch=bad_veto>0 and known_actor_role and not independent_CREW
  allowed=(allowed or(person['status']=='UNKNOWN_person_overlap'and known_actor_role))and actor_overlap==0 and(bad_veto==0 or soft_RGB_branch)and bad_unknown==0 and exact_PRE==0 and bool(unlabeled.any())
  if not allowed:remaining.append(instance);continue
  out=f/'visibility_unknown_inputs';out.mkdir(exist_ok=True);artifact=out/f'{camera}_person{instance}.npz'
  np.savez_compressed(artifact,unresolved_current_visibility=unlabeled,known_actor_A=A)
  proof=dict(frame=frame,camera=camera,instance=instance,kind='CURRENT_NN_PRE0_FLOW_ONLY_UNRESOLVED_VISIBILITY',status='explicit_geometry_UNKNOWN_source_ineligible',source_receipt_sha256=sha(source),candidate_archive_sha256=sha(cp),native_archive_sha256=sha(p),scope_mask_sha256=array_sha(unlabeled),scope_pixels=int(unlabeled.sum()),qualified_CURRENT_actor_pixels_excluded=int((people[instance]&H).sum()),CURRENT_PRE_actor_core_overlap=0,exact_native_PRE_residual_overlap=exact_PRE,known_actor_A_pixels=int(A.sum()),known_actor_role=known_actor_role,independent_CREW_priority=independent_CREW,current_actor_overlap=actor_overlap,source_ineligible_missing_pixels=bad_veto,geometry_UNKNOWN_missing_pixels=bad_unknown,conflicts=conflicts,policy_sha256=policy['policy_sha256'],no_native_RGB_actor_veto_UNKNOWN_or_weights_changed=True,current_identity_NOT_declared_CREW_or_actor=True,DATA_zero_weights_validation_required=not soft_RGB_branch,RGB_policy='UNCHANGED_ORIGINAL_SOFT_BAND'if soft_RGB_branch else'EXISTING_VETO_ZERO',raw_PRE_original_sha256=array_sha(original_PRE),qualified_UNRESOLVED_RAW_PRE_exception_sha256=sha(uncertain_exception/'proof.json')if(uncertain_exception/'proof.json').exists()else None,qualified_NONACTOR_PRE_exception_sha256=sha(exception/'proof.json')if(exception/'proof.json').exists()else None,created=time.time());proofpath=artifact.with_suffix('.json');write(proofpath,proof);person['status']='unresolved_identity_geometry_UNKNOWN_source_ineligible';person['visibility_proof']=dict(path=str(proofpath.relative_to(f)),sha256=sha(proofpath),archive=str(artifact.relative_to(f)),archive_sha256=sha(artifact));resolutions.append(person['visibility_proof'])
 report.update(unknown_person_instances=remaining,conservative_current_visibility_resolutions=resolutions,status='unknown'if remaining or report['actor_novelty_trigger']else'pass');return report

def validate_DATA(root,frame):
 root=Path(root);f=root/'frames'/f'{frame:06d}';data=f/'data';rows=[]
 from marshawn_actor_coordinates import raw_mask_coverage_to_HD
 from marshawn_actor_confidence import native_confidence
 for p in sorted((f/'visibility_unknown_inputs').glob('*.json')):
  r=json.loads(p.read_text());camera=r['camera'];sp=root/'sources'/str(frame)/camera/'receipt.json';source=json.loads(sp.read_text());native=f/'native_masks'/(camera+'.npz');assert sha(sp)==r['source_receipt_sha256']and sha(native)==r['native_archive_sha256']
  with np.load(p.with_suffix('.npz'))as z:mask=z['unresolved_current_visibility'];A=z['known_actor_A']if'known_actor_A'in z.files else np.zeros_like(mask)
  assert array_sha(mask)==r['scope_mask_sha256'];coverage=raw_mask_coverage_to_HD(mask.astype('float32'),source['camera_row'])>1e-6;Acoverage=raw_mask_coverage_to_HD(A.astype('float32'),source['camera_row'])
  with np.load(native)as nz:nominal=nz['black']|nz['grey'];veto=nz['source_ineligible'];assert not(mask&nominal).any()and nz['unknown'][mask].all()
  with np.load(data/'semantics'/(camera+'.npz'))as z:
   eligible=coverage&z['valid'];assert z['unknown'][eligible].all();Ae=(Acoverage>1e-6)&z['valid']&~z['source_ineligible'];assert(z['W32'][Ae]>0).all()and(z['W64'][Ae]>0).all()
   soft=r.get('RGB_policy')=='UNCHANGED_ORIGINAL_SOFT_BAND'
   if soft:
    for name,width in [('W32',32),('W64',64)]:
     raw,_=native_confidence(nominal,veto,width);expected=raw_mask_coverage_to_HD(raw,source['camera_row']);expected[~z['valid']|z['source_ineligible']]=0;assert np.allclose(expected,z[name],atol=2e-6,rtol=0),(camera,name,float(np.max(np.abs(expected-z[name]))))
   else:assert z['source_ineligible'][eligible].all()and(z['W32'][eligible]==0).all()and(z['W64'][eligible]==0).all()
   rows.append(dict(camera=camera,instance=r['instance'],proof_sha256=sha(p),RGB_policy=r.get('RGB_policy','EXISTING_VETO_ZERO'),HD_pixels=int(eligible.sum()),positive_W64_HD_pixels=int((z['W64'][eligible]>0).sum()),W64_sum=float(z['W64'][eligible].sum()),native_R_disjoint_actors=True,geometry_UNKNOWN_retained=True,frozen_original_confidence_exact=soft,known_actor_A_preserved_positive=True))
 out=data/'visibility_unknown_QA_v74.json';write(out,dict(frame=frame,status='pass',records=rows,no_native_or_confidence_changes_by_visibility_resolution=True,positive_soft_band_is_NOT_claimed_zero_RGB=True));return out
