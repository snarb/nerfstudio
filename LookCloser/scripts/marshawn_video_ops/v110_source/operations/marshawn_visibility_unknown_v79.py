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
  allowed=(allowed or(person['status']=='UNKNOWN_person_overlap'and known_actor_role))and actor_overlap==0 and bad_veto==0 and bad_unknown==0 and exact_PRE==0 and bool(unlabeled.any())
  if not allowed:remaining.append(instance);continue
  out=f/'visibility_unknown_inputs';out.mkdir(exist_ok=True);artifact=out/f'{camera}_person{instance}.npz'
  np.savez_compressed(artifact,unresolved_current_visibility=unlabeled,known_actor_A=A)
  proof=dict(frame=frame,camera=camera,instance=instance,kind='CURRENT_NN_PRE0_FLOW_ONLY_UNRESOLVED_VISIBILITY',status='explicit_geometry_UNKNOWN_source_ineligible',source_receipt_sha256=sha(source),candidate_archive_sha256=sha(cp),native_archive_sha256=sha(p),scope_mask_sha256=array_sha(unlabeled),scope_pixels=int(unlabeled.sum()),qualified_CURRENT_actor_pixels_excluded=int((people[instance]&H).sum()),CURRENT_PRE_actor_core_overlap=0,exact_native_PRE_residual_overlap=exact_PRE,known_actor_A_pixels=int(A.sum()),known_actor_role=known_actor_role,independent_CREW_priority=independent_CREW,current_actor_overlap=actor_overlap,source_ineligible_missing_pixels=bad_veto,geometry_UNKNOWN_missing_pixels=bad_unknown,conflicts=conflicts,policy_sha256=policy['policy_sha256'],no_native_RGB_actor_veto_UNKNOWN_or_weights_changed=True,current_identity_NOT_declared_CREW_or_actor=True,DATA_zero_weights_validation_required=True,created=time.time());proofpath=artifact.with_suffix('.json');write(proofpath,proof);person['status']='unresolved_identity_geometry_UNKNOWN_source_ineligible';person['visibility_proof']=dict(path=str(proofpath.relative_to(f)),sha256=sha(proofpath),archive=str(artifact.relative_to(f)),archive_sha256=sha(artifact));resolutions.append(person['visibility_proof'])
 report.update(unknown_person_instances=remaining,conservative_current_visibility_resolutions=resolutions,status='unknown'if remaining or report['actor_novelty_trigger']else'pass');return report

def validate_DATA(root,frame):
 root=Path(root);f=root/'frames'/f'{frame:06d}';data=f/'data';rows=[]
 from marshawn_actor_coordinates import raw_mask_coverage_to_HD
 for p in sorted((f/'visibility_unknown_inputs').glob('*.json')):
  r=json.loads(p.read_text());camera=r['camera'];sp=root/'sources'/str(frame)/camera/'receipt.json';s=json.loads(sp.read_text());assert sha(sp)==r['source_receipt_sha256']and sha(f/'native_masks'/(camera+'.npz'))==r['native_archive_sha256']
  with np.load(p.with_suffix('.npz'))as z:mask=z['unresolved_current_visibility'];A=z['known_actor_A']if'known_actor_A'in z.files else np.zeros_like(mask)
  Acoverage=raw_mask_coverage_to_HD(A.astype('float32'),s['camera_row'])
  assert array_sha(mask)==r['scope_mask_sha256'];coverage=raw_mask_coverage_to_HD(mask.astype('float32'),s['camera_row'])>1e-6
  with np.load(data/'semantics'/(camera+'.npz'))as z:
   eligible=coverage&z['valid'];assert z['source_ineligible'][eligible].all()and z['unknown'][eligible].all();assert (z['W32'][eligible]==0).all()and(z['W64'][eligible]==0).all();assert not(z['black'][eligible]|z['grey'][eligible]).any();Ae=(Acoverage>1e-6)&z['valid']&~z['source_ineligible'];assert(z['W32'][Ae]>0).all()and(z['W64'][Ae]>0).all()
  rows.append(dict(proof=str(p),proof_sha256=sha(p),camera=camera,instance=r['instance'],HD_pixels=int(eligible.sum()),source_ineligible_all=True,geometry_UNKNOWN_all=True,W32_W64_zero_all=True))
 out=data/'visibility_unknown_QA_v74.json';write(out,dict(frame=frame,status='pass',records=rows,existing_actor_RGB_and_weights_not_changed_by_resolution=True));return out
