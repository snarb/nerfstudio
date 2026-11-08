from pathlib import Path
import json,sys,fcntl,shutil,time,os,importlib.util
import numpy as np
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations/1700_EE2_readonly/application_v116';O.mkdir(exist_ok=True);sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from crew_guard_plan_v53 import candidates
write(O/'state.json',dict(stage='waiting_canonical_writer_lock',pid=os.getpid(),time=time.time()));held=open(V/'operations/source_factory_v74.lock','a+');fcntl.flock(held,fcntl.LOCK_EX)
f=V/'frames/001700';p=f/'native_masks/EE.npz';mp=p.with_suffix('.json');approval=V/'operations/root_1700_EE_current_actor_components_approval.json';assert sha(approval)=='d8e87e0047c7c5fc93a908f71419ee0cb08bfaaf270f1361305d42e9330af3f6';A=json.loads(approval.read_text());assert not(f/'native_semantic_ready.json').exists()
if(O/'complete.json').exists():assert sha(p)==json.loads((O/'complete.json').read_text())['new_native_sha256'];raise SystemExit
assert sha(p)==A['input_binding']['native_archive_sha256'];cp,_=candidates(V,1700,'EE');assert sha(cp)==A['input_binding']['person_archive_sha256'];archive=Path(A['proposal_archive']);assert sha(archive)==A['proposal_archive_sha256']
with np.load(archive)as z:part=np.logical_or.reduce([z['component'+str(i)]for i in [2,3,4,5,6]]);delta=part.copy()
assert array_sha(part)==A['proposal_union_array_sha256']and int(delta.sum())==28083
meta=json.loads(mp.read_text());up=f/meta['union_resolution']['receipt'];h=f/'qualified_hair_inputs/EE';stage=f/'ordered_current_semantic_component_stage.json';oldstage=json.loads(stage.read_text());backup=O/'before';backup.mkdir(exist_ok=True)
for src,name in [(p,'native.npz'),(mp,'native.json'),(up,'union_application.json'),(stage,'ordered_stage.json')]:shutil.copy2(src,backup/name)
shutil.copytree(h,backup/'qualified_RGB',dirs_exist_ok=True)
with np.load(p)as z:old={k:z[k].copy()for k in z.files}
base=V/'operations/marshawn_union_parts_v80.py';source=base.read_text().replace("d=f/'qualified_union_parts_v79'/cam;","d=f/'qualified_union_parts_v116'/(cam+'_instance'+str(instance)+'_components'+'_'.join(map(str,components)));");helper=O/'archival_parts_v116.py';helper.write_text(source);spec=importlib.util.spec_from_file_location('de1700approved',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);applied=module.apply(V,1700,'EE',2,[2,3,4,5,6],approval)
from marshawn_union_guard_v53 import validate_union
validate_union(V,1700,'EE')
with np.load(p)as z:new={k:z[k].copy()for k in z.files}
oldA=old['black']|old['grey'];assert np.array_equal(delta,part&~oldA);assert np.array_equal(new['black'],old['black']|delta)and np.array_equal(new['grey'],old['grey']);assert np.array_equal(new['qualified_actor_rgb'],old['qualified_actor_rgb']|part);assert np.array_equal(new['source_ineligible'],old['source_ineligible']&~part);assert np.array_equal(new['actor_ownership_uncertain'],old['actor_ownership_uncertain']|delta)
for key in set(old)-{'black','grey','qualified_actor_rgb','source_ineligible','actor_ownership_uncertain'}:assert np.array_equal(old[key],new[key]),key
conservation=O/'array_conservation.json';write(conservation,dict(status='pass',frame=1700,camera='EE',source_receipt_sha256=sha(V/'sources/1700/EE/receipt.json'),current_candidate_sha256=sha(cp),ROOT_approval_sha256=sha(approval),old_native_sha256=sha(backup/'native.npz'),new_native_sha256=sha(p),proposal_archive_sha256=sha(archive),added_actor_union_pixels=int(delta.sum()),existing_A_H_labels_preserved=True,new_ownership_uncertain=True,unknown_and_all_unrelated_arrays_bitwise_preserved=True,veto_trimmed_only_approved_current_NN_mask=True,union_guard_PASS=True,old_ordered_stage_sha256=sha(backup/'ordered_stage.json')))
for cam,digest in oldstage['native_after_application_sha256'].items():
 if cam!='EE':assert sha(f/'native_masks'/(cam+'.npz'))==digest
oldstage.setdefault('verified_current_NN_manual_extensions',[]).append(dict(camera='EE',ROOT_approval=str(approval),ROOT_approval_sha256=sha(approval),array_conservation=str(conservation),array_conservation_sha256=sha(conservation),baseline_ordered_stage_archive=str(backup/'ordered_stage.json'),baseline_ordered_stage_sha256=sha(backup/'ordered_stage.json')));oldstage['native_after_application_sha256']['EE']=sha(p);write(stage,oldstage)
complete=dict(status='pass',frame=1700,camera='EE',new_native_sha256=sha(p),added_actor_union_pixels=int(delta.sum()),current_NN_components_2_6_only=True,white_crew_component1_never_added=True,source_approval_sha256=sha(approval),array_conservation_sha256=sha(conservation),new_ordered_stage_sha256=sha(stage),explicit_source_extension_not_false_requalification=True,source_QA_DATA_guard_grid_still_required=True,application=applied,created=time.time());write(O/'complete.json',complete);write(O/'state.json',dict(stage='complete',new_native_sha256=sha(p),time=time.time()));print(json.dumps(complete),flush=True)
