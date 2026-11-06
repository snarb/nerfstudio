import ast,copy,importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]/'scripts';s=importlib.util.spec_from_file_location('production_support',ROOT/'marshawn_semantic_4k_production_support.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def contract():return dict(scope='EXPERIMENTAL_4K_ORIGINAL_R0_W64_COLD',start_mode='cold',minimum_ancestry_step=106316,bundle='4K_RGB_native_confidence_plus_4000_step_maps',pure_resolution_causality=False,data='/new4K/data',geometry_data='/originalHD',mapped_support_sha256='d60baa9186a63f3769204b93cb46dcfd2ea16e3d41a72ca788934fe19bd2659f')
def test_production_recipe_cannot_hide_other_factors():
 m.validate_contract(contract())
 for k,v in [('scope','bagfixed'),('start_mode','warm'),('minimum_ancestry_step',22000),('pure_resolution_causality',True),('data','/originalHD'),('mapped_support_sha256','changed')]:
  c=contract();c[k]=v
  with pytest.raises(ValueError):m.validate_contract(c)
def test_unreleased_production_and_technical_reuse_rejected(tmp_path):
 c=contract();release=tmp_path/'release.json';release.write_text(json.dumps(dict(production_authorized=False,technical_only=False)));c['release']=str(release);p=tmp_path/'contract.json';p.write_text(json.dumps(c));sp=tmp_path/'spec.json';sp.write_text('{}')
 with pytest.raises(ValueError,match='production root release'):m.initialize(p,sp)
 with pytest.raises(ValueError,match='Technical smoke'):m.initialize(p,sp,1057)
def test_frozen_training_loop_and_checkpoint_inventory_are_preserved():
 # Source-independent comparison against reviewed original frozen method bodies.
 frozen=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_actor_candidate_v2_20261106/code/LookCloser/scripts/train_marshawn_semantic_admission.py')
 if not frozen.exists():pytest.skip('Historical frozen source unavailable')
 a=ast.parse(frozen.read_text());b=ast.parse((ROOT/'train_marshawn_semantic_4k_production.py').read_text())
 funcs=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)};fa,fb=funcs(a),funcs(b)
 for name in ['evaluate','finish','choose_checkpoint','protected_checkpoints','inventory_and_prune']:
  assert ast.dump(fa[name],include_attributes=False)==ast.dump(fb[name],include_attributes=False)
 loop=lambda node:next(n for n in ast.walk(node)if isinstance(n,ast.While))
 assert ast.dump(loop(fa['train']),include_attributes=False)==ast.dump(loop(fb['train']),include_attributes=False)
 assert "sha(data / 'ready.json')"in (ROOT/'train_marshawn_semantic_4k_production.py').read_text()
def test_new_entrypoint_does_not_shadow_parent_factory():
 tree=ast.parse((ROOT/'train_marshawn_semantic_4k_production.py').read_text());assignment=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='MODULE'for t in n.targets));assert ast.literal_eval(assignment.value)=='train_marshawn_semantic_4k_production'
def test_mutation_and_path_escape_fail_closed(tmp_path):
 p=tmp_path/'source';p.write_bytes(b'a');inv={'source':m.sha(p)};m.verify_files(tmp_path,inv);p.write_bytes(b'b')
 with pytest.raises(ValueError):m.verify_files(tmp_path,inv)
 with pytest.raises(ValueError):m.verify_files(tmp_path,{'../source':'x'})

def test_unlisted_mask_override_is_rejected(tmp_path):
 (tmp_path/'rgb.png').write_bytes(b'x');(tmp_path/'ready.json').write_text('{}');f=tmp_path/'lookcloser_frequencies';f.mkdir();(f/'train_AA.pt').write_bytes(b'map')
 m.verify_dataset_layout(tmp_path,{'rgb.png':'sha'},{'data/lookcloser_frequencies/train_AA.pt':'sha','censor/AA.png':'sha'})
 folder=tmp_path/'actor_masks';folder.mkdir();(folder/'train_AA.npz').write_bytes(b'unbound override')
 with pytest.raises(ValueError,match='Unexpected'):m.verify_dataset_layout(tmp_path,{'rgb.png':'sha'},{'data/lookcloser_frequencies/train_AA.pt':'sha'})

def test_durable_launcher_refuses_unreleased_or_changed_plans(tmp_path):
 spec=importlib.util.spec_from_file_location('launch4k',ROOT/'launch_marshawn_semantic_4k_production.py');launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
 for n in ['launch_plan.json','controller_plan.json','contract.json','spec.json']:(tmp_path/n).write_text(json.dumps({'local_artifacts':{}}if n=='launch_plan.json'else{}))
 release={'production_authorized':False,'technical_only':False};(tmp_path/'root_release.json').write_text(json.dumps(release))
 with pytest.raises(ValueError):launch.verify(tmp_path)
 release['production_authorized']=True
 for key,name in [('launch_plan_sha256','launch_plan.json'),('controller_plan_sha256','controller_plan.json'),('contract_sha256','contract.json'),('spec_sha256','spec.json')]:release[key]=launch.sha(tmp_path/name)
 (tmp_path/'root_release.json').write_text(json.dumps(release));launch.verify(tmp_path)
 (tmp_path/'spec.json').write_text('{"changed":true}')
 with pytest.raises(ValueError,match='Release mismatch'):launch.verify(tmp_path)
