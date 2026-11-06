import copy
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]/'scripts'
def module(name):
 s=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module('marshawn_semantic_4k_technical_support');supervisor=module('run_marshawn_semantic_4k_technical')

def contract():return dict(scope='TECHNICAL_ONLY_4K_R0_W64',smoke_steps=1057,pause_after=1024,production_authorized=False,READY_creation_allowed=False,geometry_data='/HD',data='/4K',mapped_support_sha256='d60baa9186a63f3769204b93cb46dcfd2ea16e3d41a72ca788934fe19bd2659f')

def test_contract_has_no_shortcut_or_promotion():
 m.validate_contract(contract())
 for key,value in [('smoke_steps',32),('pause_after',0),('production_authorized',True),('READY_creation_allowed',True),('mapped_support_sha256','changed'),('data','/HD')]:
  c=contract();c[key]=value
  with pytest.raises(ValueError):m.validate_contract(c)

def test_bounded_exclusive_policy():
 p=dict(scope='TECHNICAL_ONLY_4K_R0_W64',production_authorized=False,gpu=2,deadline_seconds=900,candidate_mib_cap=12288,smoke_steps=1057,pause_after=1024,monitor_seconds=5);supervisor.validate_policy(p)
 for k,v in [('deadline_seconds',901),('candidate_mib_cap',12289),('gpu',8),('pause_after',32),('production_authorized',True)]:
  q=dict(p);q[k]=v
  with pytest.raises(ValueError):supervisor.validate_policy(q)

def test_strict_source_hashes(tmp_path):
 p=tmp_path/'file';p.write_bytes(b'x');inventory={'file':m.sha(p)};m.verify_files(tmp_path,inventory);p.write_bytes(b'y')
 with pytest.raises(ValueError):m.verify_files(tmp_path,inventory)
 with pytest.raises(ValueError):m.verify_files(tmp_path,{'../file':'x'})

def test_fullstate_hash_catches_tensor_dtype_and_rng_changes():
 import torch
 import numpy as np
 x={'pipeline':torch.tensor([1.,2.]),'rng':np.array([2,5],dtype=np.uint32),'optimizers':{'step':torch.tensor(32.)}}
 assert m.tree_digest(x)==m.tree_digest(copy.deepcopy(x))
 y=copy.deepcopy(x);y['pipeline']=y['pipeline'].double();assert m.tree_digest(x)!=m.tree_digest(y)
 y=copy.deepcopy(x);y['rng'][0]+=1;assert m.tree_digest(x)!=m.tree_digest(y)
 y=copy.deepcopy(x);y['optimizers']['step']+=1;assert m.tree_digest(x)!=m.tree_digest(y)

def test_unreleased_initialization_fails_before_runtime(tmp_path):
 import json
 c=contract();release=tmp_path/'release.json';release.write_text(json.dumps(dict(technical_authorized=False,production_authorized=False)));c['release']=str(release);cp=tmp_path/'contract.json';c['contract_path']=str(cp);cp.write_text(json.dumps(c));spec=tmp_path/'spec.json';spec.write_text('{}')
 with pytest.raises(ValueError,match='technical-only release'):m.initialize(cp,spec,1057)

def test_worker_keeps_bounded_gate_and_separate_source_contract():
 text=(ROOT/'train_marshawn_semantic_4k_technical.py').read_text()
 assert "smoke_steps != 1057" in text
 assert "sha(data / 'source_metadata.json')"in text
 assert "sha(data / 'ready.json')"not in text
 assert "from train_marshawn_semantic_admission import build_config as original"in text
 assert "begin(trainer, sampler, state, output, resume)"in text

def test_readonly_tree_digest_does_not_advance_rng():
 import torch
 import numpy as np
 import random
 state=(torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate())
 m.tree_digest(state)
 assert torch.equal(state[0],torch.get_rng_state())
 assert m.tree_digest(state[1])==m.tree_digest(np.random.get_state())
 assert state[2]==random.getstate()

def test_checkpoint_inventory_runs_without_training_locals(tmp_path):
 import ast,json
 text=(ROOT/'train_marshawn_semantic_4k_technical.py').read_text();tree=ast.parse(text)
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ('protected_checkpoints','inventory_and_prune')]
 namespace=dict(Path=Path,json=json,sha=m.sha,atomic_json=m.atomic)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'isolated_inventory','exec'),namespace)
 folder=tmp_path/'checkpoints';folder.mkdir();old=folder/'step-000000001.ckpt';old.write_bytes(b'old');current=folder/'step-000001024.ckpt';current.write_bytes(b'checkpoint')
 result=namespace['inventory_and_prune'](tmp_path,current,1024,[])
 assert result['1024']['retained']is True and current.exists()
 assert result['1']['retained']is False and not old.exists()


def test_main_alias_does_not_shadow_original_factory():
 import ast
 tree=ast.parse((ROOT/'train_marshawn_semantic_4k_technical.py').read_text())
 assignment=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='MODULE'for t in n.targets))
 assert ast.literal_eval(assignment.value)=='train_marshawn_semantic_4k_technical'
 calls=[]
 for node in tree.body:
  if isinstance(node,ast.FunctionDef):
   calls.extend(node.name for child in ast.walk(node)if isinstance(child,ast.Call)and isinstance(child.func,ast.Name)and child.func.id=='begin')
 assert calls==['train']
