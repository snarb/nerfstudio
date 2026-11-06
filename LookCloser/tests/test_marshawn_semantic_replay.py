import importlib.util
from pathlib import Path
import numpy as np
import pytest

def module(name):
 p=Path(__file__).parents[1]/'scripts'/name;s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_pixel_evidence_does_not_silently_accept_large_differences():
 m=module('replay_marshawn_semantic_fork_zero.py');a=np.zeros((2,2,3),np.uint8);b=a.copy();b[0,0,0]=2
 d=m.pixel_delta(a,b);assert d['max_code_delta']==2 and d['changed_values']==1
 with pytest.raises(ValueError):m.pixel_delta(a,b.astype(np.uint16))

def test_replay_contains_no_training_iteration_or_backward_path():
 import ast
 tree=ast.parse((Path(__file__).parents[1]/'scripts/replay_marshawn_semantic_fork_zero.py').read_text())
 calls=[n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
 assert not set(calls)&{'train_iteration','backward','step','optimizer_scaler_step_some'}

def test_replay_policy_rejects_technical_scope():
 m=module('run_marshawn_semantic_replay_scope.py');p=dict(gpu=3,max_seconds=600,max_process_mib=12288,allocator_cap_gib=10,production_training=False,checks_interval_seconds=5,producer_stale_seconds=180,specs={'original':'a','fork':'b'});r=dict(approved_for_readonly_overlap=True,policy_sha256='x');m.validate(p,r,'x')
 p['specs']['technical_control001']='c'
 with pytest.raises(ValueError):m.validate(p,r,'x')
 p['specs'].pop('technical_control001');p['max_seconds']=601
 with pytest.raises(ValueError):m.validate(p,r,'x')
