import ast,copy,importlib.util
from pathlib import Path
import pytest
D=Path(__file__).resolve().parents[1]/'scripts';s=importlib.util.spec_from_file_location('clear_factory_test',D/'marshawn_clear_tau_factory.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def source():
 parent=dict(m.RECIPE,semantic_admission=dict(identity_control=False,manifest_sha256='a'*64));anchor=dict(anchor_status='COMPLETED_SELECTED_R0',producer_continues=False,step=111316,fullstate_hashes={k:'b'*64 for k in m.STATE_KEYS},semantic_grid_checks={k:True for k in ['core_capture','active_capture','active_pipeline','EMA_pipeline','learned_core_pipeline']},semantic_capture_metadata=dict(step=111312,radius=0,binary_warmup=False,semantic_receipt=dict(manifest_sha256='a'*64)),source_eval=dict(step=111316));return parent,anchor

def test_exact_selected_parent_and_only_reviewed_coefficients():
 p,a=source()
 for c in(0.,.0001,.001):assert m.validate_anchor(p,a,c) is p
 for c in(.01,1.,True,0):
  with pytest.raises(ValueError):m.validate_anchor(p,a,c)
 for key,value in [('step',30376),('anchor_status','EVALUATED_INTERMEDIATE'),('producer_continues',True)]:
  b=copy.deepcopy(a);b[key]=value
  with pytest.raises(ValueError):m.validate_anchor(p,b,.001)
 for key,value in [('objective','full'),('radius',1),('width',32),('distortion_final',1.)]:
  q=dict(p);q[key]=value
  with pytest.raises(ValueError):m.validate_anchor(q,a,.001)

def test_fork_hooks_only_one_ordinary_train_iteration_and_strict_resume_remains():
 text=(D/'train_marshawn_clear_tau.py').read_text();tree=ast.parse(text)
 assert len([n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='with_auxiliary'])==1
 assert len([n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='train_iteration'])==1
 assert "aux.load_state_dict(loaded['fork_runtime']['auxiliary_state'])"in text
 assert "'auxiliary_state':aux.state_dict()"in text
 assert 'finally: aux_enabled[0] = False'in text
 assert "model.config.distortion_loss_mult = .01"in text
 assert 'grad_scaler.update('not in text and '.step()'not in text
 assert text.index('require_controller_lease(os.environ')<text.index('import torch\n    isolation')<text.index("loaded = torch.load")
