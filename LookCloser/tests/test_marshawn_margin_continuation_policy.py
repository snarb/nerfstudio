import sys,json,hashlib,copy,types
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_margin_continuation_policy import MarginPolicy,install_checkpoint_margin,sha


class Field:
    _semantic_admission_installed=True
    def _normalize_positions(self,p):
        native=(p>=0).all(-1)&(p<=1).all(-1)
        return p,native,p.shape[:-1],p


def setup(tmp_path,arm='source_margin002'):
    mask=np.zeros((128,128,128),bool);mask[64,64,64]=True
    path=tmp_path/'grid.npz';np.savez(path,semantic_gate_pad1=mask,aabb=np.array([[0,0,0],[1,1,1]],float))
    script=Path(__file__).resolve().parents[1]/'scripts';helpers={n:sha(script/n)for n in ['marshawn_source_point_admission.py','marshawn_source_margin_admission.py','marshawn_readonly_point_selector.py']}
    manifest=tmp_path/'policy.json';manifest.write_text(json.dumps(dict(kind='PAIRED_SOURCE_MARGIN002_RUNTIME',source_checkpoint_sha256='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641',geometric_Linf_margin=.002,numerical_epsilon=2e-7,outside_unknown='original_numerical_epsilon',helper_sha256=helpers,source_grid=str(path),source_grid_sha256=sha(path))))
    model=types.SimpleNamespace(field=Field(),device=torch.device('cpu'))
    before=torch.get_rng_state().clone();policy=MarginPolicy(model,None,manifest,sha(manifest),arm);assert torch.equal(before,torch.get_rng_state())
    return model,policy,manifest


def test_gate_preserves_native_veto_and_exact_binding_resume(tmp_path):
    model,policy,_=setup(tmp_path);p=torch.tensor([[.503,.503,.503],[.1,.1,.1],[-.1,.5,.5]])
    assert model.field._normalize_positions(p)[1].tolist()==[True,False,False]
    policy.load_state_dict(policy.state_dict());bad=copy.deepcopy(policy.state_dict());bad['binding']['geometric_Linf_margin']=.001
    with pytest.raises(ValueError):policy.load_state_dict(bad)
    assert not torch.cuda.is_initialized()


def test_original_control_no_selector_math_change(tmp_path):
    model,policy,_=setup(tmp_path,'original_mapped');p=torch.tensor([[.503,.503,.503],[.1,.1,.1],[-.1,.5,.5]])
    assert model.field._normalize_positions(p)[1].tolist()==[True,True,False];policy.verify_installed()


def test_strict_inference_loader_rejects_missing_lineage(tmp_path):
    model,policy,path=setup(tmp_path);spec=tmp_path/'spec.json';spec.write_text(json.dumps(dict(parent_checkpoint_sha256=policy.binding['source_checkpoint_sha256'],margin_manifest=str(path),margin_manifest_sha256=sha(path),arm='source_margin002')))
    with pytest.raises(ValueError):install_checkpoint_margin(types.SimpleNamespace(field=Field(),device=torch.device('cpu')),{},spec)
    payload={'margin_fork_runtime':{'spec_sha256':sha(spec),'source_sha256':policy.binding['source_checkpoint_sha256'],'selector_state':policy.state_dict()}}
    other=types.SimpleNamespace(field=Field(),device=torch.device('cpu'));restored=install_checkpoint_margin(other,payload,spec);assert restored.binding==policy.binding


def test_no_closure_override_or_double_install(tmp_path):
    model,policy,path=setup(tmp_path)
    with pytest.raises(ValueError):MarginPolicy(model,None,path,sha(path),'source_margin002')
    model.field._normalize_positions=lambda p:None
    with pytest.raises(ValueError):policy.verify_installed()


def test_zero_source_selection_keeps_strict_child_loader_separate(tmp_path):
    from marshawn_margin_continuation_policy import install_zero_source_margin
    from train_marshawn_semantic_late_distortion import tree_digest
    _,policy,manifest=setup(tmp_path)
    payload={k:torch.tensor([i])for i,k in enumerate(['pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture'])};payload['step']=126316
    source=policy.binding['source_checkpoint_sha256'];anchor=tmp_path/'anchor.json';anchor.write_text(json.dumps(dict(step=126316,checkpoint_sha256=source,fullstate_hashes={k:tree_digest(v)for k,v in payload.items()if k!='step'})))
    spec=tmp_path/'spec.json';spec.write_text(json.dumps(dict(parent_checkpoint_sha256=source,anchor_audit=str(anchor),anchor_audit_sha256=sha(anchor),margin_manifest=str(manifest),margin_manifest_sha256=sha(manifest),arm='source_margin002')))
    model=types.SimpleNamespace(field=Field(),device=torch.device('cpu'));assert install_zero_source_margin(model,payload,source,spec).binding==policy.binding
    payload['step']=126317
    with pytest.raises(ValueError):install_zero_source_margin(types.SimpleNamespace(field=Field(),device=torch.device('cpu')),payload,source,spec)
