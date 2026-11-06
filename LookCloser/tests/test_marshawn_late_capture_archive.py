import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from archive_marshawn_late_distortion import captured_model
from marshawn_tuning_occupancy import export_snapshot,load_snapshot

def fixture_capture():
    active=torch.zeros((1,2,2,2),dtype=torch.bool);active[0,0,1,0]=True
    values=dict(resolution=torch.tensor([2,2,2]),aabbs=torch.tensor([[0.,0.,0.,1.,1.,1.]]),occs=torch.full((1,8),.001),binaries=active)
    state={prefix+k:v.clone()for prefix in ['_model.occupancy_grid.','_model.adaptive_sampler.occupancy_grid.']for k,v in values.items()}
    return dict(step=64752,pipeline=state,actual_occupancy_capture=dict(core=active.clone(),active=active.clone(),step=64752,radius=0,binary_warmup=False))

def test_saved_core_preserved_even_when_cpu_threshold_would_disagree(tmp_path):
    payload=fixture_capture();model=captured_model(payload,16);checkpoint=tmp_path/'saved.ckpt';checkpoint.write_bytes(b'opaque saved checkpoint')
    receipt=export_snapshot(model,checkpoint,64752,tmp_path/'out')
    core,active,ema,boxes,record=load_snapshot(receipt,checkpoint)
    assert core.sum()==1 and active.sum()==1 and ema.max()==pytest.approx(.001)
    assert record['cpu_reconstruction_used']is False

def test_owner_sampler_mismatch_rejected():
    payload=fixture_capture();payload['pipeline']['_model.adaptive_sampler.occupancy_grid.binaries'].fill_(False)
    with pytest.raises(ValueError,match='aliases'):captured_model(payload,16)

def test_saved_active_mismatch_rejected():
    payload=fixture_capture();payload['actual_occupancy_capture']['active'].fill_(False)
    with pytest.raises(ValueError,match='capture differs'):captured_model(payload,16)

def test_stale_actual_capture_rejected(tmp_path):
    payload=fixture_capture();payload['actual_occupancy_capture']['step']=64736;model=captured_model(payload,16)
    with pytest.raises(ValueError,match='stale'):export_snapshot(model,tmp_path/'unused',64752,tmp_path/'out')
