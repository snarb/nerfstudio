import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_sh_geometry import SHGeometry,native_opacity_nll,SHGeometryAdapter
from marshawn_softplus_prefix_candidate import SoftplusPrefixCandidate
from test_marshawn_softplus_prefix_candidate import geometry,BOX,CAMS,Field

class SHField(Field):
    correct_sh_directions=True

def test_band_has_true_native_softplus_gradient_and_stable_opacity():
    obj=SHGeometry(geometry(),BOX,CAMS,weight=.003,band_weight=.001,rays=6);field=SHField()
    loss,_=obj.band(field,4);loss.backward();gradient=field.raw.grad.item()
    assert gradient<0
    delta=.001
    numeric=(obj.band(SHField(2.+delta),4)[0]-obj.band(SHField(2.-delta),4)[0])/(2*delta)
    assert numeric.item()==pytest.approx(gradient,rel=.003)
    tau=torch.tensor([1e-20,1e-8,.1,10.,1000.],requires_grad=True)
    value=native_opacity_nll(tau);value.sum().backward()
    assert torch.isfinite(value).all() and torch.isfinite(tau.grad).all()
    with pytest.raises(FloatingPointError):native_opacity_nll(torch.tensor([0.]))

def test_band_stream_preserves_prefix_packet_global_rng_and_restart():
    arr=geometry();base=SoftplusPrefixCandidate(arr,BOX,CAMS,weight=.003)
    a=SHGeometry(arr,BOX,CAMS,weight=.003,band_weight=.001);b=SHGeometry(arr,BOX,CAMS,weight=.003,band_weight=.001)
    rng=torch.random.get_rng_state().clone();original=base.sample(15188);band=a.sample_band(15188);packet=a.sample(15188)
    assert all(torch.equal(original[k],packet[k])for k in original)
    assert all(torch.equal(band[k],b.sample_band(15188)[k])for k in band)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert ((band['points']>torch.tensor(BOX[0]))&(band['points']<torch.tensor(BOX[1]))).all()

def test_wrong_actual_SH_and_density_activation_rejected():
    obj=SHGeometry(geometry(),BOX,CAMS,band_weight=.001)
    field=SHField();field.correct_sh_directions=False
    with pytest.raises(ValueError,match='SH'):obj(field,0)
    with pytest.raises(ValueError,match='SH'):obj.band(field,0)
    field=SHField();field.density_activation='trunc_exp'
    with pytest.raises(ValueError,match='softplus'):obj.band(field,0)

def test_production_gate_requires_hash_bound_source_and_real_receipts(tmp_path):
    import json,hashlib
    from run_marshawn_sh_geometry_release import verify
    def put(path,value):path.write_text(json.dumps(value));return hashlib.sha256(path.read_bytes()).hexdigest()
    code=tmp_path/'code';code.mkdir();(code/'one.py').write_text('original')
    source=tmp_path/'source_manifest.json';source_hash=put(source,{'files':{'one.py':hashlib.sha256((code/'one.py').read_bytes()).hexdigest()}})
    probe_hash=put(tmp_path/'probe_A.json',{'passed':True});resume_hash=put(tmp_path/'verification_A.json',{'passed':True})
    spec=tmp_path/'spec_A.json';release=tmp_path/'release_A.json';spec_hash=put(spec,{'band_weight':0.,'prefix_candidate':{'release_receipt':str(release)}})
    value=dict(approved_for_training=True,native_amp_zero_update_passed=True,technical_resume_passed=True,correct_sh_directions=True,
               spec_sha256=spec_hash,band_weight=0.,source_manifest_sha256=source_hash,
               evidence_sha256={'probe_A.json':probe_hash,'verification_A.json':resume_hash})
    put(release,value);assert verify(spec).name=='train_marshawn_sh_geometry.py'
    (code/'one.py').write_text('drift')
    with pytest.raises(ValueError,match='Frozen source changed'):verify(spec)
