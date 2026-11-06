import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_exp_geometry import ExpGeometry
from marshawn_sh_geometry import SHGeometry
from test_marshawn_softplus_prefix_candidate import geometry,BOX,CAMS,Field
from nerfstudio.field_components.activations import trunc_exp

class ExpField(Field):
    density_activation='trunc_exp'
    correct_sh_directions=True
    def density_fn(self,points):
        return trunc_exp(self.raw+1.).expand(len(points),1)

def test_actual_native_exp_integral_and_band_gradients():
    obj=ExpGeometry(geometry(),BOX,CAMS,weight=.003,band_weight=.001,rays=6)
    f=ExpField();prefix,_=obj(f,4);prefix.backward()
    expected=.003*obj.sample(4)['length'].mean()*torch.exp(f.raw.detach()+1.)
    assert prefix.item()==pytest.approx(expected.item(),rel=1e-6)
    assert f.raw.grad.item()==pytest.approx(expected.item(),rel=1e-6)
    f=ExpField();loss,_=obj.band(f,4);loss.backward()
    delta=.001
    numeric=(obj.band(ExpField(2.+delta),4)[0]-obj.band(ExpField(2.-delta),4)[0])/(2*delta)
    assert f.raw.grad.item()<0
    assert numeric.item()==pytest.approx(f.raw.grad.item(),rel=.003)

def test_preserves_actual_native_clipped_backward_not_alternate_exp():
    obj=ExpGeometry(geometry(),BOX,CAMS,weight=.003,rays=6)
    f=ExpField(17.);loss,_=obj(f,4);loss.backward()
    length=obj.sample(4)['length'].mean()
    assert loss.item()==pytest.approx((.003*length*torch.exp(torch.tensor(18.))).item(),rel=1e-6)
    assert f.raw.grad.item()==pytest.approx((.003*length*torch.exp(torch.tensor(15.))).item(),rel=1e-6)

def test_private_streams_identical_across_activation_preserve_rng():
    exp=ExpGeometry(geometry(),BOX,CAMS,band_weight=.0003)
    soft=SHGeometry(geometry(),BOX,CAMS,band_weight=.001)
    rng=torch.random.get_rng_state().clone()
    for method in ('sample','sample_band'):
        a,b=getattr(exp,method)(15188),getattr(soft,method)(15188)
        assert all(torch.equal(a[k],b[k])for k in a)
        assert ((a['points']>torch.tensor(BOX[0]))&(a['points']<torch.tensor(BOX[1]))).all()
    assert torch.equal(rng,torch.random.get_rng_state())

def test_wrong_density_SH_bounds_and_nonfinite_rejected():
    obj=ExpGeometry(geometry(),BOX,CAMS)
    for prop,value in [('density_activation','softplus'),('correct_sh_directions',False)]:
        f=ExpField();setattr(f,prop,value)
        with pytest.raises(ValueError):obj(f,0)
        with pytest.raises(ValueError):obj.band(f,0)
    f=ExpField();f.aabb[0,0]=-.1
    with pytest.raises(ValueError,match='bounds'):obj.band(f,0)
    with pytest.raises(FloatingPointError):obj(ExpField(float('nan')),0)

def test_release_binds_completed_native_exp_parent(tmp_path):
    import json,hashlib
    from run_marshawn_exp_geometry_release import verify
    def put(path,value):path.write_text(json.dumps(value));return hashlib.sha256(path.read_bytes()).hexdigest()
    code=tmp_path/'code';code.mkdir();(code/'one.py').write_text('original')
    source_hash=put(tmp_path/'source_manifest.json',{'files':{'one.py':hashlib.sha256((code/'one.py').read_bytes()).hexdigest()}})
    probe_hash=put(tmp_path/'probe_P.json',{'passed':True});resume_hash=put(tmp_path/'verification_P.json',{'passed':True})
    parent=tmp_path/'parent.json';spec=tmp_path/'spec_P.json';release=tmp_path/'release_P.json'
    spec_hash=put(spec,dict(band_weight=0.,density_activation='trunc_exp',completed_parent_receipt=str(parent),prefix_candidate={'release_receipt':str(release)}))
    value=dict(approved_for_training=True,native_amp_zero_update_passed=True,technical_resume_passed=True,correct_sh_directions=True,
        density_activation='trunc_exp',spec_sha256=spec_hash,band_weight=0.,source_manifest_sha256=source_hash,
        evidence_sha256={'probe_P.json':probe_hash,'verification_P.json':resume_hash})
    put(release,value)
    with pytest.raises(ValueError,match='not completed'):verify(spec)
    value['completed_parent_receipt_sha256']=put(parent,dict(technical_smoke=False,config={'density_activation':'trunc_exp'}));put(release,value)
    assert verify(spec).name=='train_marshawn_exp_geometry.py'
    put(parent,dict(technical_smoke=True,config={'density_activation':'trunc_exp'}))
    with pytest.raises(ValueError,match='production exp'):verify(spec)
