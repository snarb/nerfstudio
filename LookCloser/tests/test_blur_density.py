from types import SimpleNamespace
import math
import pytest
import torch
from nerfstudio.fields.lookcloser_field import LookCloserField


def holder(**kwargs):
    return SimpleNamespace(aabb=torch.tensor([[-.1,-.1,-.1],[.1,.1,.1]]),
        **(dict(density_activation='softplus',density_normalization='none',
                density_reference_length=1.0,
                density_fp32=False,density_clip=False)|kwargs))


def test_legacy_softplus_exact():
    x=torch.linspace(-12,12,65,dtype=torch.float16)
    assert torch.equal(LookCloserField.activate_density(holder(),x),torch.nn.functional.softplus(x+1))


@pytest.mark.parametrize('activation',['softplus','trunc_exp'])
def test_optical_thickness_under_world_scale(activation):
    h=holder(density_activation=activation,density_normalization='aabb',density_fp32=True)
    x=torch.tensor([-5.,0.,10.703],dtype=torch.float16,requires_grad=True)
    a=LookCloserField.activate_density(h,x);h.aabb=h.aabb*7
    b=LookCloserField.activate_density(h,x)
    torch.testing.assert_close(a*.02,b*.14)
    assert a.dtype==torch.float32 and a.isfinite().all()
    (a*1e-4).sum().backward();assert x.grad.isfinite().all() and (x.grad>0).all()


def test_eval_exp_requires_explicit_fp32_outside_autocast():
    x=torch.tensor([10.703],dtype=torch.float16)
    assert not LookCloserField.activate_density(holder(density_activation='trunc_exp'),x).isfinite().all()
    assert LookCloserField.activate_density(holder(density_activation='trunc_exp',density_fp32=True),x).isfinite().all()


def test_clipping_is_separate_and_stops_outside_gradients():
    x=torch.tensor([-17.,0.,12.],requires_grad=True)
    h=holder(density_activation='trunc_exp',density_fp32=True,density_clip=True)
    y=LookCloserField.activate_density(h,x);y.sum().backward()
    torch.testing.assert_close(y,torch.exp(torch.tensor([-15.,1.,12.])))
    assert x.grad[0]==0 and x.grad[2]==0 and x.grad[1]>0


def test_normalized_softplus_requires_fp32_for_tiny_world_units():
    h=holder(density_normalization='aabb')
    h.aabb=h.aabb*1e-4
    x=torch.tensor([10.],dtype=torch.float16,requires_grad=True)
    assert not LookCloserField.activate_density(h,x).isfinite().all()
    h.density_fp32=True
    y=LookCloserField.activate_density(h,x)
    assert y.isfinite().all()
    (y*1e-5).sum().backward()
    assert x.grad.isfinite().all() and (x.grad>0).all()


@pytest.mark.parametrize('mode',['aabb','canonical_aabb'])
def test_canonical_reference_preserves_optical_thickness_and_gradients(mode):
    h=holder(density_normalization=mode,density_reference_length=3.0 if mode=='aabb' else 1.0)
    h.aabb=torch.tensor([[-1.5]*3,[1.5]*3])
    old=torch.linspace(-12,12,65,dtype=torch.float16,requires_grad=True)
    new=old.detach().clone().requires_grad_()
    delta=torch.full_like(old,.01,dtype=torch.float32)
    expected=torch.nn.functional.softplus(old+1)*delta
    actual=LookCloserField.activate_density(h,new)*delta
    assert torch.equal(expected,actual)
    expected.sum().backward();actual.sum().backward()
    assert torch.equal(old.grad,new.grad)
    h.aabb=h.aabb*1e-5
    scaled=LookCloserField.activate_density(h,new)
    assert scaled.isfinite().all()
    torch.testing.assert_close(scaled*(delta*1e-5),expected,rtol=1e-5,atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='TCNN CUDA contract')
def test_corrected_sh_addition_theorem():
    import tinycudann as tcnn
    encoding=tcnn.Encoding(n_input_dims=3,encoding_config={'otype':'SphericalHarmonics','degree':4})
    h=SimpleNamespace(direction_encoding=encoding,correct_sh_directions=True)
    d=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,-1.],[1.,2.,3.]],device='cuda')
    d=d/d.norm(dim=-1,keepdim=True)
    result=LookCloserField.encode_directions(h,d).float()
    torch.testing.assert_close(result.square().sum(-1),torch.full((4,),16/(4*math.pi),device='cuda'),rtol=.003,atol=.003)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='TCNN CUDA field')
def test_density_query_and_occupancy_use_same_parameterization():
    aabb=torch.tensor([[-.1,-.1,-.1],[.1,.1,.1]],device='cuda')
    field=LookCloserField(aabb=aabb,freq_grid=SimpleNamespace(enabled=False),
        enable_feature_reweighting=False,log2_hashmap_size=12,
        density_activation='trunc_exp',density_normalization='aabb',density_fp32=True,
        correct_sh_directions=True).cuda()
    positions=torch.tensor([[0.,0.,0.],[.02,-.03,.04],[2.,0.,0.]],device='cuda')
    directions=torch.tensor([[0.,0.,1.]],device='cuda').expand_as(positions)
    with torch.no_grad():
        direct,_=field.query_points(positions,directions)
        proposal=field.density_fn(positions)
    torch.testing.assert_close(direct,proposal)
    assert direct[-1]==0 and direct[:-1].isfinite().all()
