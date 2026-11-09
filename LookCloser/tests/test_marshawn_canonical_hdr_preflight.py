from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import math
import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_canonical_hdr_preflight import canonical_pq_from_unbaked_ap1,FrozenLookCloserDensityProxy,SIGNED_GAIN


def test_signed_pq_matches_actual_vendor_floor_peak_and_10k_reference():
    path=Path('/fsx/oregon/gta_marshawn_2_1/six_sec_20261019/code/engine_gta_pv_export/ca4d/ca4d/pq.py')
    if not path.exists():pytest.skip('Local pinned vendor source unavailable')
    spec=importlib.util.spec_from_file_location('vendor_pq_reference',path);vendor=importlib.util.module_from_spec(spec);spec.loader.exec_module(vendor)
    rgb=np.array([-1.,0.,.005/SIGNED_GAIN/2,.005/SIGNED_GAIN,1.,4000/SIGNED_GAIN,100.,1000.])
    actual=canonical_pq_from_unbaked_ap1(torch.tensor(rgb)).numpy()
    expected=vendor.encode_tier(rgb*SIGNED_GAIN,min_nits=.005,peak_nits=4000)
    assert np.max(np.abs(actual-expected))<6e-8
    assert actual[0]==actual[1]==actual[2]==actual[3]
    assert actual[-1]==actual[-2]==actual[-3]
    assert actual[-1]<1 # 4000 is content clamp, never ST2084 normalization.


def test_canonical_encoder_floor_and_peak_gradients_are_true_clamps():
    value=torch.tensor([0.,1.,1000.],requires_grad=True)
    code=canonical_pq_from_unbaked_ap1(value);code.sum().backward()
    assert value.grad[0]==value.grad[2]==0
    assert torch.isfinite(value.grad).all() and value.grad[1]>0


class Geo(nn.Module):
    def __init__(self,logit):super().__init__();self.raw=nn.Parameter(torch.tensor(float(logit)));self.fail=False
    def forward(self,points):
        if self.fail:raise RuntimeError('density failure')
        return torch.stack([self.raw.expand(len(points)),torch.zeros(len(points))],dim=1)


class FakeLC(nn.Module):
    def __init__(self,logit):
        super().__init__();self.mlp_geo=Geo(logit);self.aabb=torch.tensor([[0.,0,0],[1.,1,1]])
        self.density_activation='trunc_exp';self.spatial_distortion=None;self.enable_feature_reweighting=False
        self.freq_grid=SimpleNamespace(enabled=False);self.density_geometry_encoding=None;self.density_geometry_mlp=None
        self._density_before_activation=None
    def density_fn(self,points):return torch.exp(self.mlp_geo(points)[:,:1]+1.)


def test_proxy_capture_is_differentiable_exact_and_does_not_populate_field_cache():
    field=FakeLC(0);proxy=FrozenLookCloserDensityProxy(field);density=proxy.density_fn(torch.full((6,3),.5))
    assert torch.allclose(density,torch.full((6,1),math.e))
    assert torch.allclose(proxy._density_before_activation,torch.zeros((6,1)))
    assert field._density_before_activation is None and not field.mlp_geo._forward_hooks
    (proxy._density_before_activation.float()+math.log(proxy.average_init_density)).sum().backward()
    assert field.mlp_geo.raw.grad.item()==6
    field.mlp_geo.fail=True
    with pytest.raises(RuntimeError,match='density failure'):proxy.density_fn(torch.full((6,3),.5))
    assert not field.mlp_geo._forward_hooks


def test_native_surface_objective_works_through_proxy_with_extreme_negative_raw_logit():
    source=Path('/fsx/tmp/marshawn_001660_hd/final_pair/code')
    if not source.exists():pytest.skip('Frozen reviewed objective source unavailable')
    sys.path.insert(0,str(source))
    from marshawn_surface_band_objective import SurfaceBandObjective
    objective=SurfaceBandObjective.__new__(SurfaceBandObjective);nn.Module.__init__(objective)
    objective.rays=2;objective.samples=3;objective.weight=.01;objective.origins=torch.zeros((2,3))
    objective.sample_batch=lambda _step,_indices:dict(points=torch.full((2,3,3),.5),length=torch.tensor([.02,.03]))
    field=FakeLC(-100);proxy=FrozenLookCloserDensityProxy(field)
    loss,metrics=objective(proxy,0,torch.tensor([0,1]));loss.backward()
    expected=.01*(99.-np.log(np.array([.02,.03]))).mean()
    assert float(loss)==pytest.approx(expected,rel=1e-6)
    assert field.mlp_geo.raw.grad.item()==pytest.approx(-.01,rel=5e-6)
    assert torch.isfinite(loss) and torch.isfinite(field.mlp_geo.raw.grad)
    assert field._density_before_activation is None and not field.mlp_geo._forward_hooks


def test_proxy_rejects_unsupported_density_or_frequency_policy():
    field=FakeLC(0);field.density_activation='softplus'
    with pytest.raises(ValueError,match='trunc_exp'):FrozenLookCloserDensityProxy(field)
    field.density_activation='trunc_exp';field.freq_grid.enabled=True
    with pytest.raises(ValueError,match='Frequency'):FrozenLookCloserDensityProxy(field)
