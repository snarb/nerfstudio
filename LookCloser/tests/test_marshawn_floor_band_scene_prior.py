from pathlib import Path
import json
import math
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from marshawn_floor_band_scene_prior import FloorBandScenePrior, sha, validate_floor_domain


def fixture_prior(tmp_path, *, weight=.003, areas_ratio=False):
    normal = np.array([.6, 0, .8])
    xy = np.array([[[0, 0], [.1, 0], [0, .1]],
                   [[-.4, -.4], [-.2 if areas_ratio else -.3, -.4], [-.4, -.2 if areas_ratio else -.3]]])
    triangles = np.concatenate([xy, (-.75 * xy[..., :1])], axis=-1)
    areas = .5 * np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=-1)
    bounds = np.array([[-1, -1, -1], [1, 1, 1.]])
    archive = tmp_path / 'domain.npz'
    np.savez(archive, triangles_xyz=triangles, triangle_area=areas, normal=normal, offset=0.,
             halfwidth=.02, aabb=bounds, triangle_common_clear_train_votes=np.ones((2, 3), bool),
             source_cameras=np.array(['AA', 'AB', 'AC']))
    transforms = tmp_path / 'transforms.json'
    transforms.write_text(json.dumps(dict(train_filenames=['images/train_'+c+'.png' for c in ['AA', 'AB', 'AC']])))
    preparation = tmp_path / 'preparation.json';preparation.write_text('{}')
    payload = dict(kind='FINITE_SOFT_GEOMETRIC_HYPOTHESIS_NOT_MEASURED_DEPTH',
        no_eval_RGB_or_gains_fitted=True, no_los_negative_constraints=True, no_carving=True,
        bindings={str(transforms):sha(transforms), str(preparation):sha(preparation)},
        domain_triangles_file=str(archive), domain_triangles_sha256=sha(archive),
        fit_normal=normal.tolist(), fit_offset=0., surface_area_preparedworld=float(areas.sum()),
        approved_for_training=False, limitations=['test scene prior, not depth GT'])
    manifest=tmp_path/'proposal.json';manifest.write_text(json.dumps(payload))
    prior=FloorBandScenePrior(manifest,sha(manifest),bounds,transforms,preparation,weight=weight)
    return prior, manifest, transforms, preparation


class NativeTruncExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):ctx.save_for_backward(value);return torch.exp(value)
    @staticmethod
    def backward(ctx, grad):return grad*torch.exp(ctx.saved_tensors[0].clamp(-15, 15))


class ConstantNativeField(torch.nn.Module):
    def __init__(self, value):
        super().__init__();self.raw=torch.nn.Parameter(torch.tensor(float(value)))
        self.aabb=torch.tensor([[-1.,-1,-1],[1.,1,1]])
        self.average_init_density=.01;self.spatial_distortion=None
    def density_fn(self, positions):
        self._density_before_activation=self.raw.expand(len(positions),1)
        return self.average_init_density*NativeTruncExp.apply(self._density_before_activation)


@pytest.mark.parametrize('raw', [-5., 0., 5.])
def test_constant_density_physical_integral_forward_and_gradient(tmp_path, raw):
    prior,*_=fixture_prior(tmp_path);field=ConstantNativeField(raw)
    value,_=prior(field,123);value.backward()
    tau=.04*.01*math.exp(raw)
    expected=-prior.weight*math.log(-math.expm1(-tau))
    expected_gradient=-prior.weight*tau/math.expm1(tau)
    assert float(value)==pytest.approx(expected,rel=2e-6)
    assert float(field.raw.grad)==pytest.approx(expected_gradient,rel=2e-5)


@pytest.mark.parametrize('raw', [-80., -30., -15., 15.])
def test_low_logit_native_STE_does_not_explode_auxiliary_gradient(tmp_path, raw):
    prior,*_=fixture_prior(tmp_path);field=ConstantNativeField(raw)
    value,_=prior(field,9);value.backward()
    assert bool(torch.isfinite(value))and bool(torch.isfinite(field.raw.grad))
    assert -prior.weight-1e-7<=float(field.raw.grad)<=0
    if raw<=-15:assert float(field.raw.grad)==pytest.approx(-prior.weight,rel=1e-5)


def test_world_normal_strata_and_stateless_private_rng(tmp_path):
    prior,*_=fixture_prior(tmp_path)
    py=random.getstate();npstate=np.random.get_state();state=torch.get_rng_state()
    a=prior.sample_batch(10001);b=prior.sample_batch(10001);c=prior.sample_batch(10002)
    assert torch.equal(a['points'],b['points'])and not torch.equal(a['points'],c['points'])
    assert torch.equal(state,torch.get_rng_state())and random.getstate()==py
    after=np.random.get_state();assert after[0]==npstate[0]and np.array_equal(after[1],npstate[1])and after[2:]==npstate[2:]
    signed=(a['points']*prior.normal).sum(-1)
    torch.testing.assert_close(signed,a['offsets'],rtol=0,atol=1e-7)
    bins=(a['offsets']+.02)/.04*32
    assert bool(((bins>=torch.arange(32))&(bins<torch.arange(32)+1)).all())
    assert not isinstance(prior,torch.nn.Module)
    assert all(not x.requires_grad for x in prior.__dict__.values()if torch.is_tensor(x))


def test_area_weighted_continuous_sampling(tmp_path):
    prior,*_=fixture_prior(tmp_path,areas_ratio=True)
    chosen=torch.cat([prior.sample_batch(step)['triangle_indices']for step in range(64)])
    assert float((chosen==1).float().mean())==pytest.approx(.8,abs=.025)
    centers=prior.sample_batch(50)['centers']
    assert len(torch.unique(centers,dim=0))==64  # not the old fixed1457 support spikes


def test_manifest_and_current_geometry_corruption_are_rejected(tmp_path):
    prior,manifest,transforms,prep=fixture_prior(tmp_path)
    with pytest.raises(ValueError,match='SHA'):
        FloorBandScenePrior(manifest,'bad',prior.bounds.numpy(),transforms,prep)
    transforms.write_text('{}')
    with pytest.raises(ValueError,match='binding'):
        FloorBandScenePrior(manifest,sha(manifest),prior.bounds.numpy(),transforms,prep)


def test_domain_band_cannot_cross_aabb_or_have_wrong_units():
    triangles=np.array([[[0,0,0],[.1,0,0],[0,.1,0]]]);areas=np.array([.005]);bounds=[[-1,-1,-1],[1,1,1]]
    with pytest.raises(ValueError,match='Unit'):
        validate_floor_domain(triangles,areas,[0,0,2],0,.02,bounds)
    with pytest.raises(ValueError,match='inside'):
        validate_floor_domain(triangles,areas,[0,0,1],0,.02,[[-1,-1,-.01],[1,1,1]])
    with pytest.raises(ValueError,match='area'):
        validate_floor_domain(triangles,areas*2,[0,0,1],0,.02,bounds)
