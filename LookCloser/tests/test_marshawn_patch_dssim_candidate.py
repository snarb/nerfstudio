import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from marshawn_patch_dssim_candidate import patch_dssim, validate_patch_indices, TrainBatchCapture


def indices():
    y, x = torch.meshgrid(torch.arange(11), torch.arange(11), indexing='ij')
    return torch.stack([torch.ones_like(x)*3, y+40, x+90], -1).reshape(-1, 3)


def test_patch_identity_and_finite_nontrivial_gradient():
    torch.manual_seed(9)
    gt = torch.rand(121, 3)
    pred = (gt*.7+.1).requires_grad_()
    assert abs(float(patch_dssim(gt, gt, indices()))) < 1e-6
    loss = patch_dssim(pred, gt, indices())
    loss.backward()
    assert loss.dtype == torch.float32 and loss > 0
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().max() > 0


@pytest.mark.parametrize('kind', ['camera', 'xy', 'permutation', 'empty', 'incomplete'])
def test_reject_misgrouped_or_zero_patches(kind):
    i = indices()
    if kind == 'camera': i[15, 0] += 1
    elif kind == 'xy': i[7, 2] += 1
    elif kind == 'permutation': i = i.flip(0)
    elif kind == 'empty': i = i[:0]
    else: i = i[:120]
    with pytest.raises(ValueError): validate_patch_indices(i)


def test_structure_displacement_penalized_despite_equal_color_histogram():
    gt = torch.zeros(11, 11, 3); gt[:, 4:7] = .8
    pred = gt.roll(3, 1)
    assert torch.equal(gt.sort(1).values, pred.sort(1).values)
    assert patch_dssim(pred.reshape(-1,3), gt.reshape(-1,3), indices()) > .1


def test_capture_restores_inherited_method_after_exception():
    class Manager:
        def next_train(self, step): return step, {'indices': indices()}
    m = Manager()
    with pytest.raises(RuntimeError):
        with TrainBatchCapture(m) as c:
            m.next_train(7)
            assert c.calls == 1
            raise RuntimeError('expected')
    assert 'next_train' not in m.__dict__ and m.next_train(9)[0] == 9


def test_actual_standard_sampler_initial_estimate_and_patch_layout():
    from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig
    sampler = PatchPixelSamplerConfig(patch_size=11, ignore_mask=True).setup(num_rays_per_batch=16)
    assert sampler.num_rays_per_batch == 0  # Known DynamicBatch initial16 pitfall.
    sampler.set_num_rays_per_batch(262144 // 2048)
    assert sampler.num_rays_per_batch == 121
    actual = sampler.sample_method(121, 63, 1012, 1920, mask=torch.ones(63,1012,1920,1,dtype=torch.bool))
    validate_patch_indices(actual)
    assert (actual[:,0]>=0).all() and (actual[:,0]<63).all()
    assert (actual[:,1]>=0).all() and (actual[:,1]<1012).all()
    assert (actual[:,2]>=0).all() and (actual[:,2]<1920).all()
