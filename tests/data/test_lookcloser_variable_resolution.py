"""Global FAS camera identity must survive ragged image gathering."""
import numpy as np
import pytest
import torch
from nerfstudio.lookcloser_pixel_sampler import LookCloserPixelSamplerConfig


def sampler(strength=1.):
    s = LookCloserPixelSamplerConfig(num_rays_per_batch=128, fas_strength=strength).setup()
    s.is_initialized = True
    s.patch_size = s.patch_stride = 1
    s.image_shapes = {2: (4, 2), 7: (2, 4), 10: (3, 3)}
    s.buckets = {i: torch.empty((0, 3), dtype=torch.int32) for i in range(16)}
    s.buckets[0] = torch.tensor([[7, 1, 3], [2, 3, 1], [10, 2, 2]])
    s.probs = np.array([1.] + [0.] * 15)
    return s


@pytest.mark.parametrize('strength', [0., .5, 1.])
def test_ragged_global_indices_and_colors(strength):
    torch.manual_seed(42)
    s = sampler(strength)
    batch = {'image': [torch.full((2, 4, 3), 7.), torch.full((4, 2, 3), 2.)],
             'image_idx': torch.tensor([7, 2])}
    result = s.sample(batch)
    ids, y, x = result['indices'].T
    assert result['image'].shape == (128, 3)
    assert torch.equal(result['image'][:, 0], ids.float())
    assert set(ids.tolist()) == {2, 7}
    assert (y[ids == 7] < 2).all() and (x[ids == 7] < 4).all()
    assert (y[ids == 2] < 4).all() and (x[ids == 2] < 2).all()
    if strength == 1.:
        assert (y[ids == 7] == 1).all() and (x[ids == 7] == 3).all()
        assert (y[ids == 2] == 3).all() and (x[ids == 2] == 1).all()
    assert s.sample_count == 1


def test_subset_change_invalidates_bucket_mapping():
    s = sampler()
    for cid, shape in [(7, (2, 4)), (2, (4, 2))]:
        result = s.sample({'image': [torch.full((*shape, 3), float(cid))], 'image_idx': torch.tensor([cid])})
        assert (result['indices'][:, 0] == cid).all()
        assert (result['image'] == cid).all()


def test_masked_ragged_fas_fails_explicitly():
    s = sampler()
    with pytest.raises(ValueError, match='unmasked supervision'):
        s.sample({'image': [torch.zeros(2, 4, 3)], 'mask': [torch.ones(2, 4, 1)], 'image_idx': torch.tensor([7])})
