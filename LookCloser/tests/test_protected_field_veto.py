import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from study_protected_field_veto import near_native, eligible


def test_near_native_uses_integer_centers_and_rejects_unknown_outside():
    depth = torch.tensor([[0., 1., 2., float('nan')]])
    u = torch.tensor([.6, 1.6, 0., 3., -1., float('nan')])
    z = torch.ones(6)
    assert near_native(depth, u, torch.zeros(6), z).tolist() == [True, False, False, False, False, False]


def test_any_near_protects_and_unknown_positive_field_cannot_change():
    t = torch.tensor([-.1, -.1, -.1, .1, -.1, -.1, float('nan')])
    w = torch.tensor([2., 2., 0., 2., 2., 2., 2.])
    n = torch.tensor([0, 1, 0, 0, 0, 0, 0])
    f = torch.tensor([6, 62, 62, 62, 5, 7, 62])
    assert eligible(t, w, n, f).tolist() == [True, False, False, False, False, True, False]


def test_torch_near_matches_independent_numpy_footprint():
    from prune_measured_free_surface import near_tap_evidence
    rng = np.random.default_rng(13); d = rng.choice([0., 1., 1.001, 1.01, np.nan], (20, 30)).astype('float32')
    uv = rng.uniform(-3, 33, (1000, 2)).astype('float32'); z = np.ones(1000, 'float32')
    expected = near_tap_evidence(d, uv, z, radius=0)
    actual = near_native(torch.from_numpy(d), torch.from_numpy(uv[:, 0]), torch.from_numpy(uv[:, 1]), torch.from_numpy(z))
    np.testing.assert_array_equal(actual.numpy(), expected)
