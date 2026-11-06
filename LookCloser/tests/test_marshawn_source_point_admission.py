import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_semantic_cell_admission import WholeCellAdmission
from marshawn_source_point_admission import SourcePointAdmission


def test_random_source_points_match_independent_summed_volume_lookup():
    rng = np.random.default_rng(17)
    mask = rng.random((7, 8, 9)) > .85
    bounds = np.array([[-.84, -.21, -.07], [.15, .42, .63]])
    points = rng.uniform(bounds[0]-.02, bounds[1]+.02, (4096, 3)).astype(np.float32)
    lookup = SourcePointAdmission(torch.from_numpy(mask), bounds)
    np.testing.assert_array_equal(lookup(torch.from_numpy(points)).numpy(), WholeCellAdmission(mask, bounds).point_admitted(points))


def test_closed_corner_contact_epsilon_and_outside_unknown():
    mask = torch.zeros((2, 2, 2), dtype=torch.bool)
    mask[1, 1, 1] = True
    lookup = SourcePointAdmission(mask, [[0, 0, 0], [1, 1, 1]], epsilon=1e-6)
    points = torch.tensor([[.5, .5, .5], [.5-5e-7]*3, [.5-2e-6]*3, [-.1, .1, .1], [.1, .1, .1]], dtype=torch.float64)
    assert lookup(points).tolist() == [True, True, False, True, False]


def test_native_grid_centers_do_not_gain_an_extra_whole_cell_layer():
    mask = torch.zeros((5, 5, 5), dtype=torch.bool)
    mask[2, 2, 2] = True
    lookup = SourcePointAdmission(mask, [[0, 0, 0], [1, 1, 1]])
    centers = (torch.stack(torch.meshgrid(*[torch.arange(5)]*3, indexing='ij'), -1)+.5)/5
    assert int(lookup(centers).sum()) == 1
    cover = WholeCellAdmission(mask.numpy(), [[0, 0, 0], [1, 1, 1]]).map_target_grid([5, 5, 5], [[0, 0, 0], [1, 1, 1]])
    assert int(cover.sum()) == 27


def test_rejects_tolerance_spanning_more_than_two_cells():
    with pytest.raises(ValueError):
        SourcePointAdmission(torch.ones((2, 2, 2), dtype=torch.bool), [[0, 0, 0], [.001, .001, .001]], epsilon=.001)
