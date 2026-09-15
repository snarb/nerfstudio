import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from contrastive_plane_guard import best_plane_decision


def test_each_depth_gets_its_own_best_orientation():
    near = np.full((2, 2, 4), .7);far = np.full_like(near, .1)
    near[0, 0] = .9;far[1, 0] = .9
    out = best_plane_decision(near, far, np.ones_like(near, bool), np.full((2, 2), 5.))
    assert out['reject_far'].tolist() == [False, True]


def test_unavailable_or_untextured_does_not_override():
    near = np.full((2, 2, 4), .9);far = np.full_like(near, .1)
    known = np.ones_like(near, bool);known[1, 0, :2] = False
    std = np.full((2, 2), 5.);std[:, 1] = 3.
    assert not best_plane_decision(near, far, known, std)['reject_far'].any()
