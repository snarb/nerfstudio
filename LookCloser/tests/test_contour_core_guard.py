import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from contour_core_guard_luster_hybrid import guard
from contour_guard_luster_hybrid import guard as soft_guard


def test_shifted_edge_returns_exact_baseline_without_changing_distant_detail():
    base = np.zeros((96, 128, 3), np.float32)
    base[:, 48:] = .8
    pred = base.copy()
    pred[:, 48:52] = 0
    pred[:, 96:100] = .9
    weight, _, _ = guard(base, pred)
    soft, _, _ = soft_guard(base, pred)
    result = base + weight[..., None] * (pred - base)
    assert np.all(weight <= soft)
    assert np.all(weight[:, 48:52] == 0)
    assert np.array_equal(result[:, 48:52], base[:, 48:52])
    assert np.all(weight[:, 96:100] == 1)
    assert np.array_equal(result[:, 96:100], pred[:, 96:100])


def test_equal_input_and_detail_over_flat_base_are_untouched():
    base = np.full((96, 96, 3), .5, np.float32)
    for pred in (base, base + np.random.default_rng(3).uniform(
            -.1, .1, base.shape).astype(np.float32)):
        weight, _, _ = guard(base, pred)
        assert np.all(weight == 1)
