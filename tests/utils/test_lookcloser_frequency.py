"""Tests for automatic threshold-free LookCloser frequency-map selection."""

import torch

from nerfstudio.utils.lookcloser_frequency import (
    bootstrap_select,
    first_crossing_levels,
    guided_median_levels,
    knee_levels,
    map_quality,
    monotonic_recovery,
    relative_ensemble_levels,
)


def test_monotonic_envelope_and_absolute_crossing():
    scores = torch.tensor([[0.2, 0.1], [0.5, 0.4], [0.45, 0.8], [0.9, 0.9]])
    envelope = monotonic_recovery(scores)
    assert bool((envelope[1:] >= envelope[:-1]).all())
    levels, unresolved = first_crossing_levels(scores, 0.7)
    torch.testing.assert_close(levels, torch.tensor([3, 2]))
    assert not bool(unresolved.any())


def test_relative_ensemble_and_knee_handle_flat_curves():
    scores = torch.tensor(
        [
            [[0.2, 0.1]],
            [[0.6, 0.1]],
            [[0.85, 0.1]],
            [[0.9, 0.1]],
        ]
    )
    relative, _ = relative_ensemble_levels(scores, center=0.5)
    knee, _ = knee_levels(scores)
    assert int(relative[0, 1]) == 0
    assert int(knee[0, 1]) == 0
    assert int(relative[0, 0]) > 0


def test_quality_penalizes_collapsed_map_and_bootstrap_is_deterministic():
    proxy = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    diverse = torch.arange(16).reshape(4, 4) % 4
    collapsed = torch.zeros((4, 4), dtype=torch.long)
    diverse_quality = map_quality(diverse, proxy, n_levels=4)
    collapsed_quality = map_quality(collapsed, proxy, n_levels=4)
    assert diverse_quality.normalized_entropy > collapsed_quality.normalized_entropy
    assert diverse_quality.nonempty_bins > collapsed_quality.nonempty_bins
    qualities = {"diverse": [diverse_quality] * 3, "collapsed": [collapsed_quality] * 3}
    first = bootstrap_select(qualities, resamples=20, seed=42)
    second = bootstrap_select(qualities, resamples=20, seed=42)
    assert first == second
    assert first["winner"] == "diverse"


def test_guided_median_removes_isolated_noise_but_preserves_structural_detail():
    levels = torch.zeros((5, 5), dtype=torch.long)
    levels[1, 1] = 12
    levels[2, 2] = 14
    proxy = torch.zeros((5, 5))
    proxy[2, 2] = 1.0
    regularized = guided_median_levels(levels, proxy, detail_fraction=0.04)
    assert int(regularized[1, 1]) == 0
    assert int(regularized[2, 2]) == 14
    before = map_quality(levels, proxy, n_levels=16)
    after = map_quality(regularized, proxy, n_levels=16)
    assert after.spatial_coherence > before.spatial_coherence
