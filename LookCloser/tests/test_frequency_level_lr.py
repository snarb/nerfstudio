"""Numerical and configuration guards for the optional progressive fit schedule."""
import math

import pytest

from nerfstudio.scripts.lookcloser_preprocess import (
    LookCloserPreprocessConfig,
    per_level_learning_rates,
)


@pytest.mark.parametrize("steps", [2, 1000, 4000])
def test_budget_endpoints_and_constant_log_step(steps):
    rates = per_level_learning_rates(.01, .001, steps)
    assert len(rates) == steps
    assert rates[0] == .01 and rates[-1] == .001
    expected = math.log(.1) / (steps - 1)
    assert all(math.log(b / a) == pytest.approx(expected, abs=1e-15)
               for a, b in zip(rates, rates[1:]))


@pytest.mark.parametrize("final", [0, -.001, .02, float("nan"), float("inf")])
def test_invalid_rates_fail_before_fit(final):
    with pytest.raises(ValueError, match="learning rates"):
        LookCloserPreprocessConfig(lr_final_per_level=final)


def test_insufficient_level_budget_and_invalid_level_count():
    with pytest.raises(ValueError, match="two updates"):
        LookCloserPreprocessConfig(train_steps_per_level=1, lr_final_per_level=.001)
    with pytest.raises(ValueError, match="two updates"):
        LookCloserPreprocessConfig(steps_per_image=31, n_levels=16, lr_final_per_level=.001)
    with pytest.raises(ValueError, match="n_levels"):
        LookCloserPreprocessConfig(n_levels=0, lr_final_per_level=.001)


def test_default_retains_constant_rate_and_tiny_legacy_budget():
    config = LookCloserPreprocessConfig(train_steps_per_level=1)
    assert config.lr == .01 and config.lr_final_per_level is None
    assert per_level_learning_rates(.01, .01, 3) == [.01, .01, .01]
