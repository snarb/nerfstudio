import sys
from pathlib import Path
import inspect
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import study_mhr_zero_margin as study


def test_only_optimizer_offset_changes():
    source = inspect.getsource(study.FROZEN_OPTIMIZE)
    changed = study.replace_exact(source, [('excess = np.maximum(values-2, 0)', 'excess = np.maximum(values-0, 0)', 1)])
    assert changed.replace('values-0, 0', 'values-2, 0') == source
    assert 'weights = np.sqrt(4.*robust/(len(rows)*count))/2.' in changed
    assert 'excess[take]/8.' in changed
    assert 'current = base.copy()' in changed


def test_fail_closed_adapter():
    with pytest.raises(AssertionError): study.replace_exact('one', [('missing', 'new', 1)])


def test_frozen_stop_rule():
    assert study.continuation.STOP['maximum_outer_iterations'] == 100
    assert study.continuation.STOP['unconstrained_step_tolerance'] == 1e-5
    assert study.continuation.STOP['consecutive_small_steps'] == 3
