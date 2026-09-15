"""The opt-in adapter must alter executed math, not only recipe metadata."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_mhr_silhouette_weight_control as control


def test_generated_adapter_changes_math_and_records_weight():
    source, driver = control.sources()
    compile(source, '<anatomical-weight-control>', 'exec')
    compile(driver, '<clearance-weight-control>', 'exec')
    assert "source.replace(old_weight,'weights = np.sqrt(16.*robust/(len(rows)*count))/2.')" in source
    assert "dict(ns['RECIPE'],silhouette_weight=16.)" in source
    assert "recipe=dict(proof['recipe'],silhouette_weight=16.)" in source
    assert 'baseline_silhouette_weight=4.' in source


def test_control_uses_disjoint_root_and_hashes_own_wrapper():
    _, driver = control.sources()
    assert "ROOT=Path('/mnt/data/dec5_mhr_silhouette_weight16')" in driver
    assert "ROOT=Path('/mnt/data/dec5_mhr_clearance_correction')" not in driver
    assert 'WEIGHT_WRAPPER]' in driver
    assert 'source=ANATOMICAL_SOURCE' in driver


def test_changed_parent_source_fails_closed(monkeypatch):
    original = Path.read_text

    def changed(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == Path(control.anatomical.__file__):
            value = value.replace('width_from_existing_anchor_protocol=True,',
                                  'width_from_existing_anchor_protocol=False,')
        return value

    monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(AssertionError, match='width_from_existing_anchor_protocol'):
        control.sources()
