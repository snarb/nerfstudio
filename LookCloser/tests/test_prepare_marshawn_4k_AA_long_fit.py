import importlib.util
from pathlib import Path
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/prepare_marshawn_4k_AA_long_fit.py'
    spec = importlib.util.spec_from_file_location('long_fit', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_only_budget_changes_at_all_three_consumers():
    source = (Path(__file__).resolve().parents[1] / 'scripts/fit_marshawn_4k_AA_frequency.py').read_text()
    output = module().extend_worker(source)
    assert output.replace('4000', '1000') == source
    assert output.count('4000') == 3
    compile(output, 'frozen_long_fit', 'exec')


def test_refuses_worker_drift():
    with pytest.raises(ValueError, match='recipe/config/metadata'):
        module().extend_worker("train_steps_per_level=1000")


def test_16000_control_preserves_all_other_source():
    source = (Path(__file__).resolve().parents[1] / 'scripts/fit_marshawn_4k_AA_frequency.py').read_text()
    m = module()
    four = m.extend_worker(source)
    sixteen = m.extend_worker(four, 16000, 4000)
    assert sixteen.replace('16000', '4000') == four
    with pytest.raises(ValueError, match='transitions'):
        m.extend_worker(source, 99999)
