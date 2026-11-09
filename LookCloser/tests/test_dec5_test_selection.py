from pathlib import Path
import json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from score_dec5_hair import validate_test_selection
from run_dec5_hair_campaign import sha


def test_test_rgb_stays_locked_until_visual_selection_and_immutable_metrics(tmp_path):
    metrics=tmp_path/'validation.json';metrics.write_text('{}')
    lock=tmp_path/'selection.json'
    record=dict(baseline=str(metrics),chosen_metrics=str(metrics),chosen_metrics_sha256=sha(metrics),independent_test_rgb_used=False,status='validation selection; repeat and visual path review still required')
    lock.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='visual selection'):validate_test_selection(lock)
    record['status']='validation and visual review locked before test RGB';lock.write_text(json.dumps(record))
    assert validate_test_selection(lock)==sha(lock)
    metrics.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed after selection'):validate_test_selection(lock)
