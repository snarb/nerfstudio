import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_close_boundary_completion import transform


def test_only_minimum_centroid_gap_changes():
    source='minimum_centroid_distance=.00002\nkeep=(center_distance>=.00002)&(maximum_distance<=.003)&free_veto'
    changed=transform(source)
    assert changed=='minimum_centroid_distance=0.0\nkeep=(center_distance>=0.0)&(maximum_distance<=.003)&free_veto'


def test_unexpected_source_fails_closed():
    with pytest.raises(ValueError):transform('changed upstream')
