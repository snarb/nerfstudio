"""The ablation diagnostics must preserve COLMAP pixel coordinates and validity."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/summarize_patchmatch_source_count_depth.py'
spec=importlib.util.spec_from_file_location('source_count_depth',SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_native_depth_orientation(tmp_path):
    expected=np.arange(12,dtype=np.float32).reshape(3,4)
    path=tmp_path/'depth.bin'
    path.write_bytes(b'4&3&1&'+expected.tobytes())
    np.testing.assert_array_equal(module.read_array(path),expected)

def test_pinned_graph_stores_column_then_row(tmp_path):
    path=tmp_path/'graph.bin'
    # x=4 exceeds height=2, distinguishing the ordering even for sparse graphs.
    path.write_bytes(b'5&2&1&'+np.array([4,1,2,7,9],np.int32).tobytes())
    expected=np.zeros((2,5),np.int16);expected[1,4]=2
    np.testing.assert_array_equal(module.read_consistency_counts(path),expected)
    path.write_bytes(b'5&2&1&'+np.array([4,1,2,7],np.int32).tobytes())
    with pytest.raises(ValueError,match='Invalid consistency'):
        module.read_consistency_counts(path)

def test_lost_support_excludes_nonfinite_depth():
    photo=np.array([[1,2],[0,4.]])
    geometric=np.array([[1,0],[3,np.nan]])
    result=module.stats(photo,geometric,np.ones((2,2),bool))
    assert result['geometric_valid']==2
    assert result['photo_valid_geo_invalid']==2
    assert result['photo_invalid_geo_valid']==1
