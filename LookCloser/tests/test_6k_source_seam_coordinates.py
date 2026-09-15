import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_6k_source_seams import boundaries


def test_portrait_crop_maps_to_landscape_crop():
    raw=np.arange(15*23).reshape(15,23)
    x0,y0,x1,y1=3,7,10,19
    actual=np.rot90(raw[x0:x1,raw.shape[1]-y1:raw.shape[1]-y0])
    np.testing.assert_array_equal(actual,np.rot90(raw)[y0:y1,x0:x1])


def test_source_boundaries_mark_both_sides_without_frame_border():
    ids=np.array([[2,2,3,3],[2,2,3,3]])
    expected=np.array([[False,True,True,False],[False,True,True,False]])
    np.testing.assert_array_equal(boundaries(ids),expected)
    assert not boundaries(np.ones((3,4))).any()
