import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from foreground_layer_evidence import rejects_far_layer


def test_only_clear_four_view_counterevidence_qualifies():
    a=np.ones((5,4),bool);inside=a.copy();known=a.copy();blue=a.copy()
    inside[1,0]=False;known[2,0]=False;blue[3,:3]=False
    result=rejects_far_layer([True,True,True,True,False],inside,a,blue,known)
    np.testing.assert_array_equal(result,[True,False,False,False,False])


def test_skin_to_skin_and_three_camera_claims_do_not_override():
    a=np.ones((1,4),bool)
    assert not rejects_far_layer([True],a,a,np.zeros((1,4),bool),a)[0]
    with pytest.raises(ValueError):rejects_far_layer([True],a[:,:3],a[:,:3],a[:,:3],a[:,:3])
