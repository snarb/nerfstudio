from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from soften_dec5_opacity_gate import blend


def test_soft_gate_preserves_endpoints_and_does_not_blow_out_white_gt_pixels():
    base=np.zeros((1,5,3),np.uint8);candidate=np.full_like(base,255)
    alpha=np.array([[0,.01,.055,.1,1]],np.float32)
    image,weight=blend(base,candidate,alpha)
    np.testing.assert_array_equal(image[0,:2],base[0,:2])
    np.testing.assert_array_equal(image[0,3:],candidate[0,3:])
    assert 186<=image[0,2,0]<=189
    assert np.isfinite(weight).all() and np.all(np.diff(weight[0])>=0)
