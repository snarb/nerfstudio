import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from contour_guard_luster_hybrid import guard


def test_identical_prediction_preserves_every_pixel():
    base=np.random.default_rng(2).random((64,64,3),dtype=np.float32)
    weight,_,_=guard(base,base)
    assert np.array_equal(weight,np.ones((64,64),np.float32))


def test_shifted_coherent_edge_is_rejected_locally():
    base=np.zeros((96,96,3),np.float32);base[:,48:]=1
    shifted=np.zeros_like(base);shifted[:,52:]=1
    weight,_,_=guard(base,shifted,sigma=2,displacement=1)
    assert weight[:,46:52].mean()<.3
    assert np.all(weight[:,:24]==1) and np.all(weight[:,76:]==1)


def test_detail_on_flat_surface_has_no_invented_contour():
    base=np.full((96,96,3),.5,np.float32)
    detail=base+np.random.default_rng(3).uniform(-.1,.1,base.shape).astype(np.float32)
    weight,_,_=guard(base,detail)
    assert np.all(weight==1)
