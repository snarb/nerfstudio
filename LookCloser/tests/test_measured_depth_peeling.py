import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_measured_depth_peeling import nearest_admitted


def test_nearest_is_per_ray_not_input_order_and_rejects_invalid():
    rays=np.array([1,0,1,0,2,2]);depth=np.array([3.,5.,2.,1.,np.inf,-1.])
    good=np.array([True,True,True,False,True,True])
    assert nearest_admitted(rays,depth,good).tolist()==[1,2]


def test_no_admitted_intersection_never_manufactures_a_fill():
    assert nearest_admitted(np.array([0]),np.array([1.]),np.array([False])).size==0
    with pytest.raises(ValueError):nearest_admitted(np.array([0,1]),np.array([1.]),np.array([True]))
