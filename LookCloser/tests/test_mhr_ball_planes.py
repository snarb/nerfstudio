import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_mhr_bounded_contact_correction import ball_planes


def test_support_plane_contains_entire_ball():
    base=np.zeros((2,3));current=np.array([[.0005,0,0],[0,0,0.]])
    n=np.array([1.,2.,3.]);n/=np.linalg.norm(n)
    a,b=ball_planes(current,base,np.arange(2),[(0,n)],.001)
    rng=np.random.default_rng(4)
    for _ in range(100):
        point=rng.normal(size=3);point*=.001/np.linalg.norm(point)
        step=np.r_[point-current[0],np.zeros(3)]
        assert (a@step>=b-1e-14).all()
    outside=np.r_[.0011*n-current[0],np.zeros(3)]
    assert (a@outside<b).all()
