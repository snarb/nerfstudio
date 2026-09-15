from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cinematic_wide_spiral import spiral_parameters


def test_broad_turn_precedes_contraction_and_exact_hold():
    u,angle,radius,xy,settle=spiral_parameters(np.arange(150))
    first=np.flatnonzero(u>=5/6)[0]
    assert first<100 and angle[0]-angle[first]>=2*np.pi
    assert radius[first]>.7
    assert np.ptp(xy[:first+1,0])>6.5 and np.ptp(xy[:first+1,1])>1.5
    np.testing.assert_allclose(xy[118:],0,atol=1e-14);np.testing.assert_allclose(radius[118:],0,atol=1e-14)
    np.testing.assert_allclose(settle[118:],1,atol=1e-14,rtol=0)


def test_curve_smooth_at_late_join_and_remains_inside_interior_rows():
    t=np.linspace(0,150,15001);_,_,_,xy,_=spiral_parameters(t)
    assert xy[:,0].min()>-5 and xy[:,0].max()<3
    assert xy[:,1].min()>-.95 and xy[:,1].max()<.95
    assert np.linalg.norm(np.diff(xy,axis=0),axis=1).max()<.006
    h=1e-3;v=spiral_parameters(np.array([118-h,118,118+h]))[3]
    assert np.linalg.norm((v[2]-v[0])/(2*h))<1e-6
