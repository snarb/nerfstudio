import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_mhr_conformance_crossings import transverse_crossings

def test_strict_transverse_intersection():
    a=np.array([[[0.,0,0],[2,0,0],[0,2,0]]]);b=np.array([[[.5,.5,-1],[.5,.5,1],[1.5,.5,1]]])
    assert transverse_crossings(a,b).tolist()==[True]

def test_disjoint_or_coplanar_contact_is_not_transverse():
    a=np.array([[[0.,0,0],[2,0,0],[0,2,0]]]);assert not transverse_crossings(a,a+[0,0,1])[0]
    assert not transverse_crossings(a,a)[0]

def test_large_rigid_normal_rotation_is_not_itself_self_intersection():
    a=np.array([[[0.,0,0],[2,0,0],[0,2,0]]]);b=a*np.array([1,-1,-1])+[0,0,3]
    assert np.dot(np.cross(a[0,1]-a[0,0],a[0,2]-a[0,0]),np.cross(b[0,1]-b[0,0],b[0,2]-b[0,0]))<0
    assert not transverse_crossings(a,b)[0]
