import sys
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mesh_contact_clearance import contact_constraints,CLEARANCE
from constrained_surface_step import lower_bound_lsq
from guard_mhr_anatomical_correction import all_pairs


def test_coplanar_in_plane_separation_survives_motion():
    v=np.array([[0.,0,0],[1.,0,0],[0,1,0],[2.,0,0],[3.,0,0],[2.,1,0]])*.0002
    t=np.array([[0,1,2],[3,4,5]]);ids=np.array([3,4,5])
    a,b,r=contact_constraints(v,t,{(0,1)},ids)
    desired=np.tile([-.0003,0,0],3)
    x,_=lower_bound_lsq(sparse.eye(9),desired,a,b)
    trial=v.copy();trial[ids]+=x.reshape(-1,3)
    assert not all_pairs(trial,t)
    axis=np.array(r[0]['axis'])
    assert (trial[t[1]]@axis).min()-(trial[t[0]]@axis).max()>=CLEARANCE-1e-12


def test_parallel_surface_gap_has_correct_sign():
    v=np.array([[0.,0,0],[1.,0,0],[0,1,0],[0,0,1],[1,0,1],[0,1,1]])*.0002
    t=np.array([[0,1,2],[3,4,5]])
    a,b,r=contact_constraints(v,t,{(0,1)},[3,4,5])
    x,_=lower_bound_lsq(sparse.eye(9),np.tile([0,0,-.0003],3),a,b)
    np.testing.assert_allclose(x.reshape(-1,3)[:,2],CLEARANCE-.0002,atol=1e-12)


def test_fixed_touching_triangles_fail_closed():
    v=np.array([[0.,0,0],[1.,0,0],[0,1,0],[1.,0,0],[2.,0,0],[1,1,0]])*.0002
    with pytest.raises(ValueError,match='Fixed vertices'):
        contact_constraints(v,np.array([[0,1,2],[3,4,5]]),{(0,1)},[])
