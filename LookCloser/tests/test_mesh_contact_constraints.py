import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mesh_contact_constraints import contact_constraints
from constrained_surface_step import lower_bound_lsq
from scipy import sparse


def test_contact_step_preserves_triangle_order():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                [0.,0.,.1],[1.,0.,.1],[0.,1.,.1]])
    t=np.array([[0,1,2],[3,4,5]]);ids=np.arange(6)
    a,b,records=contact_constraints(v,t,[(0,1)],ids)
    desired=np.zeros_like(v);desired[3:,2]=-.2
    x,_=lower_bound_lsq(sparse.eye(18),desired.ravel(),a,b)
    result=v+x.reshape(-1,3);axis=np.array(records[0]['axis'])
    assert (result[3:]@axis).min()>=(result[:3]@axis).max()-1e-10
    assert np.all(a@x>=b-1e-10)


def test_unchanged_step_is_feasible_with_fixed_triangle():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                [0.,0.,.1],[1.,0.,.1],[0.,1.,.1]])
    a,b,_=contact_constraints(v,np.array([[0,1,2],[3,4,5]]),[(0,1)],np.array([3,4,5]))
    assert a.shape==(9,9);assert (b<=0).all()
