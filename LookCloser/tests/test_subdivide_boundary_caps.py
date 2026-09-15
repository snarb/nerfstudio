import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from subdivide_boundary_caps import subdivide


def test_planar_subdivision_preserves_perimeter_and_prefix():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float)
    t=np.array([[1,3,2]]);p=np.array([[0,1,2]])
    vv,pp,notes=subdivide(v,t,p,[dict(vertices=[0,1,2],triangles=1)])
    np.testing.assert_array_equal(vv[:4],v)
    np.testing.assert_allclose(vv[4],v[p[0]].mean(0))
    edges,counts=np.unique(np.sort(pp[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    assert set(map(tuple,edges[counts==1]))=={(0,1),(0,2),(1,2)}


def test_curved_fit_is_bounded_and_can_change_shape():
    xy=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[0,-2],[2,0],[0,2],[-2,0]],float)*.001
    v=np.column_stack([xy,100*(xy[:,0]**2+xy[:,1]**2)])
    t=np.array([[0,4,1],[1,5,2],[2,6,3],[3,7,0]])
    p=np.array([[0,1,2],[0,2,3]]);notes=[dict(vertices=[0,1,2,3],triangles=2)]
    vv,pp,result=subdivide(v,t,p,notes,curved=True)
    np.testing.assert_array_equal(vv[:len(v)],v)
    displacement=np.linalg.norm(vv[len(v):]-v[p].mean(1),axis=1)
    assert (displacement>0).all() and (displacement<=.0005).all() and result[0]['applied']
