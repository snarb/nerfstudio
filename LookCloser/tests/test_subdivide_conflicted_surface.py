import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from subdivide_conflicted_surface import subdivide, verify_coverage


@pytest.mark.parametrize('bits', range(8))
def test_all_edge_patterns_conforming_and_oriented(bits):
    # Central triangle with three independent neighbours marking each shared edge.
    v = np.array([[0,0,0], [2,0,0], [0,2,0], [1,-1,0], [2,2,0], [-1,1,0]], float)
    t = np.array([[0,1,2], [1,0,3], [2,1,4], [0,2,5]])
    select = np.array([False, bool(bits&1), bool(bits&2), bool(bits&4)])
    nv, nt, parent = subdivide(v,t,select)
    verify_coverage(v,t,nv,nt,parent)
    assert (parent == 0).sum() == 1+bits.bit_count()
    # No remaining edge can contain a generated vertex in its strict interior.
    for a,b in np.unique(np.sort(nt[:, [[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0):
        edge=nv[b]-nv[a]; delta=nv[len(v):]-nv[a]
        along=delta@edge/(edge@edge)
        distance=np.linalg.norm(delta-along[:,None]*edge,axis=1)
        assert not ((distance<1e-12)&(along>1e-10)&(along<1-1e-10)).any()


def test_two_rounds_keep_original_ancestry_and_surface():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float)
    t=np.array([[0,1,2],[1,3,2]])
    nv,nt,p=subdivide(v,t,np.array([True,False]))
    nv,nt,p=subdivide(nv,nt,p==0,p)
    verify_coverage(v,t,nv,nt,p)
    assert (p==0).sum()==16
    assert (p==1).sum()>2


def test_invalid_selection_and_coverage_fail_closed():
    v=np.eye(3); t=np.array([[0,1,2]])
    with pytest.raises(ValueError):subdivide(v,t,np.array([1]))
    nv,nt,p=subdivide(v,t,np.array([True]))
    with pytest.raises(ValueError):verify_coverage(v,t,nv,nt[:,::-1],p)
