import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from boundary_cycle_blocks import cyclic_boundary_blocks


def test_articulation_does_not_hide_simple_contours():
    # Two triangular disks touch at one vertex: degree four in whole graph.
    t=np.array([[0,1,2],[0,3,4]])
    loops,stats=cyclic_boundary_blocks(t)
    assert {frozenset(c) for c in loops}=={frozenset((0,1,2)),frozenset((0,3,4))}
    assert stats['complex_blocks']==0
    assert np.array_equal(t,[[0,1,2],[0,3,4]])


def test_shared_interior_edge_excluded():
    loops,stats=cyclic_boundary_blocks(np.array([[0,1,2],[0,2,3]]))
    assert len(loops)==1 and set(loops[0])=={0,1,2,3}
    assert stats['boundary_edges']==4


def test_closed_tetrahedron_has_no_boundary():
    loops,stats=cyclic_boundary_blocks(np.array([[0,1,2],[0,3,1],[1,3,2],[0,2,3]]))
    assert not loops and stats['boundary_edges']==0
