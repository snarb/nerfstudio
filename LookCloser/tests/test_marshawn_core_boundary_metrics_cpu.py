import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_marshawn_core_boundary_metrics_cpu import regions


def test_fixed_partition_and_exterior_edge_erosion():
    m=np.ones((25,25),bool);r=regions(m)
    assert r['core_gt8px'].sum()==9*9
    assert not (r['core_gt8px']&r['boundary_le8px']).any()
    np.testing.assert_array_equal(r['core_gt8px']|r['boundary_le8px'],m)


def test_euclidean_diagonal_not_manhattan_iterations():
    m=np.ones((40,40),bool);m[20,20]=False;r=regions(m)
    assert r['core_gt8px'][26,26]  # sqrt72>8, not L1-radius interpretation.
    assert not r['core_gt8px'][25,25]
