import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mhr_surface_silhouette as base
import mhr_dense_surface_silhouette as dense
import run_mhr_dense_sampling_control as control


@pytest.mark.parametrize('order,count', [(2,4), (4,13), (8,43)])
def test_nested_nonvertex_lattice(order, count):
    p = dense.barycentric_lattice(order)
    assert p.shape == (count, 3) and len(np.unique(p, axis=0)) == count
    assert (p >= 0).all() and (p.max(1) < 1).all()
    np.testing.assert_allclose(p.sum(1), 1)
    np.testing.assert_array_equal(p[:4], base.BARY)
    if order == 8:
        np.testing.assert_array_equal(p[:13], dense.barycentric_lattice(4))


def test_order2_exactly_replays_original_math_and_does_not_mutate_globals():
    v = np.array([[1., 0, -4.], [2., 0, -4.], [1., 1., -4.]])
    t = np.array([[0,1,2]]); active = np.array([True, True, False])
    row = dict(transform_matrix=np.eye(4).tolist(), fl_x=16., fl_y=16., cx=32., cy=32.)
    y,x = np.mgrid[:64,:64]; sdf = (x+.3*y-20).astype(np.float32)
    args = (v,t,active,[row],[sdf])
    a,b = base.linearize(*args); old = base.association
    aa,bb = dense.linearizer(2)(*args)
    np.testing.assert_array_equal(a.toarray(), aa.toarray()); np.testing.assert_array_equal(b,bb)
    assert base.association is old
    assert dense.linearizer(4)(*args)[0].shape == (13,6)


@pytest.mark.parametrize('order', [4,8])
def test_adapter_pins_executed_lattice_and_output(order):
    source, driver = control.sources(order)
    assert f'dense_helper.linearizer({order})' in source
    assert f'dec5_mhr_sampling_dense{order}' in driver
    assert 'DENSE_WRAPPER,DENSE_HELPER' in driver
    assert "'vertex_weight': 16" in source and "'additional_surface_weight': 16" in source


def test_invalid_order_rejected():
    with pytest.raises(ValueError): dense.linearizer(3)
    with pytest.raises(ValueError): control.sources(2)
