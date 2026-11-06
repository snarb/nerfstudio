import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_semantic_cell_admission import WholeCellAdmission


def test_whole_cell_retains_corner_support_that_center_only_misses():
    grid=np.zeros((4,4,4),bool);grid[0,0,0]=True
    gate=WholeCellAdmission(grid,[[0,0,0],[1,1,1]],epsilon=0.)
    lo=np.array([[.20,.20,.20]]);hi=np.array([[.60,.60,.60]])
    assert gate.boxes_admitted(lo,hi)[0]
    assert not gate.point_admitted((lo+hi)/2)[0]


def test_closed_contact_outside_unknown_and_xyz_anisotropy():
    grid=np.zeros((2,3,4),bool);grid[1,0,3]=True
    gate=WholeCellAdmission(grid,[[1,2,3],[3,8,15]],epsilon=1e-10)
    assert gate.boxes_admitted([[1.9,2.1,12.1]],[[2.,3.,13.]])[0]
    assert not gate.point_admitted([[1.4,6.,5.]])[0]
    assert gate.boxes_admitted([[.99,3,5]],[[1.01,4,6]])[0]
    assert gate.point_admitted([[2.5,3.,13.]])[0]


def test_summed_volume_matches_independent_box_overlap_enumeration():
    rng=np.random.default_rng(42);grid=rng.random((5,6,7))>.8;bounds=np.array([[-.84,-.21,-.07],[.15,.42,.63]])
    gate=WholeCellAdmission(grid,bounds,epsilon=0.);lo=rng.uniform(bounds[0],bounds[1],(100,3));hi=np.minimum(bounds[1],lo+rng.random((100,3))*.15)
    ijk=np.argwhere(grid);pitch=(bounds[1]-bounds[0])/grid.shape;source_lo=bounds[0]+ijk*pitch;source_hi=source_lo+pitch
    expected=np.array([np.any(np.all((source_hi>=a)&(source_lo<=b),axis=1))for a,b in zip(lo,hi)])
    assert np.array_equal(gate.boxes_admitted(lo,hi),expected)


def test_full_mapping_covers_all_allowed_source_cell_centers_and_unknown():
    grid=np.zeros((3,4,5),bool);grid[0,2,4]=True;grid[2,1,1]=True;bounds=np.array([[0,0,0],[1,2,3]])
    gate=WholeCellAdmission(grid,bounds);mapped=gate.map_target_grid([8,9,10],bounds,chunk=17)
    target=WholeCellAdmission(mapped,bounds)
    points=bounds[0]+(np.argwhere(grid)+.5)/np.array(grid.shape)*(bounds[1]-bounds[0])
    assert target.point_admitted(points).all()
    assert WholeCellAdmission(np.zeros((2,2,2),bool),bounds).point_admitted([[2,3,4]])[0]
    with pytest.raises(ValueError):gate.boxes_admitted([[0,0,0]],[[-1,0,0]])
