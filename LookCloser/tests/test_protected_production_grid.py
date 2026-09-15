import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from protected_production_grid import grid_plan,protected_removal


def test_only_missing_production_pixels_receive_prior_depth():
    domain=np.ones((3,3),bool);old=np.ones((3,3));old[1,1]=0;prior=np.full((3,3),2.)
    active,grid,depth=grid_plan(domain,prior,old)
    assert active.sum()==1 and active[1,1] and grid.sum()==5
    np.testing.assert_array_equal(depth[grid&~active],old[grid&~active]);assert depth[1,1]==2


def test_complete_original_surface_cannot_be_replaced():
    active,grid,depth=grid_plan(np.ones((2,2),bool),np.full((2,2),3.),np.ones((2,2)))
    assert not active.any() and not grid.any()
    np.testing.assert_array_equal(depth,np.ones((2,2)))


def test_prefix_preserved_even_if_all_faces_selected_for_removal():
    original=np.ones(7,bool);removed=protected_removal(original,5)
    np.testing.assert_array_equal(removed,[False]*5+[True]*2);assert original.all()


def test_outside_domain_hole_not_filled_and_invalid_depth_rejected():
    domain=np.zeros((3,3),bool);domain[1,1]=True
    active,grid,_=grid_plan(domain,np.ones((3,3)),np.zeros((3,3)))
    assert active.sum()==grid.sum()==1
    with pytest.raises(ValueError):grid_plan(domain,np.zeros((3,3)),np.zeros((3,3)))
