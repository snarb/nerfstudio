import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from confidence_boundary_completion import solve_depth,grid_faces

def test_untrusted_boundary_does_not_impose_false_depth():
    domain=np.ones((7,7),bool);model=np.full((7,7),.1);old=model.copy();old[3,0]=.13
    trusted=np.zeros_like(domain);trusted[0,[0,3,6]]=True
    solved,stats=solve_depth(domain,model,old,trusted)
    assert np.allclose(solved,.1) and stats['measured_pins']==3
    assert np.array_equal(solved[trusted],old[trusted])

def test_measured_boundary_is_hard_constraint_and_residual_is_bounded():
    domain=np.ones((7,7),bool);model=np.full((7,7),.1);old=model.copy();trusted=np.zeros_like(domain)
    trusted[0,[0,3,6]]=True;old[trusted]=.102
    solved,_=solve_depth(domain,model,old,trusted)
    assert np.array_equal(solved[trusted],old[trusted])
    assert solved.min()>=.1 and solved.max()<=.102
    assert solved[-1,3]<solved[1,3]

def test_insufficient_pins_rejected():
    domain=np.ones((3,3),bool)
    with pytest.raises(ValueError):solve_depth(domain,np.ones((3,3)),np.ones((3,3)),np.zeros((3,3),bool))

def test_grid_assembly_recovers_previously_clipped_cell_before_extent_check():
    domain=np.ones((2,2),bool);active=domain.copy();index=np.arange(4).reshape(2,2)
    assert np.array_equal(grid_faces(domain,active,index),[[0,1,2],[1,3,2]])
