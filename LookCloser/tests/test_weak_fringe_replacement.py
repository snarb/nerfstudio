import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_weak_fringe_replacement import removable
from study_weak_fringe_replacement import background_witnesses
from audit_weak_fringe_replacement import integral_background


def test_any_supported_sample_protects_triangle():
    votes=np.array([[0,0,0,0],[0,1,1,0],[5,1,1,1],[0,0,2,0],[0,0,0,0]])
    np.testing.assert_array_equal(removable(votes,[6,20,48,30,5]),[True,True,False,False,False])


def test_integral_and_maximum_filter_background_agree():
    row=dict(physical_camera='a',transform_matrix=np.eye(4).tolist(),fl_x=1.,fl_y=1.,cx=960.,cy=540.)
    masks=np.zeros((1,1080,1920),bool);masks[0,540,960]=True
    points=np.array([[[0.,0.,-1.]]*4,[[20.,0.,-1.]]*4])
    expected=np.array([[False,True]])
    np.testing.assert_array_equal(background_witnesses(points,[row],masks,['a']),expected)
    np.testing.assert_array_equal(integral_background(points,[row],masks,['a']),expected)
