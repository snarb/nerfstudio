import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_forearm_plane_transfer_v2 import eligible_points


def test_unknown_does_not_veto_two_real_skin_views_but_disagreement_does():
    actual=eligible_points(np.ones(4),np.ones(4),np.array([3,2,1,2]),np.array([0,0,0,1]),np.zeros(4))
    assert actual.tolist()==[True,True,False,False]


def test_free_space_distance_and_positive_depth_are_hard_vetoes():
    actual=eligible_points(np.array([1,1,1,0,np.nan,-1]),np.array([100,101,1,1,1,1]),
        np.full(6,2),np.zeros(6),np.array([0,0,1,0,0,0]))
    assert actual.tolist()==[True,False,False,False,False,False]


def test_no_measured_interior_vote_is_needed_or_invented():
    # Interior measurement count is intentionally not an input to this prior gate.
    assert eligible_points(np.array([.6]),np.array([30]),np.array([2]),np.array([0]),np.array([0])).item()
