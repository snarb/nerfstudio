import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_foundation_foreground_patch import missing_foreground


def test_missing_foreground_includes_deeper_mesh_but_not_existing_surface():
    eligible=missing_foreground(np.array([True,True,True,True,False]),
        np.array([np.inf,.65,.601,.59,.65]),np.full(5,.6))
    np.testing.assert_array_equal(eligible,[True,True,False,False,False])


def test_invalid_prior_cannot_enter_geometry():
    np.testing.assert_array_equal(missing_foreground(np.ones(3,bool),np.full(3,np.inf),
        np.array([np.nan,-1,0])),np.zeros(3,bool))
