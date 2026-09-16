import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recover_supported_front_surface import eligible


def test_every_geometry_color_and_support_gate_is_required():
    arguments=[True,True,True,0,True,True,3,True]
    assert eligible(*arguments)
    for i,bad in enumerate([False,False,False,1,False,False,2,False]):
        trial=arguments.copy();trial[i]=bad
        assert not eligible(*trial),i


def test_array_selection_preserves_already_colored_surface():
    yes=np.ones(3,bool)
    np.testing.assert_array_equal(eligible(np.array([True,False,True]),yes,yes,np.array([0,0,1]),
        yes,yes,np.full(3,10),yes),[True,False,False])
