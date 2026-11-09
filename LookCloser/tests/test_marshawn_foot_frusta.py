import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_marshawn_feet_aabb import region_frustum_envelope


def test_independent_opposed_cones_and_padding():
    matrices = {'a': np.array([[1.,0,0,0],[0,1.,0,0],[0,0,1.,0]]),
                'b': np.array([[-1.,0,0,0],[0,1.,0,0],[0,0,-1.,4.]])}
    regions = {k: [-.5,-.5,.5,.5] for k in matrices}
    result = region_frustum_envelope(regions, matrices)
    np.testing.assert_allclose(result, [[-1,-1,1e-6],[1,1,4-1e-6]], atol=1e-8)
    padded = region_frustum_envelope(regions, matrices, .1)
    np.testing.assert_allclose(padded[:,:2], [[-1.2,-1.2],[1.2,1.2]], atol=1e-8)
    assert (padded[0] <= result[0]+1e-9).all()
    assert (padded[1] >= result[1]-1e-9).all()


def test_unbounded_single_camera_does_not_borrow_aabb():
    matrices = {'a': np.array([[1.,0,0,0],[0,1.,0,0],[0,0,1.,0]])}
    with pytest.raises(ValueError, match='unbounded'):
        region_frustum_envelope({'a': [-.5,-.5,.5,.5]}, matrices)


@pytest.mark.parametrize('padding', [-1, float('nan')])
def test_invalid_padding_fails(padding):
    with pytest.raises(ValueError, match='padding'):
        region_frustum_envelope({}, {}, padding)
