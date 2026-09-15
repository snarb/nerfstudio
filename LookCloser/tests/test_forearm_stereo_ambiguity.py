import numpy as np
from diagnose_forearm_stereo_ambiguity import ncc, patch


def test_ncc_mean_centering_and_degenerate_inputs():
    a = np.arange(27).reshape(9, 3)
    assert np.isclose(ncc(a, a*2+[3, 5, 7]), 1)
    assert np.isclose(ncc(a, -a), -1)
    assert np.isnan(ncc(a, np.ones_like(a)))
    b = a.astype(float)
    b[0, 0] = np.nan
    assert np.isnan(ncc(a, b))


def test_patch_subpixel_and_array_boundary():
    y, x = np.mgrid[:8, :8]
    im = np.stack([x, y, x+y], axis=-1).astype(float)
    p = patch(im, [3.5, 4.5], 1)
    np.testing.assert_allclose(p[4], [3.5, 4.5, 8.])
    assert np.isnan(patch(im, [0, 0], 1)).any()
