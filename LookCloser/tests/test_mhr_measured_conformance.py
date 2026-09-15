import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from conform_mhr_measured_surface import uniform_laplacian,barycentric_matrix


def test_laplacian_constant_and_boundary_pin():
    tri=np.array([[0,1,2],[0,2,3]])
    lap=uniform_laplacian(tri,4)
    np.testing.assert_allclose(lap@np.ones(4),0,atol=1e-15)
    # Pinned outside vertex remains a zero displacement, not a removed boundary.
    reduced=lap[[0,1,2]][:,[0,1,2]]
    np.testing.assert_allclose(reduced@np.ones(3),(lap@np.array([1,1,1,0]))[:3])


def test_barycentric_correspondence_and_validation():
    tri=np.array([[0,1,2],[1,2,3]])
    bary=np.array([[.2,.3,.5],[0.,0.,1.]])
    matrix=barycentric_matrix(tri,bary,4)
    np.testing.assert_allclose(matrix@np.arange(4),[1.3,3.])
    with pytest.raises(ValueError):barycentric_matrix(tri,bary*2,4)
    with pytest.raises(ValueError):barycentric_matrix(tri,np.full_like(bary,np.nan),4)
