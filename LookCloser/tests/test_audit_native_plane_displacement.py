import numpy as np
import pytest
from audit_native_plane_displacement import fit_native_plane, normal_displacement, displacement_consensus


def camera():
    return dict(fl_x=900.,fl_y=950.,cx=960.,cy=540.,transform_matrix=np.eye(4).tolist())


def test_exact_slanted_plane_and_pinhole_centers():
    f=camera(); yy,xx=np.mgrid[-2:3,-2:3];center=np.array([985,572])
    rays=np.stack(((xx+center[0]+.5-f['cx'])/f['fl_x'],
        (yy+center[1]+.5-f['cy'])/f['fl_y'],np.ones((5,5))),-1)
    n=np.array([.2,-.15,1.]); z=.7/(rays@n)
    got=fit_native_plane(z,center,f)
    assert got['valid'] and got['taps_used']==25
    expected=n*[1,-1,-1];expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(got['normal'],expected,atol=1e-11)
    assert got['offset']==pytest.approx(.7/np.linalg.norm(n))
    point=(rays[2,2]*z[2,2])*[1,-1,-1]+expected*.002
    move=normal_displacement(got,point,expected)
    assert move['valid'] and move['displacement']==pytest.approx(-.002)


def test_transformed_camera_plane():
    f=camera();f['transform_matrix']=[[0,-1,0,.2],[1,0,0,-.3],[0,0,1,.1],[0,0,0,1]]
    got=fit_native_plane(np.full((5,5),.7),[960,540],f)
    np.testing.assert_allclose(got['normal'],[0,0,-1],atol=1e-11)
    assert got['offset']==pytest.approx(.6)


@pytest.mark.parametrize('bad',[0.,np.nan,np.inf,-1.])
def test_missing_taps_never_create_plane(bad):
    z=np.full((5,5),.7);z.ravel()[:6]=bad
    assert not fit_native_plane(z,[960,540],camera())['valid']


def test_five_outliers_excluded_but_two_layers_rejected():
    z=np.full((5,5),.7);z[0]=1.2
    got=fit_native_plane(z,[960,540],camera())
    assert got['valid'] and got['taps_used']==20
    z[1]=1.2
    assert not fit_native_plane(z,[960,540],camera())['valid']


def test_wrong_normals_and_farther_layers_rejected():
    plane=dict(valid=True,normal=[0,0,1],offset=.7)
    assert not normal_displacement(plane,[0,0,.7],[1,0,0])['valid']
    assert not normal_displacement(plane,[0,0,.6],[0,0,1])['valid']
    with pytest.raises(ValueError):
        normal_displacement(plane,[0,0,.7],[0,0,2])


def test_consensus_requires_multiple_consistent_observations():
    assert not displacement_consensus([])['eligible']
    assert not displacement_consensus([0,0])['eligible']
    assert displacement_consensus([.002,.0021,.0019,-.001])['eligible']
    assert not displacement_consensus([-.003,-.002,-.001,.001,.002,.003])['eligible']


def test_consensus_is_order_invariant():
    values=[.002,.0021,.0019,-.001]
    assert displacement_consensus(values)==displacement_consensus(values[::-1])
