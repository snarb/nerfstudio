import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from calibrated_stereo_rectification import portrait_calibration,rectify_pair,rectify_local_pair,disparity_to_world


def camera(position):
    pose=np.eye(4);pose[:3,3]=position
    return dict(transform_matrix=pose.tolist(),fl_x=800.,fl_y=820.,cx=320.,cy=180.,w=640,h=360)


def project(x,k,e):
    p=x@e[:3,:3].T+e[:3,3];h=p@k.T
    return h[:,:2]/h[:,2:]


def test_native_to_rotated_pixel_centers():
    row=camera([0,0,0]);k,e=portrait_calibration(row)
    x=np.array([[.1,.2,-2.],[-.1,.1,-3.]])
    uv=project(x,k,e)
    native=np.column_stack((800*x[:,0]/-x[:,2]+319.5,-820*x[:,1]/-x[:,2]+179.5))
    np.testing.assert_allclose(uv,np.column_stack((native[:,1],639-native[:,0])),atol=1e-12)


def test_positive_disparity_recovers_exact_world_coordinates():
    a=camera([0,0,0]);b=camera([0,-.1,0])
    q,_=rectify_pair(a,b);x=np.array([[.1,.2,-2.],[-.1,.1,-3.]])
    e1=q['rectified_extrinsic'];e2=np.eye(4)
    e2[:3,:3]=q['R2']@q['E2'][:3,:3];e2[:3,3]=q['R2']@q['E2'][:3,3]
    p1=project(x,q['P1'][:,:3],e1);p2=project(x,q['P2'][:,:3],e2)
    np.testing.assert_allclose(p1[:,1],p2[:,1],atol=1e-10)
    actual=disparity_to_world(p1[:,0],p1[:,1],p1[:,0]-p2[:,0],q['P1'][:,:3],e1,q['baseline'])
    np.testing.assert_allclose(actual,x,atol=1e-10)
    with pytest.raises(ValueError):rectify_pair(b,a)


def test_distorted_input_fails_closed():
    r=camera([0,0,0]);r['k1']=.01
    with pytest.raises(ValueError):portrait_calibration(r)


def test_local_principal_offset_preserves_metric_depth():
    a=camera([0,0,0]);b=camera([0,-.1,0]);focus=np.array([.1,.2,-2.])
    q,_=rectify_local_pair(a,b,focus)
    e1=q['rectified_extrinsic'];e2=np.eye(4)
    e2[:3,:3]=q['R2']@q['E2'][:3,:3];e2[:3,3]=q['R2']@q['E2'][:3,3]
    p1=project(focus[None],q['P1'][:,:3],e1);p2=project(focus[None],q['P2'][:,:3],e2)
    np.testing.assert_allclose(p1,[[383.5,383.5]],atol=1e-10)
    np.testing.assert_allclose(p1[:,0]-p2[:,0],128,atol=1e-10)
    actual=disparity_to_world(p1[:,0],p1[:,1],p1[:,0]-p2[:,0],q['P1'][:,:3],e1,q['baseline'],q['disparity_offset'])
    np.testing.assert_allclose(actual,focus[None],atol=1e-10)
