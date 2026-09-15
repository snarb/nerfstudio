import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_mhr_silhouette_conformance import project_jacobian, sample_sdf


def test_project_analytic_world_jacobian():
    angle = .4
    pose = np.eye(4)
    pose[:3,:3] = [[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
    row = dict(transform_matrix=pose, fl_x=1000., fl_y=900., cx=960., cy=540.)
    points = np.array([[.2,.1,-2.],[-.1,.3,-1.5]])
    uv, z, jac = project_jacobian(points,row)
    for axis in range(3):
        step = np.zeros_like(points);step[:,axis] = 1e-6
        difference = (project_jacobian(points+step,row)[0]-project_jacobian(points-step,row)[0])/(2e-6)
        np.testing.assert_allclose(difference,jac[:,:,axis],rtol=1e-7,atol=1e-7)
    assert (z>0).all()


def test_bilinear_sdf_value_and_gradient():
    y,x = np.mgrid[:20,:30]
    grid = 2*x-3*y+.1*x*y
    uv = np.array([[4.3,5.4],[12.8,9.1]])
    value,gradient = sample_sdf(grid,uv)
    np.testing.assert_allclose(value,2*uv[:,0]-3*uv[:,1]+.1*uv[:,0]*uv[:,1])
    np.testing.assert_allclose(gradient,np.c_[2+.1*uv[:,1],-3+.1*uv[:,0]])


def test_renderer_half_pixel_convention():
    row = dict(transform_matrix=np.eye(4), fl_x=1000., fl_y=1000., cx=960., cy=540.)
    uv,_,_ = project_jacobian(np.array([[0.,0.,-1.]]),row)
    np.testing.assert_array_equal(uv,[[959.5,539.5]])
