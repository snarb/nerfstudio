import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from observed_ring_curvature import boundary_curve_vertices
from study_confidence_depth_prior import unproject,project_integer


def test_unknown_mask_boundary_does_not_flatten_the_shape():
    camera=dict(transform_matrix=np.eye(4).tolist(),fl_x=100,fl_y=100,cx=50,cy=50)
    accepted=np.zeros((100,100),bool);accepted[20:80,20:80]=True
    md=np.zeros((100,100));md[19,20:80]=1
    xy=np.array([[25,19],[20,70],[40,21]])
    v=unproject(camera,xy[:,0],xy[:,1],np.ones(3))
    fit=dict(reference_center=[50,50],all_camera_coefficients=[0,0,1.001,0,0,0])
    out,r=boundary_curve_vertices(v,0,camera,fit,np.arange(3),accepted,md)
    np.testing.assert_array_equal(out[0],v[0])
    _,z=project_integer(camera,out)
    np.testing.assert_allclose(z[1],1/1.001,atol=1e-7)
    assert z[1]<z[2]<1 and r['boundary_ring_exact']


def test_no_old_depth_ring_means_no_artificial_planar_constraint():
    camera=dict(transform_matrix=np.eye(4).tolist(),fl_x=100,fl_y=100,cx=50,cy=50)
    accepted=np.zeros((100,100),bool);accepted[20:80,20:80]=True
    v=unproject(camera,np.array([20]),np.array([70]),np.ones(1))
    fit=dict(reference_center=[50,50],all_camera_coefficients=[0,0,1.001,0,0,0])
    out,r=boundary_curve_vertices(v,0,camera,fit,[0],accepted,np.zeros((100,100)))
    np.testing.assert_allclose(project_integer(camera,out)[1],[1/1.001],atol=1e-7)
    assert r['observed_ring_pixels']==0
