import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from forearm_quadric_rays import world_quadric,world_plane,intersect_near_plane
from study_confidence_depth_prior import unproject


def test_same_camera_recovers_exact_fitted_depth():
    pose=np.eye(4);pose[:3,3]=[.1,.2,.3]
    camera=dict(transform_matrix=pose.tolist(),fl_x=900,fl_y=950,cx=960,cy=540)
    fit=dict(reference_center=[950,530],all_camera_coefficients=[.02,-.03,1.3,.001,.002,.003])
    uv=np.array([[930,540],[980,500],[900,560]],float);q=(uv-fit['reference_center'])/100
    design=np.column_stack([q,np.ones(len(q)),q[:,0]**2,q[:,0]*q[:,1],q[:,1]**2]);z=1/(design@fit['all_camera_coefficients'])
    origin=pose[:3,3];directions=unproject(camera,uv[:,0],uv[:,1],np.ones(len(uv)))-origin
    quad=world_quadric(camera,fit);plane=world_plane(camera,[0,0,1.3])
    actual,_=intersect_near_plane(origin,directions,quad,plane)
    np.testing.assert_allclose(actual,z,rtol=1e-12)
    points=unproject(camera,uv[:,0],uv[:,1],z);h=np.column_stack([points,np.ones(len(points))])
    np.testing.assert_allclose(np.einsum('ni,ij,nj->n',h,quad,h),0,atol=1e-12)


def test_other_camera_intersects_same_world_surface():
    camera=dict(transform_matrix=np.eye(4).tolist(),fl_x=100,fl_y=100,cx=0,cy=0)
    fit=dict(reference_center=[0,0],all_camera_coefficients=[0,0,.5,0,0,0])
    center=np.array([.2,0,0]);directions=np.array([[0,0,-1],[.1,.1,-1]])
    z,_=intersect_near_plane(center,directions,world_quadric(camera,fit),world_plane(camera,[0,0,.5]))
    np.testing.assert_allclose(z,[2,2])


def test_parallel_ray_does_not_invent_a_hit():
    quad=np.zeros((4,4));quad[2,3]=quad[3,2]=.5;quad[3,3]=1
    z,_=intersect_near_plane(np.zeros(3),np.array([[1,0,0.]]),quad,np.array([0,0,1,1.]))
    assert np.isnan(z[0])


def test_rotated_second_camera_recovers_points_on_curved_surface():
    a=.3;r=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    pose=np.eye(4);pose[:3,:3]=r;pose[:3,3]=[.1,-.2,.3]
    camera=dict(transform_matrix=pose.tolist(),fl_x=900,fl_y=950,cx=960,cy=540)
    fit=dict(reference_center=[960,540],all_camera_coefficients=[.02,-.03,1.3,.01,.002,.003])
    uv=np.array([[900,500],[1000,590],[940,540]],float);q=(uv-fit['reference_center'])/100
    z=1/(np.column_stack([q,np.ones(3),q[:,0]**2,q[:,0]*q[:,1],q[:,1]**2])@fit['all_camera_coefficients'])
    points=unproject(camera,uv[:,0],uv[:,1],z)
    center=np.array([-.1,.05,.25]);b=-.15
    other_rotation=np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
    expected=-((points-center)@other_rotation)[:,2]
    directions=(points-center)/expected[:,None]
    actual,_=intersect_near_plane(center,directions,world_quadric(camera,fit),world_plane(camera,[0,0,1.3]))
    np.testing.assert_allclose(actual,expected,rtol=1e-12)
    np.testing.assert_allclose(center+actual[:,None]*directions,points,atol=1e-12)


def test_no_real_intersection_and_behind_plane_rejected():
    # Unit sphere, first ray misses it; second points away from the reference plane.
    quad=np.diag([1.,1.,1.,-1.]);plane=np.array([0.,0.,1.,0.])
    depth,_=intersect_near_plane(np.array([0.,0.,2.]),np.array([[1.,0.,-.1],[0.,0.,1.]]),quad,plane)
    assert np.isnan(depth).all()


def test_empty_ray_batch_is_valid():
    depth,reference=intersect_near_plane(np.zeros(3),np.empty((0,3)),np.eye(4),np.ones(4))
    assert depth.shape==reference.shape==(0,)
