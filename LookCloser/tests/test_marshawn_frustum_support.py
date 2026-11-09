import sys,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_marshawn_frustum_support import camera_planes,project,points_inside,cells_possible,grid_support,conservative_axial_near


def camera():
 return dict(transform_matrix=np.eye(4).tolist(),fl_x=100.,fl_y=80.,cx=50.,cy=40.,w=100,h=80,physical_camera='synthetic')


def test_opengl_depth_image_y_and_halfpixel_projection():
 c=camera();uv,depth=project(np.array([[0,0,-1],[.005,-.00625,-1],[0,0,1],[0,0,0]]),c)
 assert np.array_equal(uv[0],[50,40])
 assert np.allclose(uv[1],[50.5,40.5]) # Raster index50,40 center coordinates.
 assert np.array_equal(depth,[1,1,-1,0])and np.isnan(uv[2:]).all()
 assert np.array_equal(points_inside(np.array([[0,0,-1],[0,0,1]]),camera_planes(c)),[True,False])


def test_exact_near_and_raster_boundaries_are_retained():
 c=camera();p=camera_planes(c)
 points=np.array([[0,0,-.052],[-.5,0,-1],[.5,0,-1],[0,.5,-1],[0,-.5,-1]])
 assert points_inside(points,p,tol=0).all()
 assert not points_inside(np.array([[0,0,-.051],[.501,0,-1],[0,.501,-1]]),p,tol=0).any()


def test_straddling_center_outside_not_false_discarded():
 p=camera_planes(camera());centers=np.array([[.55,0,-1],[.8,0,-1],[0,0,0]])
 assert not points_inside(centers,p).any()
 possible=cells_possible(centers,np.array([.1,.1,.1]),p)
 assert np.array_equal(possible,[True,False,True]) # Last box reaches beyond nearplane.


def test_parallel_face_and_touching_are_retained_without_division():
 p=camera_planes(camera())
 assert cells_possible(np.array([[0,0,-.002]]),np.array([0.,0.,.05]),p,tol=0)[0]
 assert not cells_possible(np.array([[0,0,0]]),np.array([.01,.01,.05]),p,tol=0)[0]
 assert np.isfinite(p).all()


def test_center_radius_matches_explicit_all8corner_plane_rejection():
 rng=np.random.default_rng(42);p=camera_planes(camera());center=rng.uniform(-2,2,(200,3));half=np.array([.07,.13,.09])
 corners=center[:,None,:]+np.array(list(itertools.product([-1,1],repeat=3)))[None]*half
 explicit=~((corners@p[:,:3].T+p[:,3]) < -1e-7).all(axis=1).any(axis=1)
 assert np.array_equal(cells_possible(center,half,p),explicit)


def test_rigid_rotated_camera_looks_in_world_positive_x():
 c=camera();c['transform_matrix']=np.array([[0,0,-1,2],[0,1,0,0],[1,0,0,0],[0,0,0,1]],float).tolist()
 # camera back points-X, hence forward+X. right points+Z.
 uv,d=project(np.array([[3,0,0],[3,0,.2],[1,0,0]]),c)
 assert np.allclose(uv[:2],[[50,40],[70,40]])and np.allclose(d,[1,1,-1])
 assert np.array_equal(points_inside(np.array([[3,0,0],[1,0,0]]),camera_planes(c)),[True,False])


def test_grid_union_is_monotonic_and_bounded_by_exact_corner_rejection():
 c=camera();back=camera();back['physical_camera']='opposite';M=np.eye(4);M[0,0]=-1;M[2,2]=-1;back['transform_matrix']=M.tolist()
 box=np.array([[-1,-1,-2],[1,1,2]],float);a,_=grid_support(box,[5,5,7],[c],chunk=11);b,_=grid_support(box,[5,5,7],[c,back],chunk=9)
 assert a.dtype==bool and a.shape==(1,5,5,7)and not(a&~b).any()and b.sum()>a.sum()


def test_native_euclidean_near_corner_ray_not_discarded_by_axial_plane():
 c=camera();direction=np.array([-.5,.5,-1.]);direction/=np.linalg.norm(direction);point=direction*.052
 assert np.isclose(np.linalg.norm(point),.052)
 assert not points_inside(point[None],camera_planes(c),tol=0)[0]
 assert points_inside(point[None],camera_planes(c,conservative_axial_near(c)),tol=1e-15)[0]
