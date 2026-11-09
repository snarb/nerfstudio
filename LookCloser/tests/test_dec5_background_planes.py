import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_dec5_background_planes import triangulate,fit_planes
from dec5_hair import project
from dec5_plane_background import PlaneBackground
from audit_dec5_fresh_hair_geometry import multiview_point


def camera(x):
    pose=np.diag([1.,-1.,-1.,1.]);pose[0,3]=x
    return dict(transform_matrix=pose.tolist(),fl_x=800.,fl_y=800.,cx=640.,cy=480.)


def test_fixed_pose_triangulation_recovers_plane_and_rejects_epipolar_outlier():
    rng=np.random.RandomState(8);points=rng.uniform(-.5,.5,(40,3));points[:,2]=3.
    first,second=camera(0),camera(.4)
    a,_=project(points,first);b,_=project(points,second);b[-1,1]+=30
    descriptors=rng.normal(size=(40,128)).astype(np.float32)
    recovered,error=triangulate((a,descriptors),(b,descriptors),first,second)
    assert len(recovered)==39
    distances=np.linalg.norm(recovered[:,None]-points[None],axis=-1).min(-1)
    assert distances.max()<1e-9
    assert error.max()<1e-9


def test_fresh_multiview_hair_point_rejects_wrong_witness_and_weak_parallax():
    frames=[camera(x) for x in [0,.4,-.3]];point=np.array([[.2,.1,3.]])
    uv=[project(point,f)[0][0] for f in frames]
    result=multiview_point(uv,frames)
    np.testing.assert_allclose(result[0],point[0],atol=1e-12)
    wrong=[p.copy() for p in uv];wrong[-1][1]+=30
    assert multiview_point(wrong,frames) is None
    close=[camera(x) for x in [0,.001,-.001]]
    assert multiview_point([project(point,f)[0][0] for f in close],close) is None


def test_plane_hypothesis_recovers_supported_surface_with_outliers():
    rng=np.random.RandomState(9)
    plane=rng.uniform(-1,1,(300,3));plane[:,2]=5+rng.normal(0,.001,300)
    outliers=rng.uniform(-2,2,(80,3));points=np.concatenate([plane,outliers])
    result=fit_planes(points,.01)
    assert len(result)==1
    normal=np.asarray(result[0]['normal'])
    assert abs(normal[2])>.99999
    assert result[0]['inlier_points']==300
    assert result[0]['median_distance']<.003


def test_background_lookup_rejects_occluded_rgb_and_unsupported_plane_extent():
    model=object.__new__(PlaneBackground)
    model.planes=[dict(normal=[0.,0.,1.],offset=-2.,origin=[0.,0.,2.],
        basis=[[1.,0.,0.],[0.,1.,0.]],hull_equations=[[1.,0.,-2.],[-1.,0.,-2.],[0.,1.,-2.],[0.,-1.,-2.]])]
    model.frames=[dict(w=4,h=4)]*4
    good=torch.tensor([.3,.4,.5])[None,:,None,None].expand(1,3,4,4)
    model.images=[torch.tensor([1.,0.,0.])[None,:,None,None].expand(1,3,4,4)]+[good]*3
    model.masks=[torch.zeros(1,1,4,4)]+[torch.ones(1,1,4,4)]*3
    model.poses=[torch.eye(4)]*4;model.intrinsics=[torch.tensor([2.,2.,2.,2.])]*4
    origins=torch.zeros(3,3);directions=torch.tensor([[0.,0.,1.],[.1,0.,1.],[10.,0.,1.]])
    rgb,valid,depth,support=model.predict(origins,directions)
    assert valid.tolist()==[True,True,False]
    torch.testing.assert_close(rgb[:2],torch.tensor([[.3,.4,.5],[.3,.4,.5]]))
    assert support.tolist()==[3,3,0]
    assert torch.isinf(depth[-1])
