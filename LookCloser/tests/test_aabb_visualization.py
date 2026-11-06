import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from aabb_visualization import box_edges, edge_samples, overlay_depth_aware, depth_visibility
from render_aabb_review import first_opaque_depth


def camera():
    return dict(w=101,h=101,fl_x=80.,fl_y=80.,cx=50.5,cy=50.5,
                transform_matrix=np.eye(4).tolist())


BOX=[[-.5,-.5,-3.],[.5,.5,-2.]]


def test_perspective_samples_have_actual_line_depth():
    c=camera();corners,_=box_edges(BOX)
    for line in edge_samples(BOX,c):
        uv=line['uv'];directions=np.c_[(uv[:,0]-50)/80,-(uv[:,1]-50)/80,-np.ones(len(uv))]
        xyz=directions/np.linalg.norm(directions,axis=1)[:,None]*line['distance'][:,None]
        a,b=corners[list(line['edge'])];direction=b-a
        parameter=(xyz-a)@direction/(direction@direction)
        np.testing.assert_allclose(xyz,a+parameter[:,None]*direction,atol=1e-12)
        assert parameter.min()>=-1e-12 and parameter.max()<=1+1e-12


def test_opaque_foreground_hides_all_edges_and_unknown_is_not_visible():
    rgb=Image.new('RGB',(101,101),(30,40,50))
    for depth in [np.ones((101,101)),np.full((101,101),np.nan)]:
        image,report=overlay_depth_aware(rgb,BOX,camera(),depth)
        np.testing.assert_array_equal(image,rgb)
        assert report['visible_samples']==0
    image,report=overlay_depth_aware(rgb,BOX,camera(),np.full((101,101),np.inf))
    assert report['visible_samples']>0 and report['hidden_samples']==0
    assert np.any(np.asarray(image)!=np.asarray(rgb))


def test_depth_is_ray_distance_not_camera_axis_z():
    uv=np.array([90.,90.]);distance=3*np.linalg.norm([(uv[0]-50)/80,-(uv[1]-50)/80,-1])
    assert not depth_visibility(distance,3.)
    assert depth_visibility(distance,distance)


def test_line_does_not_bridge_a_foreground_occluder():
    depth=np.full((101,101),np.inf)
    depth[40:61,27:34]=1.
    rgb=Image.new('RGB',(101,101),(30,40,50))
    image,_=overlay_depth_aware(rgb,BOX,camera(),depth,dash=1000,gap=0)
    np.testing.assert_array_equal(np.asarray(image)[40:61,27:34],np.asarray(rgb)[40:61,27:34])
    assert np.any(np.asarray(image)[30:40,27:34]!=np.asarray(rgb)[30:40,27:34])


def test_near_plane_clipping_is_finite_and_bounded():
    lines=edge_samples([[-.5,-.5,-2],[.5,.5,.2]],camera())
    assert lines
    assert sum(len(x['uv'])for x in lines)<5000
    assert all(np.isfinite(x['uv']).all()and np.isfinite(x['distance']).all()for x in lines)


def test_bad_raster_and_distortion_rejected():
    with pytest.raises(ValueError,match='rasters'):
        overlay_depth_aware(Image.new('RGB',(100,101)),BOX,camera(),np.ones((101,101)))
    c=camera();c['k1']=.01
    with pytest.raises(ValueError,match='undistorted'):
        edge_samples(BOX,c)


def test_packed_surface_crossing_and_transparent_ray():
    import torch
    start=torch.tensor([0.,1.,0.,2.]);end=start+1
    sigma=torch.tensor([.3,2.,.1,10.]);trans=torch.tensor([1.,np.exp(-.3),1.,1.])
    alpha=-torch.expm1(-sigma)
    depths=first_opaque_depth(start,end,sigma,trans,alpha,torch.tensor([0,0,1,2]),4)
    assert depths[0].item()==pytest.approx(1+(np.log(2)-.3)/2,abs=1e-6)
    assert depths[2].item()==pytest.approx(2+np.log(2)/10,abs=1e-6)
    assert torch.isinf(depths[[1,3]]).all()
