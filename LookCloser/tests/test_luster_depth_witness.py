"""Calibrated plane and occlusion checks for confidence-qualified MVS priors."""
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_luster_depth_confidence import witness_residuals


def camera(x=0.,fx=40.,width=32,height=24):
    pose=torch.eye(4)
    pose[0,3]=x
    return dict(transform_matrix=pose.tolist(),fl_x=fx,fl_y=fx,cx=width/2,cy=height/2,w=width,h=height)


def test_translated_cameras_agree_on_planar_depth():
    depth=torch.full((24,32),2.)
    visible,relative,pixel,absolute,farther=witness_residuals(depth,camera(),depth,camera(.1))
    assert visible.sum()>24*25
    assert relative[visible].max()<1e-6
    assert pixel[visible].max()<1e-4
    assert absolute[visible].max()<1e-6
    assert not farther[visible].any()


def test_nearer_occluder_is_not_farther_geometry_contradiction():
    reference=torch.full((24,32),2.)
    observed=torch.ones(24,32)
    visible,relative,pixel,absolute,farther=witness_residuals(reference,camera(),observed,camera())
    assert relative[visible].min()>.4
    assert not farther.any()


def test_subpixel_projection_backprojects_the_measured_pixel():
    depth=torch.full((24,32),2.)
    # Disparity is 1.4 pixels, so nearest-neighbor source depth is measured
    # 0.4 pixels away from the continuous projected coordinate.
    visible,relative,pixel,absolute,_=witness_residuals(depth,camera(),depth,camera(.07))
    assert relative[visible].max()<1e-6
    assert torch.allclose(pixel[visible],torch.full_like(pixel[visible],.4),atol=1e-5)
    assert absolute[visible].max()<1e-6


def test_invalid_source_depth_does_not_confirm_reference():
    depth=torch.full((24,32),2.)
    source=torch.zeros_like(depth)
    visible,*_=witness_residuals(depth,camera(),source,camera())
    assert not visible.any()


def test_camera_z_and_ray_distance_have_distinct_off_axis_lengths():
    x,y,fx,cx,cy=25.,16.,40.,16.,12.
    z=2.
    direction=torch.tensor([(x+.5-cx)/fx,-(y+.5-cy)/fx,-1.])
    ray_distance=z*direction.norm()
    point=direction/direction.norm()*ray_distance
    assert torch.allclose(point[2],torch.tensor(-z))
    assert ray_distance>z
