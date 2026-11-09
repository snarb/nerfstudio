import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('free_candidates',scripts/'build_marshawn_free_space_rays.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_camera_outside_and_inside_near_clip():
    origins=np.array([[-2.,0,0],[0,0,0]]);directions=np.array([[1.,0,0],[1.,0,0]])
    start,end,margin,ok=builder.clip_intervals(origins,directions,[4.,.8],[[-1,-1,-1],[1,1,1]])
    np.testing.assert_allclose(start,[1,.01]);np.testing.assert_allclose(end,[3,.76]);np.testing.assert_allclose(margin,[.2,.04]);assert ok.all()


def test_physical_margin_is_invariant_across_boxes():
    o=np.array([[0.,0,0]]);d=np.array([[1.,0,0]])
    _,end_a,margin_a,_=builder.clip_intervals(o,d,[2],[[-1,-1,-1],[3,1,1]])
    _,end_b,margin_b,_=builder.clip_intervals(o,d,[2],[[-2,-2,-2],[4,2,2]])
    assert margin_a==margin_b==.1 and end_a==end_b==1.9


def test_parallel_miss_tangent_and_empty_surface_margin():
    o=np.array([[-2.,2,0],[-2,1,0],[0,0,0]]);d=np.array([[1.,0,0]]*3)
    _,_,_,ok=builder.clip_intervals(o,d,[4.,4.,.005],[[-1,-1,-1],[1,1,1]])
    assert not ok.any()


def test_projection_opengl_pixel_convention_and_negative_depth():
    pose=np.eye(4);points=np.array([[0,0,-2],[1,1,-2],[0,0,1]])
    uv,z=builder.project(points,pose,[100,120],[320,240]);np.testing.assert_allclose(uv[:2],[[320,240],[370,180]]);np.testing.assert_allclose(z,[2,2,-1])


def test_invalid_unit_direction_is_not_accepted():
    _,_,_,ok=builder.clip_intervals([[0,0,0]],[[2,0,0]],[2],[[-1,-1,-1],[1,1,1]])
    assert not ok.any()
    with pytest.raises(ValueError):builder.clip_intervals([[0,0,0]],[[1,0,0]],[2],[[0,0,0],[0,1,1]])
