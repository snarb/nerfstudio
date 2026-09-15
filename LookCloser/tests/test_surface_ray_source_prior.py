import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from surface_ray_source_prior import ray_weights,admit_quality,gather_relative
from hard_surface_texture import gather_hard_rgb


def test_ray_prior_rigid_invariant_and_same_center_best():
    points=np.array([[0.,0.,1.],[1.,2.,3.]])
    centers=np.array([[3.,0.,0.],[0.,4.,0.]])
    a=ray_weights(points,centers,centers[0],4.)
    np.testing.assert_allclose(a[0],1)
    assert (a[1]<a[0]).all()
    rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    np.testing.assert_allclose(a,ray_weights(points@rot+5,centers@rot+5,centers[0]@rot+5,4.),atol=1e-7)
    # No optical-axis or lens input exists: panning at the same center cannot
    # change these weights for the same reconstructed points.


def test_bad_geometry_rejected():
    with pytest.raises(ValueError):ray_weights(np.zeros((1,3)),np.zeros((2,3)),np.ones(3),4)
    with pytest.raises(ValueError):ray_weights(np.ones((1,3)),np.zeros((2,3)),np.zeros(3),0)


def test_relative_source_is_hard_and_visibility_preserving():
    colors=torch.arange(18,dtype=torch.float32).reshape(2,3,3)
    weights=torch.tensor([[.1,.8,0.],[1.,1.,0.]])
    preferred=torch.tensor([0,0,0])
    rgb,source,fallback=gather_relative(colors,weights,preferred,.5)
    assert source.tolist()==[1,0,-1] and fallback.tolist()==[True,False,False]
    torch.testing.assert_close(rgb[:,0],colors[1,:,0]);torch.testing.assert_close(rgb[:,1],colors[0,:,1])
    assert not rgb[:,2].any()
    original=gather_hard_rgb(colors,weights,preferred)
    for a,b in zip(original,gather_relative(colors,weights,preferred,0)):torch.testing.assert_close(a,b)


def test_admission_uses_per_point_prior_before_clip():
    np.testing.assert_allclose(admit_quality([[10,1],[1,10]],[[.01,1],[1,.01]]),[[0,1],[1,0]])
