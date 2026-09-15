from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from view_consistent_source_quality import quality


def test_incidence_control_and_angle_only_do_not_average_colors():
    incidence=np.array([[.1,-.5,1.]])
    d=np.array([[.2,.5,1.]])
    np.testing.assert_allclose(quality(incidence,d,'incidence2'),np.array([[.25,1.,1.]]))
    np.testing.assert_array_equal(quality(incidence,d,'angular_only'),np.ones_like(incidence))


def test_invalid_geometry_and_mode_rejected():
    for i,d in [(np.ones(2),np.zeros(2)),(np.array([np.nan]),np.ones(1)),(np.ones(2),np.ones(3))]:
        with pytest.raises(ValueError):quality(i,d,'angular_only')
    with pytest.raises(ValueError):quality(np.ones(2),np.ones(2),'unknown')


def test_pixel_visibility_can_override_an_occluded_centroid_label(monkeypatch):
    from types import SimpleNamespace
    import torch
    import study_pixel_angular_head_texture as module
    from hard_surface_texture import gather_hard_rgb
    fake=SimpleNamespace(gather_hard_rgb=gather_hard_rgb)
    monkeypatch.setattr(module.study,'renderer',fake)
    monkeypatch.setattr(module.study,'install',lambda mode:'isolated-test')
    module.install()
    colors=torch.tensor([[[1.,1.,1.],[0.,0.,0.],[0.,0.,0.]],[[0.,0.,0.],[1.,1.,1.],[0.,0.,0.]]])
    weights=torch.tensor([[1.,0.,0.],[.5,1.,0.]])
    rgb,source,_=fake.gather_hard_rgb(colors,weights,torch.tensor([1,1,1]))
    assert source.tolist()==[0,1,-1]
    torch.testing.assert_close(rgb,torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]]))


def test_zero_registration_does_not_modify_sampling_coordinates():
    import torch
    from study_unwarped_head_texture import zero_registration
    q=torch.randn(2,1,5,2);before=q.clone()
    shift=zero_registration(None,None,q)
    assert shift.shape==q.shape and shift.device==q.device and shift.dtype==q.dtype
    torch.testing.assert_close(shift,torch.zeros_like(q));torch.testing.assert_close(q,before)
