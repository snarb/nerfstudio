from copy import deepcopy
import torch

from nerfstudio.fields.lookcloser_field import LookCloserField


def field(split):
    f=LookCloserField.__new__(LookCloserField);torch.nn.Module.__init__(f)
    torch.manual_seed(12)
    f.encoding=torch.nn.Linear(3,4)
    f.mlp_geo=torch.nn.Linear(4,3)
    f.mlp_color=torch.nn.Linear(5,3)
    f.embedding_appearance=None;f.density_activation='softplus'
    f.enable_feature_reweighting=False
    f.density_geometry_encoding=deepcopy(f.encoding) if split else None
    f.density_geometry_mlp=deepcopy(f.mlp_geo) if split else None
    f._normalize_positions=lambda p:(p,torch.ones(len(p),dtype=torch.bool),p.shape[:-1],p)
    f.encode_directions=lambda d:d
    f._activate_rgb=torch.sigmoid
    return f


def test_cloned_density_preserves_point_rgb_and_density():
    shared,split=field(False),field(True)
    p=torch.randn(8,3);d=torch.randn(8,3)
    a=shared.query_points(p,d);b=split.query_points(p,d)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])


def test_geometry_prior_gradient_does_not_modify_color_features():
    f=field(True);p=torch.randn(8,3);d=torch.randn(8,3)
    density,_=f.query_points(p,d);density.sum().backward()
    assert f.encoding.weight.grad is None
    assert f.mlp_geo.weight.grad is None
    assert f.mlp_color.weight.grad is None
    assert f.density_geometry_encoding.weight.grad is not None
    assert f.density_geometry_mlp.weight.grad is not None


def test_color_updates_do_not_change_density_at_fixed_points():
    f=field(True);p=torch.randn(8,3);d=torch.randn(8,3)
    density,color=f.query_points(p,d)
    color.sum().backward()
    assert f.density_geometry_encoding.weight.grad is None
    assert f.density_geometry_mlp.weight.grad is None
    optimizer=torch.optim.SGD([*f.encoding.parameters(),*f.mlp_geo.parameters(),*f.mlp_color.parameters()],lr=.1)
    optimizer.step();after,rgb=f.query_points(p,d)
    assert torch.equal(density,after)
    assert not torch.equal(color,rgb)
