"""Discrete frequency cells must not become false signed-geometry roots."""
import torch
from nerfstudio.fields.lookcloser_surface_field import LookCloserSurfaceField


def field():
    f=LookCloserSurfaceField.__new__(LookCloserSurfaceField)
    torch.nn.Module.__init__(f)
    f.register_buffer('aabb',torch.tensor([[-2.,-2.,-2.],[2.,2.,2.]]))
    f.encoding=torch.nn.Identity()
    f._normalize_positions=lambda p:(p,torch.ones(len(p),dtype=torch.bool),p.shape[:-1],p)
    f._encode_with_optional_reweighting=lambda p,query_positions_flat: p*torch.where(p[:,:1]<.5,1.,20.)
    class Geo(torch.nn.Module):
        def forward(self,p):return torch.cat([p[:,:1],p[:,:1],torch.zeros(len(p),14)],-1)
    class Color(torch.nn.Module):
        def forward(self,p):return p[:,:3]
    f.mlp_geo=Geo();f.mlp_color=Color();f._activate_rgb=lambda p:p
    f.encode_directions=lambda d:torch.zeros(len(d),16)
    f.opacity_head=torch.nn.Linear(15,1)
    f.enable_feature_reweighting=True
    return f


def test_signed_geometry_continuity_preserves_frequency_aware_appearance():
    f=field();p=torch.tensor([[.5-1e-5,0.,0.],[.5+1e-5,0.,0.]])
    f.continuous_sdf=False
    original=f.query_surface(p)
    assert float(original[1]-original[0])>.9
    f.continuous_sdf=True
    signed=f.query_surface(p)
    assert float((signed[1]-signed[0]).abs())<1e-4
    s,rgb,_=f.query_surface(p,torch.tensor([[0.,0.,1.],[0.,0.,1.]]))
    assert torch.equal(s,signed)
    assert float(rgb[1,0]-rgb[0,0])>9.
