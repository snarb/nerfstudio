"""A hard density boundary must preserve occlusion and report missing surfaces."""
from types import SimpleNamespace
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_luster_density_surface import hard_hits


class TwoSolids:
    def query_points(self,points,directions):
        z=points[:,2]
        front=10*(1-(z-1.5).abs()/.5).clamp_min(0)
        back=10*(1-(z-3.5).abs()/.5).clamp_min(0)
        density=(front+back)*(points[:,0]<1)
        color=torch.stack([(z<2).float(),torch.zeros_like(z),(z>2).float()],-1)
        return density[:,None],color


def rays_and_samples():
    rays=SimpleNamespace(origins=torch.tensor([[0.,0.,0.],[10.,0.,0.]]),
        directions=torch.tensor([[0.,0.,1.],[0.,0.,1.]]),nears=torch.zeros(2,1),fars=torch.full((2,1),4.))
    edges=torch.linspace(0,1,33)
    packed=dict(packed_ray_indices=torch.repeat_interleave(torch.arange(2,dtype=torch.int32),32),
                packed_spacing_starts=edges[:-1].repeat(2)[:,None],
                packed_spacing_ends=edges[1:].repeat(2)[:,None])
    return rays,packed


def test_first_boundary_occludes_a_second_solid_and_empty_ray_stays_black():
    rays,packed=rays_and_samples()
    rgb,depth,stats=hard_hits(TwoSolids(),rays,packed,5.)
    assert torch.allclose(depth,torch.tensor([1.25,0.]),atol=2e-5)
    assert torch.equal(rgb,torch.tensor([[1.,0.,0.],[0.,0.,0.]]))
    assert stats['hit_fraction']==.5
    assert stats['root_residual_below_1pct_fraction']==1.


def test_density_level_changes_surface_position_with_no_color_blending():
    rays,packed=rays_and_samples()
    low,low_depth,_=hard_hits(TwoSolids(),rays,packed,2.)
    high,high_depth,_=hard_hits(TwoSolids(),rays,packed,8.)
    assert torch.allclose(low_depth[:1],torch.tensor([1.1]),atol=2e-5)
    assert torch.allclose(high_depth[:1],torch.tensor([1.4]),atol=2e-5)
    assert torch.equal(low,high)


def test_no_samples_has_no_surface_and_no_spurious_color():
    rays,_=rays_and_samples()
    packed=dict(packed_ray_indices=torch.empty(0,dtype=torch.long),
                packed_spacing_starts=torch.empty(0,1),packed_spacing_ends=torch.empty(0,1))
    rgb,depth,stats=hard_hits(TwoSolids(),rays,packed,5.)
    assert torch.count_nonzero(rgb)==torch.count_nonzero(depth)==0
    assert stats['hit_fraction']==0.


def test_independent_color_density_cannot_move_the_geometry_boundary():
    class Green:
        def query_points(self,p,d):
            return torch.zeros(len(p),1),torch.tensor([[0.,1.,0.]]).expand(len(p),3)
    rays,packed=rays_and_samples()
    rgb,depth,stats=hard_hits(TwoSolids(),rays,packed,5.,color_field=Green())
    assert torch.allclose(depth,torch.tensor([1.25,0.]),atol=2e-5)
    assert torch.equal(rgb,torch.tensor([[0.,1.,0.],[0.,0.,0.]]))
    assert stats['root_residual_below_1pct_fraction']==1.
