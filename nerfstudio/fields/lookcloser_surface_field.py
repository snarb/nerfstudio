"""LookCloser hash/SH appearance with an opt-in signed surface head."""
from __future__ import annotations
import torch
from torch import Tensor
from nerfstudio.fields.lookcloser_field import LookCloserField


class LookCloserSurfaceField(LookCloserField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continuous_sdf = False
        self.register_buffer('surface_field_version',torch.tensor(1,dtype=torch.int64))
        self.opacity_head = torch.nn.Linear(self.geo_feat_dim,1)
        torch.nn.init.zeros_(self.opacity_head.weight)
        torch.nn.init.constant_(self.opacity_head.bias,4.)

    def query_surface(self, positions: Tensor, directions: Tensor | None = None):
        if not len(positions):
            sdf=positions.new_empty((0,1))
            return sdf if directions is None else (sdf,positions.new_empty((0,3)),sdf)
        normalized, selector, _, world = self._normalize_positions(positions)
        # Nearest-cell frequency reweighting is discontinuous. A sign jump
        # across its voxel boundary is not a zero of a signed distance field.
        # Continuous geometry uses raw interpolated hash features, while
        # appearance retains LookCloser's frequency-aware feature weighting.
        features = (self.encoding(normalized) if self.continuous_sdf else
                    self._encode_with_optional_reweighting(normalized,query_positions_flat=world))
        h = self.mlp_geo(features)
        # Train-only camera focus is the origin. A small object and outer room
        # shell give RGB gradients without a foreground mask or measured depth.
        aabb = self.aabb.to(positions)
        center = (aabb[0]+aabb[1])*.5
        room = ((aabb[1]-aabb[0])*.475-(positions-center).abs()).amin(-1,keepdim=True)
        actor = positions.norm(dim=-1,keepdim=True)-.25
        sdf = torch.minimum(actor,room) + .1*h[...,:1].float()
        if directions is None:
            return sdf
        color_h = h
        if self.continuous_sdf and self.enable_feature_reweighting:
            color_h=self.mlp_geo(self._encode_with_optional_reweighting(normalized,query_positions_flat=world))
        rgb = self._activate_rgb(self.mlp_color(torch.cat([color_h[...,1:],self.encode_directions(directions)],-1)))
        opacity = torch.sigmoid(self.opacity_head(h[...,1:].float()))
        return sdf,rgb,opacity

    def density_fn(self, positions: Tensor) -> Tensor:
        # Only used by inherited probes; surface rendering never traverses the
        # volume occupancy grid or applies a density activation to signed SDF.
        return torch.exp(-self.query_surface(positions).abs()*200.)*200.

    def finite_gradients(self, positions: Tensor, epsilon: float = 1e-3) -> Tensor:
        offsets = torch.eye(3,device=positions.device,dtype=positions.dtype)*epsilon
        plus = positions[:,None]+offsets
        minus = positions[:,None]-offsets
        values = self.query_surface(torch.cat([plus.reshape(-1,3),minus.reshape(-1,3)]))
        a,b = values.reshape(2,-1,3).unbind(0)
        return (a-b)/(2*epsilon)
