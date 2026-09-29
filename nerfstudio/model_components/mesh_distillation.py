"""Opt-in mesh distillation contracts and frozen, spatially separate backgrounds.

No stock LookCloser defaults are changed by importing this module. Coordinates
remain in the immutable teacher gauge; background never supplies actor depth.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def tensor_digest(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def compose_actor_background(actor_rgb: Tensor, opacity: Tensor, background_rgb: Tensor) -> Tensor:
    """Actor RGB is already premultiplied by volume-rendering weights."""
    return actor_rgb + (1.0 - opacity.clamp(0, 1)) * background_rgb


def weighted_charbonnier(pred: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    weight = torch.broadcast_to(weight, pred.shape)
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Supervision weights must be finite and nonnegative")
    # torch.where avoids NaN contamination from unknown targets, even at weight 0.
    error = torch.where(weight > 0, pred - target, 0.0)
    return (torch.sqrt(error.square() + 1e-4) * weight).sum() / weight.sum().clamp_min(1e-8)


def weighted_tail_mean(value: Tensor, weight: Tensor, fraction: float = 1.) -> Tensor:
    """Optional hard-ray reduction, preserving relative confidence weights.

    Selection ranks weighted errors. Normalization uses the original total
    confidence times the selected fraction, not selected confidence alone.
    Thus uncertain targets cannot regain full confidence merely by being hard.
    """
    if not 0 < fraction <= 1: raise ValueError('Hard-ray fraction must be in (0,1]')
    weight = torch.broadcast_to(weight, value.shape)
    valid = weight > 0
    weighted = torch.where(valid, value, 0.)*weight
    if fraction == 1.: return weighted.sum()/weight.sum().clamp_min(1.)
    errors = weighted[valid]
    if not errors.numel(): return weighted.sum()
    count = max(1, math.ceil(errors.numel()*fraction))
    return errors.topk(count, sorted=False).values.sum()/(weight.sum()*(count/errors.numel())).clamp_min(1.)


def matte_tail_objective(cross_entropy: Tensor, target: Tensor, weight: Tensor, fraction: float = 1.) -> Tensor:
    """Mine prediction error, not the irreducible entropy of fractional alpha."""
    if fraction == 1.:
        return weighted_tail_mean(cross_entropy, weight, fraction)
    entropy = -target*target.clamp_min(1e-8).log()-(1-target)*(1-target).clamp_min(1e-8).log()
    excess = (cross_entropy-entropy).clamp_min(0.)
    return weighted_tail_mean(excess, weight, fraction)


def camera_z_to_distance(depth: Tensor, ray_norm: Tensor) -> Tensor:
    return depth * ray_norm


def projected_frequency(resolution: Tensor, fx: Tensor, fy: Tensor, width: Tensor,
                        height: Tensor, camera_z: Tensor, aabb_size: Tensor) -> Tensor:
    """UV hash resolution -> cycles/pixel -> scene-normalized 3D resolution.

    The conservative maximum over image axes avoids losing detail when W != H.
    Scaling both camera-z and the AABB, or image dimensions and intrinsics,
    leaves the result unchanged.
    """
    focal_uv = torch.maximum(fx / width, fy / height)
    return resolution * focal_uv * aabb_size.max() / camera_z.clamp_min(1e-8)


def ray_box(origins: Tensor, directions: Tensor, bounds: Tensor):
    parallel = directions.abs() < 1e-10
    safe = torch.where(parallel, torch.ones_like(directions), directions)
    a, b = (bounds[0] - origins) / safe, (bounds[1] - origins) / safe
    low, high = torch.minimum(a, b), torch.maximum(a, b)
    inside = (origins >= bounds[0]) & (origins <= bounds[1])
    low = torch.where(parallel & inside, -torch.inf, low)
    high = torch.where(parallel & inside, torch.inf, high)
    low = torch.where(parallel & ~inside, torch.inf, low)
    high = torch.where(parallel & ~inside, -torch.inf, high)
    near, far = low.amax(-1).clamp_min(0), high.amin(-1)
    return near, far, far > near


def ray_plane_slab(origins: Tensor, directions: Tensor, slab: Tensor):
    """Ray interval inside lo <= unit_normal.x + offset <= hi.

    Parallel rays inside the slab impose no upper bound; outside rays miss.
    The caller intersects this interval with its finite scene AABB.
    """
    # Small wall gaps require full-precision world geometry even while the
    # feature networks train under autocast. Matmul would silently use FP16.
    signed = (origins.float() * slab[:3].float()).sum(-1) + slab[3].float()
    velocity = (directions.float() * slab[:3].float()).sum(-1)
    parallel = velocity.abs() < 1e-10
    safe = torch.where(parallel, torch.ones_like(velocity), velocity)
    a, b = (slab[4] - signed) / safe, (slab[5] - signed) / safe
    near, far = torch.minimum(a, b), torch.maximum(a, b)
    inside = (signed >= slab[4]) & (signed <= slab[5])
    near = torch.where(parallel & inside, -torch.inf, near)
    far = torch.where(parallel & inside, torch.inf, far)
    near = torch.where(parallel & ~inside, torch.inf, near).clamp_min(0)
    far = torch.where(parallel & ~inside, -torch.inf, far)
    return near, far, far > near


class SeparateBackground(nn.Module):
    """A plane texture or small bounded radiance field, behind the actor box.

    `plane` is n.x + offset = 0 with actor on the positive side. A halfspace
    excludes all actor geometry from background density, including rays that
    miss the actor box. Geometry and UV bounds are checkpointed buffers.
    """
    def __init__(self, kind: str, geometry: Dict, texture_resolution: int | None = None):
        super().__init__()
        if kind not in {"plane", "field"}:
            raise ValueError(kind)
        self.kind = kind
        texture_resolution = int(texture_resolution or geometry.get("texture_resolution", 1024))
        self.samples = int(geometry.get("field_samples", 48))
        self.density_plane_bias = bool(geometry.get("density_plane_bias", False))
        self.physical_density = bool(geometry.get("physical_density", False))
        self.view_independent = bool(geometry.get("view_independent", False))
        for key in ("plane", "basis", "uv_bounds", "bounds", "actor_bounds"):
            self.register_buffer(key, torch.tensor(geometry[key], dtype=torch.float32))
        self.register_buffer("behind_limit", torch.tensor(float(geometry["behind_limit"])))
        if kind == "plane":
            self.texture = nn.Parameter(torch.zeros(1, 3, texture_resolution, texture_resolution))
            if geometry.get("train_plane", False):
                initial = self.plane.clone()
                del self._buffers["plane"]
                self.plane = nn.Parameter(initial)
        else:
            import tinycudann as tcnn
            self.encoding = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": 8,
                "n_features_per_level": 2, "log2_hashmap_size": int(geometry.get("field_log2_size",17)), "base_resolution": 16,
                "per_level_scale": float((geometry.get("field_max_res",512) / 16) ** (1 / 7))})
            self.network = tcnn.Network(self.encoding.n_output_dims + 3, 4,
                {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                 "n_neurons": 64, "n_hidden_layers": 2})

    def forward(self, origins: Tensor, directions: Tensor, samples: int | None = None) -> Dict[str, Tensor]:
        samples = samples or self.samples
        n, offset = self.plane[:3], self.plane[3]
        if self.kind == "plane":
            with torch.autocast(device_type=origins.device.type, enabled=False):
                origins, directions = origins.float(), directions.float()
                n, offset = n.float(), offset.float()
                denom = directions @ n
                t = -(origins @ n + offset) / torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
                point = origins + directions * t[:, None]
                uv = point @ self.basis.float().T
                uv = 2 * (uv - self.uv_bounds[0]) / (self.uv_bounds[1] - self.uv_bounds[0]) - 1
                valid = (t > 0) & (denom.abs() > 1e-8) & (uv.abs() <= 1).all(-1)
                rgb = F.grid_sample(self.texture.float().sigmoid(), uv[None, :, None],
                                    align_corners=True, padding_mode="border")[0, :, :, 0].T
                return {"rgb": rgb, "depth": t[:, None], "valid": valid[:, None]}
        near, far, hit = ray_box(origins, directions, self.bounds)
        # For rays through the actor, the background starts beyond its exit.
        _, actor_far, actor_hit = ray_box(origins, directions, self.actor_bounds)
        near = torch.where(actor_hit, torch.maximum(near, actor_far), near)
        hit &= far > near
        near = torch.where(hit, near, 0.)
        far = torch.where(hit, far, 1e-6)
        edges = torch.linspace(0, 1, samples + 1, device=origins.device)
        starts = near[:, None] + (far - near)[:, None] * edges[:-1]
        ends = near[:, None] + (far - near)[:, None] * edges[1:]
        mids = (starts + ends) / 2
        point = origins[:, None] + directions[:, None] * mids[..., None]
        norm = (point - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        encoded = self.encoding(norm.reshape(-1, 3).contiguous())
        dirs = directions[:, None].expand_as(point).reshape(-1, 3)
        if self.view_independent:
            dirs = torch.zeros_like(dirs)
        values = self.network(torch.cat([encoded, (dirs + 1) / 2], -1)).float().reshape(-1, samples, 4)
        allowed = ((point @ n + offset) < self.behind_limit) & hit[:, None]
        if self.physical_density:
            density = torch.exp((values[...,0]+1).clamp(-15,12)) / (self.bounds[1]-self.bounds[0]).max() * allowed
        elif self.density_plane_bias:
            # Explicit, trainable residual around a coarse wall initialization;
            # residual logits can suppress this surface or create another one.
            sdf = (point @ n + offset) / n.norm().clamp_min(1e-8)
            log_prior = -6 + 12 * torch.exp(-(sdf / .04).square())
            density = torch.exp((values[..., 0] + log_prior).clamp(-15, 12)) * allowed
        else:
            density = F.softplus(values[..., 0] + 2) * allowed
        alpha = 1 - torch.exp(-density * (ends - starts))
        trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], -1), -1)[:, :-1]
        weights = alpha * trans
        # A terminal colour uses the same field's far sample; all queries are in
        # the separate background domain and cannot grow actor geometry.
        colors = values[..., 1:].sigmoid()
        rgb = (weights[..., None] * colors).sum(1)
        if not self.physical_density:
            rgb = rgb + (1 - weights.sum(-1))[:, None] * colors[:, -1]
        depth = (weights * mids).sum(-1, keepdim=True) / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        result={"rgb": rgb, "depth": depth, "valid": hit[:, None]}
        if self.physical_density and self.training:
            result.update(weights=weights,distances=mids)
        return result
