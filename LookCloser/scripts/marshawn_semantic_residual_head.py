"""Optional CPU-prepared semantic refinement; no fitter or rendering integration.

Keep a copied coarse logit field frozen and learn only a zero-initialized fine
logit residual. This does not change native radiance-field density or weights.
"""
from __future__ import annotations

from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from marshawn_frozen_semantic_head import FrozenActorGrid


class FrozenResidualActorGrid(nn.Module):
    """p = sigmoid(trilerp(coarse) + trilerp(residual)) inside the same box.

    This opt-in helper accepts only the direct-LC identity floor frame. Coarse
    logits/bounds/frame are independent copies stored as buffers; the input
    coarse head is neither modified nor retained. The sole parameter is
    ``residual_logits``. Use ordinary state_dict plus the optimizer state for
    restoration, with identical grid shapes. Source hashes, camera/label
    bindings, RNG and sampling history remain the future controller's job.

    No regularizer, probability prior, threshold or data policy is added.
    Nodes never reached by a gradient stay zero under ordinary Adam initialized
    from this state. Previously reached nodes can keep moving through momentum.
    """

    def __init__(self, coarse: FrozenActorGrid, resolution: int = 256):
        super().__init__()
        if not isinstance(coarse, FrozenActorGrid):
            raise TypeError("An explicit FrozenActorGrid coarse head is required")
        if isinstance(resolution, bool) or not isinstance(resolution, Integral) or resolution < 2:
            raise ValueError("Fine node resolution must be an integer >= 2")
        identity = torch.eye(4, dtype=coarse.world_to_box.dtype, device=coarse.world_to_box.device)
        if not torch.equal(coarse.world_to_box, identity):
            raise ValueError("Only the accepted direct-LC identity floor frame is supported")
        if not torch.isfinite(coarse.logits).all():
            raise ValueError("Finite coarse logits required")
        if resolution <= max(coarse.logits.shape[-3:]):
            raise ValueError("Residual resolution must exceed the coarse node resolution")
        self.register_buffer("bounds", coarse.bounds.detach().clone())
        self.register_buffer("world_to_box", coarse.world_to_box.detach().clone())
        self.register_buffer("coarse_logits", coarse.logits.detach().clone())
        self.residual_logits = nn.Parameter(coarse.logits.new_zeros((1, 1, resolution, resolution, resolution)))

    def forward(self, xyz):
        # Match FrozenActorGrid arithmetic, including explicit identity transform.
        points = xyz.detach() @ self.world_to_box[:3, :3].T + self.world_to_box[:3, 3]
        unit = (points-self.bounds[0])/(self.bounds[1]-self.bounds[0])
        inside = ((unit >= 0) & (unit <= 1)).all(-1)
        coordinates = (2*unit-1)[:, [2, 1, 0]].reshape(1, 1, 1, -1, 3)
        # Tensor node axes XYZ; grid_sample coordinate axes W/H/D = Z/Y/X.
        coarse = F.grid_sample(self.coarse_logits, coordinates, align_corners=True, padding_mode="border").reshape(-1)
        residual = F.grid_sample(self.residual_logits, coordinates, align_corners=True, padding_mode="border").reshape(-1)
        return (coarse+residual).sigmoid()*inside.to(coarse.dtype)


def dense_fp32_head_memory(resolution: int, coarse_resolution: int = 128):
    """Analytic tensor bytes; no allocation and no CUDA initialization.

    Adam minimum is residual parameter + dense gradient + two dense moments,
    plus the frozen coarse buffer. Excludes scalar step, metadata, source head,
    activations, cache, allocator reservation and optimizer temporary tensors.
    """
    for value in (resolution, coarse_resolution):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
            raise ValueError("Integer node resolutions >= 2 required")
    nodes = int(resolution)**3
    residual_bytes = 4*nodes
    coarse_bytes = 4*int(coarse_resolution)**3
    return dict(nodes=nodes, grid_bytes=residual_bytes, frozen_coarse_bytes=coarse_bytes,
                hierarchical_inference_bytes=coarse_bytes+residual_bytes,
                hierarchical_Adam_minimum_bytes=coarse_bytes+4*residual_bytes)
