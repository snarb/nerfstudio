"""Independent semantics for immutable packed radiance fields; no density training.

The rendered feature is a probability over native *termination contributions*,
not optical alpha ground truth. RGB fitting and model restoration are external.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EVAL_CAMERAS = frozenset({"DC", "HC", "KA"})


def tensor_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_exact_restoration(module, saved_state):
    """Fail closed on omitted/extra keys, dtype/shape differences, or tensor drift."""
    actual = module.state_dict()
    if actual.keys() != saved_state.keys():
        raise ValueError("Restoration state keys differ")
    for name, value in actual.items():
        prior = saved_state[name]
        if value.dtype != prior.dtype or value.shape != prior.shape or not torch.equal(value.detach().cpu(), prior.detach().cpu()):
            raise ValueError("Restoration tensor differs: " + name)


class FrozenActorGrid(nn.Module):
    """Small continuous 3D class grid in explicitly supplied floor coordinates.

    Its output is view independent. Outside the accepted actor box p_actor=0;
    full-scene RGB/density/traversal remain untouched. A soft grid can be exported
    as sigma_actor=sigma*p_actor; that export requires NEW transmittance.
    """
    def __init__(self, bounds, world_to_box, resolution=128, initial_probability=.99):
        super().__init__()
        bounds = torch.as_tensor(bounds, dtype=torch.float32)
        frame = torch.as_tensor(world_to_box, dtype=torch.float32)
        if bounds.shape != (2, 3) or not torch.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
            raise ValueError("Finite ordered bounds required")
        if frame.shape != (4, 4) or not torch.allclose(frame[3], torch.tensor([0., 0., 0., 1.])):
            raise ValueError("Explicit rigid world-to-box required")
        if not torch.allclose(frame[:3, :3].T @ frame[:3, :3], torch.eye(3), atol=1e-5) or torch.det(frame[:3, :3]) < .999:
            raise ValueError("world_to_box must be a proper rigid transform")
        if resolution < 2 or not 0 < initial_probability < 1:
            raise ValueError("Invalid grid initialization")
        self.register_buffer("bounds", bounds)
        self.register_buffer("world_to_box", frame)
        self.logits = nn.Parameter(torch.full((1, 1, resolution, resolution, resolution), float(np.log(initial_probability/(1-initial_probability)))))

    def forward(self, xyz):
        points = xyz.detach() @ self.world_to_box[:3, :3].T + self.world_to_box[:3, 3]
        unit = (points-self.bounds[0])/(self.bounds[1]-self.bounds[0])
        inside = ((unit >= 0) & (unit <= 1)).all(-1)
        # Tensor storage is XYZ, while grid_sample's coordinates are W,H,D=Z,Y,X.
        coordinates = (2*unit-1)[:, [2, 1, 0]].reshape(1, 1, 1, -1, 3)
        logits = F.grid_sample(self.logits, coordinates, align_corners=True, padding_mode="border").reshape(-1)
        return logits.sigmoid()*inside.to(logits.dtype)


def rendered_classes(probability, native_weights, ray_indices, ray_count):
    """[actor termination, nonactor termination, escape] with frozen native weights.

    q_actor <= native_alpha; escape is irreducible and has no semantic gradient.
    Never recompute weights with p during semantic fitting.
    """
    p = probability.reshape(-1)
    w = native_weights.detach().reshape(-1)
    ids = ray_indices.detach().long().reshape(-1)
    if p.shape != w.shape or ids.shape != w.shape or not torch.isfinite(p).all() or not torch.isfinite(w).all():
        raise ValueError("Finite equal packed sample arrays required")
    if (p < 0).any() or (p > 1).any() or (w < 0).any() or (ids < 0).any() or (ids >= ray_count).any():
        raise ValueError("Invalid packed probabilities/weights/indices")
    alpha = torch.zeros(ray_count, device=p.device, dtype=p.dtype).index_add(0, ids, w)
    if (alpha > 1+2e-5).any():
        raise ValueError("Weights are not a native termination subprobability")
    actor = torch.zeros_like(alpha).index_add(0, ids, w*p)
    return torch.stack((actor, (alpha-actor).clamp_min(0), (1-alpha).clamp_min(0)), -1)


def conditional_class_objective(classes, semantic_target, label_weight):
    """Native-opacity-weighted conditional class CE, not an opacity objective.

    Equivalent target masses are [A*y,A*(1-y),1-A], whose escape term is
    constant. y is a semantic class label, NOT a measured optical coverage.
    UNKNOWN/invalid/protected pixels have label_weight=0. The mean uses a fixed
    ray count, never sum(batch confidence), so a fixed sampling distribution is
    required for comparisons. Zero-alpha rays provide no gradient.
    """
    y = semantic_target.detach().to(classes).reshape(-1)
    confidence = label_weight.detach().to(classes).reshape(-1)
    if classes.shape != (len(y), 3) or confidence.shape != y.shape or not torch.isfinite(y).all() or not torch.isfinite(confidence).all():
        raise ValueError("Finite matching labels/confidence required")
    if (y < 0).any() or (y > 1).any() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError("Labels/confidence must be within [0,1]")
    alpha = classes[:, :2].sum(-1).detach()
    conditional = (classes[:, 0]/alpha.clamp_min(1e-12)).clamp(1e-7, 1-1e-7)
    terms = -y*conditional.log()-(1-y)*torch.log1p(-conditional)
    return (confidence*alpha*terms).mean()


def ray_capacity(native_weights, inside, ray_indices, ray_count):
    """Actual maximum actor mass under the accepted box, not reconstructed alpha."""
    w = native_weights.detach().reshape(-1)
    ids = ray_indices.long().reshape(-1)
    total = torch.zeros(ray_count, device=w.device, dtype=w.dtype).index_add(0, ids, w)
    cap = torch.zeros_like(total).index_add(0, ids, w*inside.reshape(-1))
    return total, cap


@torch.no_grad()
def capture_native_packed(model, ray_bundle):
    """Read actual native sampler/field/weights without changing field or grid.

    Only the native NGP packed VolumetricSampler path is supported initially.
    Call model.eval(); freeze all parameters; restore/check ALL model state before
    using this adapter. No callbacks, occupancy updates, or RGB target edits.
    """
    import nerfacc
    from nerfstudio.field_components.field_heads import FieldHeadNames
    from nerfstudio.model_components.ray_samplers import VolumetricSampler
    if model.training or not isinstance(model.sampler, VolumetricSampler):
        raise ValueError("Frozen eval-mode native packed NGP required")
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Freeze every native parameter before capture")
    if model.collider is not None:
        ray_bundle = model.collider(ray_bundle)
    samples, ids = model.sampler(ray_bundle=ray_bundle,
        near_plane=model.config.near_plane, far_plane=model.config.far_plane,
        render_step_size=model.config.render_step_size,
        alpha_thre=model.config.alpha_thre, cone_angle=model.config.cone_angle)
    outputs = model.field(samples)
    starts, ends = samples.frustums.starts[..., 0], samples.frustums.ends[..., 0]
    sigma = outputs[FieldHeadNames.DENSITY][..., 0]
    rgb = outputs[FieldHeadNames.RGB]
    weights, _, _ = nerfacc.render_weight_from_density(t_starts=starts, t_ends=ends,
        sigmas=sigma, packed_info=nerfacc.pack_info(ids, len(ray_bundle)))
    accumulated = nerfacc.accumulate_along_rays(weights, values=rgb, ray_indices=ids, n_rays=len(ray_bundle))
    alpha = nerfacc.accumulate_along_rays(weights, values=None, ray_indices=ids, n_rays=len(ray_bundle))
    # Native renderer includes the model's actual black/white/background policy.
    native = model.get_outputs(ray_bundle)
    composed = model.renderer_rgb(rgb=rgb, weights=weights[..., None], ray_indices=ids, num_rays=len(ray_bundle))
    discrepancy = (native["rgb"]-composed).abs().max().item()
    if discrepancy > 2e-5:
        raise ValueError("Native RGB/weight parity failed: " + str(discrepancy))
    return dict(xyz=samples.frustums.get_positions(), starts=starts, ends=ends,
        sigma=sigma, rgb=rgb, weight=weights, ray_indices=ids,
        native_rgb=native["rgb"], native_foreground_rgb=accumulated,
        native_alpha=alpha, parity_error=discrepancy)


def validate_cache(path, train_cameras, expected_model_digest):
    """Cache files carry no RGB labels and cannot accept EVAL views for fitting."""
    with np.load(Path(path), allow_pickle=False) as src:
        row = {k: src[k].copy() for k in src.files}
    cameras = set(row["camera"].tolist())
    if len(set(train_cameras)) != 63 or set(train_cameras) & EVAL_CAMERAS or not cameras <= set(train_cameras) or cameras & EVAL_CAMERAS:
        raise ValueError("Only declared TRAIN63 camera samples can enter a fit cache")
    if str(row["model_digest"].item()) != expected_model_digest:
        raise ValueError("Cache model state differs")
    n = len(row["camera"])
    if row["semantic_target"].shape != (n,) or row["label_weight"].shape != (n,) or row["pixel_xy"].shape != (n, 2):
        raise ValueError("Invalid per-ray cache dimensions")
    ids, weights, xyz = row["ray_indices"], row["weight"], row["xyz"]
    if ids.dtype.kind not in "iu" or ids.shape != weights.shape or xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all() or not np.isfinite(weights).all() or (weights < 0).any() or (ids < 0).any() or (ids >= n).any():
        raise ValueError("Invalid native packed cache")
    if (np.bincount(ids, weights=weights, minlength=n) > 1+2e-5).any():
        raise ValueError("Cached weights exceed unit native alpha")
    return row


def export_density(sigma, actor_probability):
    """Physical 3D mixture thinning; caller MUST recompute ray transmittance.

    A small nonzero p does NOT guarantee removal: sigma=1e9,p=.001 still
    produces opaque density at ordinary ray intervals. Physical renders must
    inspect recomputed alpha, not only semantic classification accuracy.
    Removing nonactors may reveal originally occluded actor/uncertain geometry.
    Do not confuse this result with sum(native_weights*p*RGB), which is only a
    contribution partition under the original full-scene occlusion.
    """
    return sigma.detach()*actor_probability
