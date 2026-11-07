"""Observe one frozen LookCloser native forward; never substitute its sampler.

This adapter is an inference contribution cache, not a physical visibility or
empty-space certificate. All distances are the actual frustum ray parameters.
"""
from __future__ import annotations

import torch


def packed_density_weights(starts, ends, sigma, ids, ray_count):
    """Independent segmented emission/absorption proof, including empty rays."""
    tau = sigma.double() * (ends.double()-starts.double())
    prefix = torch.cumsum(tau, 0)-tau
    first = torch.ones_like(ids, dtype=torch.bool)
    first[1:] = ids[1:] != ids[:-1]
    bases = torch.zeros(ray_count, dtype=torch.float64, device=tau.device)
    bases[ids[first]] = prefix[first]
    return torch.exp(-(prefix-bases[ids])) * -torch.expm1(-tau)


def _max_error(a, b):
    if a.shape != b.shape:
        raise ValueError("Array shape mismatch")
    return float((a-b).abs().max()) if a.numel() else 0.


@torch.no_grad()
def capture_lc_native(model, ray_bundle, *, density_key=None, rgb_key=None, tolerance=2e-5):
    """Temporary field.forward observer around exactly one native get_outputs.

    Caller owns source restoration, identity/lease, RNG, nonpersistent LUT, and
    full-pipeline-state checks. This routine restores the exact Python forward
    binding even when inference or a proof fails. Empty packed forwards require
    ZERO field calls; nonempty forwards require exactly ONE. No second render,
    density_fn query, resampling, optimizer, callback, or update occurs here.
    """
    if density_key is None or rgb_key is None:
        from nerfstudio.field_components.field_heads import FieldHeadNames
        density_key, rgb_key = FieldHeadNames.DENSITY, FieldHeadNames.RGB
    if model.training or model.field.training or model.adaptive_sampler.training:
        raise ValueError("Frozen native eval mode required")
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Freeze all native parameters")
    if model._resolved_ray_sampling_mode() != "adaptive":
        raise ValueError("Only native packed LookCloser adaptive path qualified")
    if model.config.background_color != "black":
        raise ValueError("Only deterministic source black background qualified")
    if model.config.corrected_arm_allocator or model.config.max_steps_per_ray != 1024:
        raise ValueError("Preserve source legacy1024 ARM")
    had_binding = "forward" in model.field.__dict__
    binding = model.field.__dict__.get("forward")
    original = model.field.forward
    captured = []

    def observe(samples, *args, **kwargs):
        result = original(samples, *args, **kwargs)
        captured.append((samples, result))
        return result

    model.field.forward = observe
    try:
        # Base Model.forward applies exactly this collider before get_outputs.
        rays = model.collider(ray_bundle) if model.collider is not None else ray_bundle
        native = model.get_outputs(rays)
    finally:
        if had_binding:
            model.field.forward = binding
        else:
            delattr(model.field, "forward")
    ids = native["packed_ray_indices"].reshape(-1)
    w = native["packed_weights"].reshape(-1)
    n, count = len(ids), len(rays)
    if len(captured) != int(n > 0):
        raise ValueError("Final packed field.forward must occur once (zero for empty)")
    if ids.dtype not in (torch.int32, torch.int64) or ids.shape != w.shape:
        raise ValueError("Packed index/weight shape or dtype mismatch")
    if (ids < 0).any() or (ids >= count).any() or (ids[1:] < ids[:-1]).any():
        raise ValueError("Packed indices must be sorted and in range")
    if n:
        samples, field = captured[0]
        f = samples.frustums
        starts, ends = f.starts.reshape(-1), f.ends.reshape(-1)
        xyz, rgb = f.get_positions(), field[rgb_key]
        sigma = field[density_key].reshape(-1)
        if starts.shape != (n,) or sigma.shape != (n,) or rgb.shape != (n, 3) or xyz.shape != (n, 3):
            raise ValueError("Final field and packed arrays do not align")
        if not torch.equal(samples.spacing_starts, native["packed_spacing_starts"]) or not torch.equal(samples.spacing_ends, native["packed_spacing_ends"]):
            raise ValueError("Captured final samples differ from native packed outputs")
        if not torch.equal(f.origins, rays.origins[ids]) or not torch.equal(f.directions, rays.directions[ids]):
            raise ValueError("Packed sample/camera ray ordering differs")
        expected_xyz = rays.origins[ids]+rays.directions[ids]*((starts+ends)*.5)[:, None]
        if _max_error(xyz, expected_xyz) > 2e-7:
            raise ValueError("Physical XYZ/ray parameter chain differs")
        if _max_error(samples.deltas.reshape(-1), ends-starts) > 2e-7:
            raise ValueError("Actual physical deltas differ")
        if (ends <= starts).any() or (sigma < 0).any():
            raise ValueError("Positive physical intervals and nonnegative density required")
        if (ids[1:] == ids[:-1]).any():
            same = ids[1:] == ids[:-1]
            if (starts[1:][same] < ends[:-1][same]-2e-7).any():
                raise ValueError("Native packed intervals overlap or are unsorted")
        # Spacing is retained only as a native alignment check, never a distance.
        spacing_starts = samples.spacing_starts.reshape(-1)
        spacing_ends = samples.spacing_ends.reshape(-1)
    else:
        starts = ends = sigma = w.new_empty(0)
        xyz = rgb = w.new_empty((0, 3))
        spacing_starts = spacing_ends = w.new_empty(0)
    arrays = (starts, ends, sigma, xyz, rgb, w, native["rgb"], native["accumulation"])
    if any(not torch.isfinite(v).all() for v in arrays) or (w < 0).any():
        raise ValueError("Nonfinite/negative native capture")
    proof_weights = packed_density_weights(starts, ends, sigma, ids.long(), count)
    alpha = w.new_zeros(count).index_add(0, ids.long(), w)
    product = w[:, None]*rgb
    foreground = product.new_zeros((count, 3)).index_add(0, ids.long(), product)
    # Use actual renderer too, preserving its background override/policy. The
    # independent black composition above must agree, so overrides fail closed.
    composed = model.renderer_rgb(rgb=rgb, weights=w[:, None], ray_indices=ids, num_rays=count)
    errors = dict(weight=_max_error(w.double(), proof_weights),
        alpha=_max_error(alpha, native["accumulation"].reshape(-1)),
        RGB=_max_error(foreground, native["rgb"]),
        renderer_RGB=_max_error(composed, native["rgb"]))
    if max(errors.values()) > tolerance or (alpha > 1+tolerance).any():
        raise ValueError("Native density/weights/RGB/alpha proof failed: "+str(errors))
    counts = torch.bincount(ids.long(), minlength=count)
    if not torch.equal(counts, native["num_samples_per_ray"].reshape(-1).long()):
        raise ValueError("Native sample counts differ")
    return dict(xyz=xyz.detach(), starts=starts.detach(), ends=ends.detach(),
        sigma=sigma.detach(), rgb=rgb.detach(), weight=w.detach(), ray_indices=ids.detach(),
        spacing_starts=spacing_starts.detach(), spacing_ends=spacing_ends.detach(),
        native_rgb=native["rgb"].detach(), native_alpha=alpha.detach(),
        num_samples_per_ray=counts.detach(), cap_hit=counts >= model.config.max_steps_per_ray,
        proof=errors, field_forward_calls=len(captured))
