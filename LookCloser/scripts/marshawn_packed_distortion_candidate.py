"""Opt-in native packed distortion proposal; no model/field source changes.

The exact frozen LookCloser FP64 packed formula is loaded from its AST. Hooks
exist only around the actual training forward and are removed even on failure.
No sampling, RNG, optimizer, occupancy or neural state changes are introduced.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import torch


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cpu_pack_info(ray_indices, num_rays):
    """CPU counterpart of nerfacc.pack_info's same index_add/cumsum algorithm."""
    counts = torch.zeros(num_rays, dtype=ray_indices.dtype, device=ray_indices.device)
    counts.index_add_(0, ray_indices, torch.ones_like(ray_indices))
    return torch.stack([counts.cumsum(0) - counts, counts], -1)


def load_frozen_formula(source, expected_sha256):
    path = Path(source)
    if file_sha(path) != expected_sha256:
        raise ValueError("Frozen packed-formula source SHA mismatch")
    tree = ast.parse(path.read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                 and node.name == "_packed_distortion_loss"]
    if len(functions) != 1:
        raise ValueError("Expected exactly one frozen packed distortion function")
    function = functions[0]
    function.decorator_list = []  # Remove class-only staticmethod wrapper, not its body.
    if "float64" not in ast.dump(function):
        raise ValueError("Expected the tested FP64 prefix implementation")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    import nerfacc
    gpu_namespace = {"torch": torch, "nerfacc": nerfacc}
    cpu_namespace = {"torch": torch, "nerfacc": SimpleNamespace(pack_info=_cpu_pack_info)}
    compiled = compile(module, str(path), "exec")
    exec(compiled, gpu_namespace)
    exec(compiled, cpu_namespace)
    def formula(starts, ends, weights, ray_indices, num_rays):
        namespace = gpu_namespace if weights.is_cuda else cpu_namespace
        value = namespace["_packed_distortion_loss"](starts, ends, weights, ray_indices, num_rays)
        # Preserve a zero gradient for all-empty packets without changing its value.
        return value + weights.sum() * 0
    formula.source_sha256 = expected_sha256
    formula.source_path = str(path)
    return formula


def ray_aabb_intervals(origins, directions, bounds, near_plane, far_plane):
    """World-distance ROI intersection; zero-direction slabs handled explicitly."""
    with torch.no_grad():
        origins, directions = origins.detach().double(), directions.detach().double()
        bounds = bounds.detach().double().reshape(2, 3)
        parallel = directions.abs() < 1e-12
        inverse = torch.where(parallel, torch.ones_like(directions), directions).reciprocal()
        a = (bounds[0] - origins) * inverse
        b = (bounds[1] - origins) * inverse
        low, high = torch.minimum(a, b), torch.maximum(a, b)
        outside = parallel & ((origins < bounds[0]) | (origins > bounds[1]))
        low = torch.where(parallel, torch.full_like(low, -torch.inf), low)
        high = torch.where(parallel, torch.full_like(high, torch.inf), high)
        enter = low.max(-1).values.clamp_min(float(near_plane))
        leave = high.min(-1).values.clamp_max(float(far_plane))
        valid = (~outside.any(-1)) & (leave > enter)
        return enter, leave, valid


def normalized_spacing(starts, ends, indices, near, far, mode):
    if mode not in ("linear", "inverse_depth"):
        raise ValueError("Explicit linear or inverse_depth coordinate required")
    n, f = near[indices], far[indices]
    if not torch.isfinite(n).all() or not torch.isfinite(f).all() or not (f > n).all() or not (n > 0).all():
        raise ValueError("Sampled rays need finite positive nonempty AABB intervals")
    start = torch.maximum(torch.minimum(starts.reshape(-1).double(), f), n)
    end = torch.maximum(torch.minimum(ends.reshape(-1).double(), f), n)
    if (end < start).any():
        raise ValueError("Native packed interval order is invalid")
    if mode == "linear":
        return ((start - n) / (f - n))[:, None], ((end - n) / (f - n))[:, None]
    denominator = n.reciprocal() - f.reciprocal()
    return ((n.reciprocal() - start.reciprocal()) / denominator)[:, None], \
           ((n.reciprocal() - end.reciprocal()) / denominator)[:, None]


def packet_terms(packet, formula, weight=.01, modes=("linear", "inverse_depth")):
    starts, ends, weights, indices = [packet[k] for k in ("starts", "ends", "weights", "ray_indices")]
    num_rays = packet["num_rays"]
    if not math.isfinite(float(weight)) or weight < 0 or num_rays < 0:
        raise ValueError("Finite nonnegative weight and ray count required")
    if not all(torch.isfinite(v).all() for v in (starts, ends, weights)) or (weights < 0).any():
        raise ValueError("Finite native intervals and nonnegative weights required")
    if indices.numel() and ((indices[1:] < indices[:-1]).any() or indices.min() < 0 or indices.max() >= num_rays):
        raise ValueError("Expected packed ray-sorted native samples")
    terms = {}
    for mode in modes:
        s, e = normalized_spacing(starts, ends, indices, packet["near"], packet["far"], mode)
        per_ray = formula(s, e, weights, indices, num_rays)
        terms[mode] = float(weight) * (per_ray.mean() if num_rays else weights.sum() * 0)
    return terms


class PackedDistortionCapture:
    """Context-scoped renderer pre-hook retaining native graph-connected weights."""
    def __init__(self, model, *, training_only=True):
        self.model, self.training_only = model, training_only
        self.handles, self.packet, self.ray_bundle = [], None, None
        self.calls = 0

    def __enter__(self):
        self.handles = [self.model.register_forward_pre_hook(self._model_pre, with_kwargs=True),
                        self.model.renderer_depth.register_forward_pre_hook(self._depth_pre, with_kwargs=True)]
        return self

    def _model_pre(self, module, args, kwargs):
        self.ray_bundle = None
        if self.training_only and not module.training:
            return
        if not torch.is_grad_enabled():
            return
        self.ray_bundle = kwargs.get("ray_bundle", args[0] if args else None)

    def _depth_pre(self, module, args, kwargs):
        if self.ray_bundle is None:
            return
        if args:
            raise ValueError("Reviewed native depth renderer must pass named packed arguments")
        weights, samples, indices, num_rays = [kwargs[k] for k in
            ("weights", "ray_samples", "ray_indices", "num_rays")]
        if weights.numel() and (not weights.requires_grad or weights.grad_fn is None):
            raise ValueError("Native weights were detached; distortion would be ineffective")
        bounds = self.model.occupancy_grid.aabbs[0].reshape(2, 3)
        near, far, valid = ray_aabb_intervals(self.ray_bundle.origins, self.ray_bundle.directions,
            bounds, self.model.config.near_plane, self.model.config.far_plane)
        if indices.numel() and not valid[indices].all():
            raise ValueError("Sampled native ray outside the declared AABB interval")
        self.packet = dict(starts=samples.frustums.starts, ends=samples.frustums.ends,
            weights=weights, ray_indices=indices, num_rays=int(num_rays), near=near, far=far,
            aabb=bounds.detach().clone())
        self.calls += 1

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.ray_bundle = None


def __getattr__(name):
    if name != "NativePackedDistortionPipeline":
        raise AttributeError(name)
    from marshawn_final_pipeline import FinalNativePipeline
    class NativePackedDistortionPipeline(FinalNativePipeline):
        """Only explicit pipeline objective/config changes; unchanged native NN."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.config.packed_distortion_mode not in ("linear", "inverse_depth"):
                raise ValueError("Explicit reviewed coordinate required")
            if not math.isfinite(self.config.packed_distortion_weight) or self.config.packed_distortion_weight < 0:
                raise ValueError("Nonnegative explicit distortion weight required")
            formula = load_frozen_formula(self.config.packed_distortion_source,
                                           self.config.packed_distortion_source_sha256)
            object.__setattr__(self, "_packed_formula", formula)

        def get_train_loss_dict(self, step):
            with PackedDistortionCapture(self.model) as capture:
                outputs, losses, metrics = super().get_train_loss_dict(step)
                if capture.calls != 1 or capture.packet is None:
                    raise ValueError("Expected exactly one native packed training forward")
                terms = packet_terms(capture.packet, self._packed_formula,
                    self.config.packed_distortion_weight, (self.config.packed_distortion_mode,))
                losses["packed_distortion"] = terms[self.config.packed_distortion_mode]
            return outputs, losses, metrics
    NativePackedDistortionPipeline.__name__ = name
    NativePackedDistortionPipeline.__qualname__ = name
    NativePackedDistortionPipeline.__module__ = __name__
    globals()[name] = NativePackedDistortionPipeline
    return NativePackedDistortionPipeline
