"""Opt-in exact-forward LOS with the true exponential derivative.

The frozen validated FreeSpaceObjective still owns sampling, geometry, FP32
reduction and metrics. Only its density accessor changes. No field modification,
new field query, NN parameters, occupancy edits or global RNG consumption.
"""
import math
import torch


def exact_forward_true_exp(native_sigma, raw, scale):
    """Keep native density bytes, replace its truncated-exp surrogate gradient.

    FP64 evaluates the derivative branch without prematurely underflowing at the
    raw-logit boundary. The zero-valued straight-through correction preserves
    the original forward, including its FP32 underflow. Nonfinite inputs fail;
    this helper neither clips density nor conceals numerical failures.
    """
    if native_sigma.numel() != raw.numel():
        raise ValueError('One cached raw logit per density sample required')
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError('Positive finite native density scale required')
    if not bool(torch.isfinite(native_sigma).all() and torch.isfinite(raw).all()):
        raise FloatingPointError('Nonfinite native density/raw logits')
    with torch.autocast(device_type=raw.device.type, enabled=False):
        true_sigma = raw.double().exp() * float(scale)
        if not bool(torch.isfinite(true_sigma).all()):
            raise FloatingPointError('True density exceeds FP64; no silent clamp')
        zero = (true_sigma - true_sigma.detach()).to(native_sigma.dtype)
        return native_sigma.detach() + zero.reshape_as(native_sigma)


class _TrueDensityView:
    def __init__(self, field):
        if hasattr(field, 'average_init_density'):
            self.field = field
        else:
            # Existing reviewed temporary h0 hook for the bounded LC field.
            from marshawn_canonical_hdr_preflight import FrozenLookCloserDensityProxy
            self.field = FrozenLookCloserDensityProxy(field)

    def density_fn(self, points):
        bounds = self.field.aabb
        if not bool(((points > bounds[0]) & (points < bounds[1])).all()):
            raise ValueError('True exponential requires strictly inside-AABB LOS points')
        sigma = self.field.density_fn(points)
        raw = self.field._density_before_activation  # Immediately after same query.
        return exact_forward_true_exp(sigma, raw, self.field.average_init_density)


class TrueTauFreeSpaceObjective:
    """Plain, non-state-dict adapter around an existing validated objective."""
    def __init__(self, existing):
        if isinstance(existing, TrueTauFreeSpaceObjective):
            raise ValueError('Do not install the true-LOS adapter twice')
        if not callable(existing) or not callable(getattr(existing, 'sample_batch', None)):
            raise TypeError('An existing validated FreeSpaceObjective is required')
        self.geometry = existing

    @classmethod
    def from_existing(cls, existing):
        return cls(existing)

    def __getattr__(self, name):
        return getattr(self.geometry, name)

    def __call__(self, field, step):
        return self.geometry(_TrueDensityView(field), step)
