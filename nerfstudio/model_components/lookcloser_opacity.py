"""Sampling-aware surface and opacity operations (independent of CUDA fields)."""
from __future__ import annotations

import math
import torch
from torch import Tensor


def first_crossing(sdf: Tensor) -> tuple[Tensor, Tensor]:
    """First sign change along each ray, accepting either surface orientation."""
    crossing = (sdf[..., :-1] * sdf[..., 1:] <= 0) & (
        (sdf[..., :-1] != 0) | (sdf[..., 1:] != 0))
    hit = crossing.any(-1)
    return crossing.to(torch.int64).argmax(-1), hit


def sdf_weights(sdf: Tensor, inverse_width: float) -> Tensor:
    """CDF-based training surrogate; foreground and room surfaces may face either way."""
    cdf = torch.sigmoid(sdf.float() * inverse_width)
    left, right = cdf[..., :-1], cdf[..., 1:]
    alpha = ((left-right).abs() / torch.maximum(left,right).clamp_min(1e-6)).clamp(0,1)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[...,:1]),(1-alpha).clamp_min(1e-7)],-1),-1)[...,:-1]
    return alpha * trans


def segmented_before(weights: Tensor, indices: Tensor, num_rays: int) -> Tensor:
    """Exclusive per-ray weight sums, avoiding float32 global-cumsum cancellation."""
    w = weights.reshape(-1)
    if not w.numel():
        return w
    cumulative = torch.cat([w.new_zeros(1,dtype=torch.float64),w.double().cumsum(0)])
    starts = torch.full((num_rays,),w.numel(),device=w.device,dtype=torch.long)
    starts.scatter_reduce_(0,indices,torch.arange(w.numel(),device=w.device),reduce='amin')
    return (cumulative[:-1] - cumulative[starts[indices]]).to(w.dtype)


def opacity_profile_loss(weights: Tensor, starts: Tensor, ends: Tensor,
                         indices: Tensor, num_rays: int) -> Tensor:
    """Integral T(1-T) in normalized [0,1] ray coordinates, including empty gaps.

    Within each interval sigma is constant. Integrate its exponential exactly;
    the value is invariant to splitting an interval into equivalent subintervals.
    """
    w, lo, hi = [v.reshape(-1).double() for v in (weights,starts,ends)]
    if not w.numel():
        return weights.sum()*0
    before = (1-segmented_before(w,indices,num_rays)).clamp(0,1)
    after = (before-w).clamp(0,1)
    tau = (before.clamp_min(1e-12).log()-after.clamp_min(1e-12).log()).clamp_min(0)
    average = (before-after)/tau.clamp_min(1e-8)
    average2 = (before.square()-after.square())/(2*tau.clamp_min(1e-8))
    middle = (before+after)*.5
    integral = torch.where(tau>1e-5,average-average2,middle*(1-middle)) * (hi-lo)
    next_start = torch.cat([lo[1:],lo.new_ones(1)])
    same_ray = torch.cat([indices[1:] == indices[:-1],indices.new_zeros(1,dtype=torch.bool)])
    gap = torch.where(same_ray,next_start,1.)-hi
    integral = integral + after*(1-after)*gap.clamp_min(0)
    return (integral.sum()/max(num_rays,1)).to(weights.dtype)


def depth_interval_loss(weights: Tensor, mids: Tensor, indices: Tensor, num_rays: int,
                        depth: Tensor, tolerance: Tensor, valid: Tensor,
                        objective: str = 'distance') -> Tensor:
    """Concentrate termination in a trusted interval; no constraints on invalid rays.

    Penalizing off-interval termination plus the missing in-interval mass avoids
    the two-layer cancellation permitted by expected-depth supervision.
    """
    w, t = weights.reshape(-1), mids.reshape(-1)
    d, eps = depth.reshape(-1), tolerance.reshape(-1).clamp_min(1e-6)
    trusted = valid.reshape(-1).bool()
    if not trusted.any():
        return w.sum()*0
    # Smooth admission provides gradients close to the interval boundary.
    admission = torch.sigmoid(8*(1-(t-d[indices]).abs()/eps[indices]))
    if objective == 'distance':
        distance = ((t-d[indices]).abs()-eps[indices]).clamp_min(0)/eps[indices]
        penalty = torch.where(distance<1,.5*distance.square(),distance-.5)
    elif objective == 'interval_mass':
        # Small remote layers must not dominate solely because of distance.
        # The missing-mass term still requires termination in the measured slab.
        penalty = 1-admission
    else:
        raise ValueError('Unknown depth interval objective')
    per_ray = w.new_zeros(num_rays).scatter_add(0,indices,w*penalty)
    mass = w.new_zeros(num_rays).scatter_add(0,indices,w*admission)
    return (per_ray[trusted]+(1-mass[trusted]).square()).mean()


def density_band_loss(band_sigma: Tensor, tolerance: Tensor, free_sigma: Tensor,
                      opacity_target: float = .99) -> Tensor:
    """Require an opaque measured slab and empty space ahead, without RGB resampling.

    Sigma is sampled uniformly across depth +/- tolerance. Trapezoid optical
    thickness is dimensionless, so narrowing the allowed slab requires a
    correspondingly denser surface. Only already-qualified rays enter here.
    """
    if not 0 < opacity_target < 1 or band_sigma.ndim != 2 or band_sigma.shape[1] < 2:
        raise ValueError('Valid opacity target and at least two band samples required')
    if len(band_sigma)==0:
        return band_sigma.sum()*0+free_sigma.sum()*0
    sigma=band_sigma.float().clamp_min(0)
    width=2*tolerance.reshape(-1).float()
    tau=torch.trapezoid(sigma,dx=1/(sigma.shape[1]-1),dim=-1)*width
    target_tau=-math.log1p(-opacity_target)
    missing=(1-tau/target_tau).clamp_min(0).square()
    # Linear tails keep pressure on large free-space spikes; an opacity
    # sigmoid would saturate and cease suppressing those unwanted surfaces.
    free_tau=free_sigma.float().clamp_min(0)*width[:,None]/target_tau
    free=torch.where(free_tau<1,.5*free_tau.square(),free_tau-.5).mean(-1)
    return (missing+free).mean()


def density_element_loss(sigma: Tensor, element_length: float, sparse_bias: float = .99) -> Tensor:
    """Bimodal element opacity with an L1 occupancy cost favoring sparse support.

    Convert physical density to opacity over one spatial element. Dense opaque
    elements have almost zero derivative, whereas diffuse intermediate elements
    are pushed toward zero. RGB can maintain an element when it is necessary.
    """
    if not math.isfinite(element_length) or element_length <= 0 or not 0 <= sparse_bias < 1:
        raise ValueError('Positive element length and sparse bias in [0,1) required')
    if not sigma.numel():return sigma.sum()*0
    alpha=-torch.expm1(-sigma.float().clamp_min(0)*element_length)
    return (alpha*(1-alpha)+sparse_bias*alpha).mean()


def floor_ray_geometry_loss(free_sigma: Tensor, free_length: Tensor,
                            band_sigma: Tensor, tolerance: Tensor,
                            free_monte_carlo: bool = False) -> Tensor:
    """Empty observed floor-ray segments plus opaque backing at measured floor."""
    if free_sigma.ndim!=2 or band_sigma.ndim!=2 or free_sigma.shape[1]<2 or band_sigma.shape[1]<2:
        raise ValueError('Floor integration requires at least two points per interval')
    if len(free_sigma)!=len(band_sigma) or free_length.shape!=(len(free_sigma),) or tolerance.shape!=free_length.shape:
        raise ValueError('Floor ray arrays must have matching batches')
    if (free_length<0).any() or (tolerance<=0).any():raise ValueError('Invalid floor interval lengths')
    if not len(free_sigma):return free_sigma.sum()*0+band_sigma.sum()*0
    positive=free_sigma.float().clamp_min(0)
    free_tau=(positive.mean(dim=-1) if free_monte_carlo else
              torch.trapezoid(positive,dx=1/(free_sigma.shape[1]-1),dim=-1))*free_length
    band_tau=torch.trapezoid(band_sigma.float().clamp_min(0),dx=1/(band_sigma.shape[1]-1),dim=-1)*2*tolerance
    missing=(1-band_tau/(-math.log(.01))).clamp_min(0).square()
    return (free_tau+missing).mean()
