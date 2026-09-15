"""Opt-in point-to-camera angle and relative hard-source admission.

Neither function uses RGB to rank sources. Visibility is mandatory and supplied
by the existing renderer. No camera response, geometry or sampling is changed.
"""
import numpy as np


def ray_weights(points, centers, target_center, sigma_degrees):
    points=np.asarray(points);centers=np.asarray(centers);target=np.asarray(target_center)
    if points.ndim!=2 or points.shape[1]!=3 or centers.ndim!=2 or centers.shape[1]!=3 or target.shape!=(3,):
        raise ValueError('Invalid point/camera shapes')
    if not all(np.isfinite(a).all() for a in (points,centers,target)) or not np.isfinite(sigma_degrees) or sigma_degrees<=0:
        raise ValueError('Invalid angle inputs')
    source=centers[:,None]-points;query=target-points
    sl=np.linalg.norm(source,axis=-1);ql=np.linalg.norm(query,axis=-1)
    if (sl<=1e-12).any() or (ql<=1e-12).any():raise ValueError('Camera coincides with surface point')
    cosine=((source/sl[...,None])*(query/ql[:,None])[None]).sum(-1)
    angles=np.rad2deg(np.arccos(np.clip(cosine,-1,1)))
    return np.maximum(np.exp(-.5*(angles/sigma_degrees)**2),1e-5).astype(np.float32)


def admit_quality(quality,weights):
    q=np.asarray(quality)*np.asarray(weights)
    return np.where(q>=q.max(0)*.12,q,0)


def gather_relative(colors,weights,preferred,minimum_relative_weight):
    """Retain graph label only when its quality is >= ratio of best valid source."""
    import torch
    from hard_surface_texture import gather_hard_rgb
    if not np.isfinite(minimum_relative_weight) or not 0<=minimum_relative_weight<=1:
        raise ValueError('Invalid relative source threshold')
    if weights.ndim!=2 or preferred.shape!=(weights.shape[1],):raise ValueError('Invalid weights/labels')
    p=preferred.to(device=weights.device,dtype=torch.long)
    index=torch.arange(weights.shape[1],device=weights.device)
    safe=p.clamp(0,weights.shape[0]-1)
    good=(p>=0)&(p<weights.shape[0])&(weights[safe,index]>0)
    good&=weights[safe,index]>=minimum_relative_weight*weights.max(0).values
    return gather_hard_rgb(colors,weights,torch.where(good,p,-1))
