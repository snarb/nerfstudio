"""Opt-in numerical pixel-center handling; nonzero footprint taps remain gated."""
import torch
from joint_temporal_texture import sample as original_sample


def snap_centers(uv, tolerance=.001):
    nearest=uv.round()
    return torch.where((uv-nearest).abs()<=tolerance,nearest,uv)


def relevant_tap(uv,dx,dy):
    fraction=(uv-uv.floor())[:,0]
    wx=fraction[...,0] if dx else 1-fraction[...,0]
    wy=fraction[...,1] if dy else 1-fraction[...,1]
    return (wx*wy)>0


def sample_native(images,uv):
    """Exact gather at integral coordinates avoids normalized-grid roundoff.

    Fractional coordinates retain the original bilinear sampler and border mode.
    No invalid tap is renormalized into RGB; no cross-camera color blending.
    """
    rounded=uv.round();integer=(uv==rounded).all(-1)
    x=rounded[...,0].long().clamp(0,images.shape[-1]-1)
    y=rounded[...,1].long().clamp(0,images.shape[-2]-1)
    indices=(y*images.shape[-1]+x).reshape(images.shape[0],1,-1).expand(-1,images.shape[1],-1)
    direct=torch.gather(images.flatten(2),2,indices).reshape(images.shape[0],images.shape[1],*uv.shape[1:3])
    if bool(integer.all()):return direct
    return torch.where(integer[:,None],direct,original_sample(images,uv))
