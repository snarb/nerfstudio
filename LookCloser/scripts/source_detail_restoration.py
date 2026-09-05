"""Bounded within-source detail control, never a mixture of camera RGB.

This is an explicit diagnostic filter, NOT unchanged pointwise reprojection or
an identified optical deconvolution. A four-neighbor inverse-heat step has at
most 2x spectral gain. Visibility/depth/clipping guards forbid cross-surface taps.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import minimum_filter,maximum_filter
from patchmatch_color_calibration import decode_exposed_linear,encode_exposed_linear


def relative_restoration_amount(variance,available,fit_error):
    """Common relative-variance gauge cancels; no extrapolated/unseen reference."""
    if (variance.shape!=available.shape or variance.ndim<2 or not np.isfinite(variance).all()
            or not np.isfinite(fit_error) or fit_error<0):
        raise ValueError('Invalid source variance, availability or fit uncertainty')
    best=np.min(np.where(available,variance,np.inf),axis=0)
    two=available.sum(0)>=2
    excess=np.where(available&two,np.maximum(variance-best-2*fit_error,0),0)
    return np.minimum(excess*.5,.125).astype(np.float32)


def restore_source_detail(rgb,valid,depth,amount):
    """HWC display RGB -> HWC display RGB, using pixels of THIS source only."""
    if (rgb.ndim!=3 or rgb.shape[-1]!=3 or valid.shape!=rgb.shape[:2]
            or depth.shape!=valid.shape or amount.shape!=valid.shape
            or not np.isfinite(rgb).all() or (rgb<0).any() or (rgb>1).any()
            or not np.isfinite(depth).all() or not np.isfinite(amount).all()
            or (amount<0).any() or (amount>.125+1e-7).any()):
        raise ValueError('Invalid finite RGB/depth/visibility/restoration amount')
    unclipped=(rgb>.01).all(-1)&(rgb<.98).all(-1)
    safe=minimum_filter((valid&(depth>0)&unclipped).astype(np.uint8),size=3,mode='constant',cval=0)>0
    low=minimum_filter(depth,size=3,mode='constant',cval=0)
    high=maximum_filter(depth,size=3,mode='constant',cval=0)
    safe&=np.log(np.maximum(high,1e-9)/np.maximum(low,1e-9))<=.0075
    applied=np.where(safe,amount,0)
    linear=decode_exposed_linear(rgb)
    lap=4*linear-sum(np.roll(linear,delta,axis=axis) for axis in [0,1] for delta in [-1,1])
    candidate=linear+applied[...,None]*lap
    corrected=encode_exposed_linear(candidate)
    out=np.where((applied>0)[...,None],corrected,rgb).astype(rgb.dtype)
    return out,applied,dict(eligible_pixels=int(safe.sum()),filtered_pixels=int((applied>0).sum()),
        maximum_amount=float(applied.max()),negative_exposed_channels_clipped=int(((candidate<0)&(applied>0)[...,None]).sum()),
        source_rgb_averaging=False,filters_single_source_rgb=True,maximum_constant_coefficient_spectral_gain=2.,
        max_log_depth_range=.0075,filter_footprint='3x3 guarded, 4 axial neighbor taps',
        domain='inverse_srgb_inverse_reinhard_exposed_linear')
