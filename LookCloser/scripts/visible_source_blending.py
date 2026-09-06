"""Opt-in geometrically visible multi-camera blending, explicitly not hard RGB.

Visibility feathering suppresses abrupt hand-occlusion handoffs. Full RGB and a
low-frequency-only mixture are matched controls; the latter retains detail from
the existing hard source label. No target RGB or semantic segmentation is used.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from surface_color_field import solve_surface_field


def to_linear(rgb):
    return torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).pow(2.4))


def to_display(linear):
    linear=linear.clamp(0,1)
    return torch.where(linear<=.0031308,linear*12.92,1.055*linear.pow(1/2.4)-.055).clamp(0,1)


def visibility_weights(valid_masks,camera_distances,*,feather_pixels=32.):
    """Camera-distance prior times distance to geometric visibility boundary."""
    distances=np.asarray(camera_distances,dtype=np.float64)
    if (len(valid_masks)!=len(distances) or not len(distances) or not np.isfinite(distances).all()
            or (distances<0).any() or not math.isfinite(feather_pixels) or feather_pixels<=0
            or any(v.shape!=valid_masks[0].shape or v.dtype!=torch.bool for v in valid_masks)):
        raise ValueError('Invalid visibility/geometry weight inputs')
    bandwidth=max(float(np.sort(distances)[min(2,len(distances)-1)]),1e-6)
    prior=np.maximum(np.exp(-.5*(distances/bandwidth)**2),1e-6)
    weights=[]
    for valid,weight in zip(valid_masks,prior):
        footprint=valid.cpu().numpy()
        # Pad to include the image edge as a visibility boundary.
        distance=distance_transform_edt(np.pad(footprint,1))[1:-1,1:-1]
        taper=np.minimum(distance/feather_pixels,1)*weight
        weights.append(torch.as_tensor(taper,device=valid.device,dtype=torch.float32))
    weights=torch.stack(weights);total=weights.sum(0)
    weights=weights/total.clamp_min(1e-12)
    return weights,dict(camera_distances=distances.tolist(),distance_bandwidth=bandwidth,
                        geometric_priors=prior.tolist(),feather_pixels=feather_pixels)


def blend_visible_sources(warped,valid_masks,selection,depth,camera_distances,*,feather_pixels=32.,base_smoothness=64.):
    if (len(warped)!=len(valid_masks) or not len(warped) or not math.isfinite(base_smoothness) or base_smoothness<=0
            or depth.shape!=selection.shape or any(w.shape!=(3,*depth.shape) for w in warped)
            or any(v.shape!=depth.shape for v in valid_masks) or not bool(torch.isfinite(depth).all())
            or any(not bool(torch.isfinite(w).all()) for w in warped)
            or bool((selection>=len(warped)).any()) or bool((selection<-1).any())):
        raise ValueError('Invalid RGB/depth/source inventory')
    for rank,valid in enumerate(valid_masks):
        if bool(((selection==rank)&~valid).any()):raise ValueError('Hard detail source is invisible')
    valid=[v&(depth>0) for v in valid_masks]
    weights,weight_stats=visibility_weights(valid,camera_distances,feather_pixels=feather_pixels)
    support=weights.sum(0)>0
    if bool((support&(selection<0)).any()):raise ValueError('Visible surface lacks a hard detail source')
    rgb=torch.stack([torch.where(v[None],w,0) for w,v in zip(warped,valid)])
    # Linearize sRGB only. Do not invert Reinhard on quantized near-white JPEG
    # pixels: this is display-image compositing, not recovered raw radiance.
    linear=to_linear(rgb)
    full_linear=(weights[:,None]*linear).sum(0)
    bases=[];solvers=[]
    for image,visible in zip(linear,valid):
        base,solver=solve_surface_field(image[None].double(),visible[None,None].double(),
            torch.where(visible,depth,0).double(),smoothness=base_smoothness,ridge=1e-6,
            max_iterations=2048,tolerance=1e-7)
        if solver['max_relative_residual']>5e-7:raise RuntimeError(f'Source low-pass failed: {solver}')
        bases.append(base[0].to(linear.dtype));solvers.append(solver)
    bases=torch.stack(bases)
    mixed_base=(weights[:,None]*bases).sum(0)
    selected=torch.zeros_like(full_linear);selected_base=torch.zeros_like(full_linear)
    for rank in range(len(warped)):
        selected=torch.where((selection==rank)[None],linear[rank],selected)
        selected_base=torch.where((selection==rank)[None],bases[rank],selected_base)
    detail=selected-selected_base
    low_linear=mixed_base+detail
    full=torch.where(support[None],to_display(full_linear),0)
    low=torch.where(support[None],to_display(low_linear),0)
    effective=1/weights.square().sum(0).clamp_min(1e-12)
    stats=dict(method='visibility_feathered_multicamera_blending',source_rgb_averaging=True,
        full_rgb='weighted linear-display RGB mixture',low_band='weighted smooth bases plus one hard source detail residual',
        low_band_source_detail_labels_unchanged=True,uses_eval_rgb=False,uses_semantic_masks=False,
        geometry_changed=False,visibility_changed=False,primary_rgb_unchanged=False,
        domain='sRGB-linearized display RGB; Reinhard remains applied',weights=weight_stats,
        base_smoothness=base_smoothness,lowpass_solvers=solvers,
        supported_pixels=int(support.sum()),effective_sources_median=float(effective[support].median()) if bool(support.any()) else 0.,
        low_band_clipped_channels=int((((low_linear<0)|(low_linear>1))&support[None]).sum()))
    return {'full_rgb':full,'low_band':low},weights,stats
