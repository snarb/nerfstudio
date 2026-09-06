"""Blend all visible fallback cameras only near/inside primary-source occlusion."""
from __future__ import annotations
import numpy as np
import torch
from seam_local_source_blending import depth_graph,limited_graph_distance,seam_local_weights
from surface_color_field import solve_surface_field
from visible_source_blending import visibility_weights,to_linear,to_display


def fallback_weights(valid,selection,depth,camera_distances,*,radius=32,visibility_feather=8.):
    # Reuse strict source/label/depth validation, not its source-local weights.
    _,_,_=seam_local_weights(valid.cpu().numpy(),selection.cpu().numpy(),depth.cpu().numpy(),
        radius=radius,visibility_feather=visibility_feather)
    support=(selection>=0).cpu().numpy();native_depth=depth.cpu().numpy()
    horizontal,vertical=depth_graph(native_depth,support)
    primary_missing=support&~valid[0].cpu().numpy()
    distance=limited_graph_distance(primary_missing,horizontal,vertical,radius)
    t=np.minimum(distance/radius,1);alpha=t*t*(3-2*t)
    band=support&(distance<radius)
    base,prior=visibility_weights(list(valid&(depth>0)),camera_distances,feather_pixels=visibility_feather)
    secondary=base[1:].sum(0)
    alpha_t=torch.as_tensor(alpha,device=depth.device,dtype=torch.float32)*valid[0]
    alpha_t=torch.where((secondary<=0)&valid[0],1,alpha_t)
    weights=base.clone();weights[0]=alpha_t
    weights[1:]=base[1:]/secondary.clamp_min(1e-12)*(1-alpha_t)
    hard=torch.arange(len(valid),device=depth.device)[:,None,None]==selection
    band_t=torch.as_tensor(band,device=depth.device)
    weights=torch.where(band_t[None],weights,hard.float())
    assert not bool(weights[~valid].any())
    stats=dict(radius=radius,visibility_feather=visibility_feather,max_depth_log_jump=.0075,
        missing_primary_pixels=int(primary_missing.sum()),band_pixels=int(band.sum()),supported_pixels=int(support.sum()),
        mixed_pixels=int(((weights>0).sum(0)>1).sum()),weights=prior,
        scope='All visible fallback cameras inside primary occlusion and its same-depth feather band',
        band_is_semantic_mask=False,outside_band_weights_exact_hard=True)
    return weights,band_t,stats


def blend_fallback_sources(warped,valid,selection,depth,camera_distances,*,radius=32,visibility_feather=8.,base_smoothness=64.):
    if (not np.isfinite(base_smoothness) or base_smoothness<=0 or len(warped)!=len(valid)
            or any(w.shape!=(3,*depth.shape) or not bool(torch.isfinite(w).all()) for w in warped)):
        raise ValueError('Invalid fallback RGB sources or base smoothness')
    weights,band,stats=fallback_weights(valid,selection,depth,camera_distances,radius=radius,visibility_feather=visibility_feather)
    visible=valid&(depth>0)
    rgb=torch.stack([torch.where(v[None],w,0) for w,v in zip(warped,visible)])
    linear=to_linear(rgb);selected=torch.zeros_like(rgb[0]);selected_linear=torch.zeros_like(rgb[0])
    for rank in range(len(warped)):
        selected=torch.where((selection==rank)[None],rgb[rank],selected)
        selected_linear=torch.where((selection==rank)[None],linear[rank],selected_linear)
    full=to_display((weights[:,None]*linear).sum(0))
    bases=[];solvers=[]
    for source,mask in zip(linear,visible):
        base,solver=solve_surface_field(source[None].double(),mask[None,None].double(),torch.where(mask,depth,0).double(),
            smoothness=base_smoothness,ridge=1e-6,max_iterations=2048,tolerance=1e-7)
        if solver['max_relative_residual']>5e-7:raise RuntimeError('Unconverged fallback source low-pass')
        bases.append(base[0].to(linear.dtype));solvers.append(solver)
    bases=torch.stack(bases);selected_base=torch.zeros_like(selected)
    for rank in range(len(warped)):
        selected_base=torch.where((selection==rank)[None],bases[rank],selected_base)
    low_linear=(weights[:,None]*bases).sum(0)+selected_linear-selected_base
    # Identity outside the primary-occlusion band; inside, a different sole
    # visible fallback may legitimately replace the original hard detail source.
    outputs={name:torch.where(band[None],value,selected) for name,value in [('full_rgb',full),('low_band',to_display(low_linear))]}
    stats.update(method='primary_occlusion_fallback_blend',source_rgb_averaging=True,uses_eval_rgb=False,
        uses_semantic_masks=False,geometry_changed=False,visibility_changed=False,base_smoothness=base_smoothness,
        lowpass_solvers=solvers,domain='sRGB-linearized display RGB; Reinhard remains applied',
        outside_band_exact_rgb_identity=True,low_band_detail_source_labels_unchanged=True,
        low_band_clipped_channels=int((((low_linear<0)|(low_linear>1))&band[None]).sum()))
    return outputs,weights,band,stats
