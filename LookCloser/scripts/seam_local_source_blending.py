"""Opt-in source mixing restricted to depth-connected hard-label seams.

The seam is a change of texture source, not a semantic object boundary. Neither
the target RGB nor a skin/face/person segmentation participates in the rule.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from surface_color_field import solve_surface_field
from visible_source_blending import to_linear, to_display


def depth_graph(depth, support, max_depth_log_jump=.0075):
    logz=np.log(np.maximum(depth,1e-12))
    horizontal=support[:,:-1]&support[:,1:]&(np.abs(logz[:,:-1]-logz[:,1:])<max_depth_log_jump)
    vertical=support[:-1]&support[1:]&(np.abs(logz[:-1]-logz[1:])<max_depth_log_jump)
    return horizontal,vertical


def limited_graph_distance(seeds, horizontal, vertical, radius):
    """Four-connected shortest distance, without crossing a forbidden depth edge."""
    distance=np.full(seeds.shape,radius+1,np.int32);distance[seeds]=0
    frontier=seeds.copy()
    for step in range(1,radius+1):
        nxt=np.zeros_like(frontier)
        nxt[:,1:]|=frontier[:,:-1]&horizontal;nxt[:,:-1]|=frontier[:,1:]&horizontal
        nxt[1:]|=frontier[:-1]&vertical;nxt[:-1]|=frontier[1:]&vertical
        nxt&=distance>radius
        if not nxt.any():break
        distance[nxt]=step;frontier=nxt
    return distance


def seam_local_weights(valid, selection, depth, *, radius=32, visibility_feather=8.):
    """Only sources selected nearby on the same depth layer may enter a blend."""
    valid=np.asarray(valid);selection=np.asarray(selection);depth=np.asarray(depth)
    if (valid.ndim!=3 or not len(valid) or valid.dtype!=np.bool_ or selection.shape!=depth.shape
            or selection.shape!=valid.shape[1:] or not np.issubdtype(selection.dtype,np.integer)
            or not np.isfinite(depth).all() or (depth<0).any()
            or not isinstance(radius,int) or radius<1 or radius>1024
            or not math.isfinite(visibility_feather) or visibility_feather<=0
            or (selection<-1).any() or (selection>=len(valid)).any()):
        raise ValueError('Invalid seam geometry, labels or visibility parameters')
    support=selection>=0;yy,xx=np.indices(selection.shape)
    if not valid[np.maximum(selection,0),yy,xx][support].all() or (depth[support]<=0).any():
        raise ValueError('Selected source must be visible on positive target depth')
    horizontal,vertical=depth_graph(depth,support)
    sx=horizontal&(selection[:,:-1]!=selection[:,1:]);sy=vertical&(selection[:-1]!=selection[1:])
    seams=np.zeros_like(support)
    seams[:,:-1]|=sx;seams[:,1:]|=sx;seams[:-1]|=sy;seams[1:]|=sy
    distance=limited_graph_distance(seams,horizontal,vertical,radius)
    band=support&(distance<radius)
    hard=np.arange(len(valid))[:,None,None]==selection
    raw=[]
    for rank,visible in enumerate(valid):
        nearby=limited_graph_distance(hard[rank],horizontal,vertical,radius)
        weight=np.maximum(1-nearby/radius,0)**2
        # A source fades before its own occlusion, even if the hard seam lies
        # precisely at that visibility boundary. Never make an invisible source valid.
        margin=distance_transform_edt(np.pad(visible&support,1))[1:-1,1:-1]
        weight*=np.minimum(margin/visibility_feather,1)
        raw.append(np.where(band&visible,weight,hard[rank]).astype(np.float32))
    weights=np.stack(raw);total=weights.sum(0)
    if (total[support]<=0).any():raise RuntimeError('Lost original surface support')
    weights/=np.maximum(total,1e-12)
    assert not weights[~valid].any()
    stats=dict(radius=radius,visibility_feather=visibility_feather,max_depth_log_jump=.0075,
        seam_pixels=int(seams.sum()),band_pixels=int(band.sum()),supported_pixels=int(support.sum()),
        mixed_pixels=int(((weights>0).sum(0)>1).sum()),band_is_semantic_mask=False,
        outside_band_weights_exact_hard=True)
    return weights,band,stats


def blend_seam_sources(warped, valid, selection, depth, *, radius=32, visibility_feather=8., base_smoothness=64.):
    if (not math.isfinite(base_smoothness) or base_smoothness<=0 or len(warped)!=len(valid)
            or any(w.shape!=(3,*depth.shape) or not bool(torch.isfinite(w).all()) for w in warped)):
        raise ValueError('Invalid source images or base smoothness')
    weights_np,band_np,stats=seam_local_weights(valid.cpu().numpy(),selection.cpu().numpy(),depth.cpu().numpy(),
        radius=radius,visibility_feather=visibility_feather)
    weights=torch.as_tensor(weights_np,device=depth.device);band=torch.as_tensor(band_np,device=depth.device)
    visible=valid&(depth>0)
    rgb=torch.stack([torch.where(v[None],w,0) for w,v in zip(warped,visible)])
    linear=to_linear(rgb);selected=torch.zeros_like(rgb[0]);selected_linear=torch.zeros_like(rgb[0])
    for rank in range(len(warped)):
        selected=torch.where((selection==rank)[None],rgb[rank],selected)
        selected_linear=torch.where((selection==rank)[None],linear[rank],selected_linear)
    full=to_display((weights[:,None]*linear).sum(0))
    bases=[];solvers=[]
    for source,mask in zip(linear,visible):
        base,solver=solve_surface_field(source[None].double(),mask[None,None].double(),
            torch.where(mask,depth,0).double(),smoothness=base_smoothness,ridge=1e-6,max_iterations=2048,tolerance=1e-7)
        if solver['max_relative_residual']>5e-7:raise RuntimeError('Unconverged source low-pass')
        bases.append(base[0].to(linear.dtype));solvers.append(solver)
    bases=torch.stack(bases);selected_base=torch.zeros_like(selected)
    for rank in range(len(warped)):
        selected_base=torch.where((selection==rank)[None],bases[rank],selected_base)
    low_linear=(weights[:,None]*bases).sum(0)+selected_linear-selected_base
    low=to_display(low_linear)
    # Exact identity, including rounding and true object boundaries, wherever
    # the seam rule does not mix sources. No global change to the primary face.
    mixed=(weights>0).sum(0)>1
    outputs={name:torch.where(mixed[None],value,selected) for name,value in [('full_rgb',full),('low_band',low)]}
    stats.update(method='depth_connected_local_seam_blend',source_rgb_averaging=True,uses_eval_rgb=False,
        uses_semantic_masks=False,geometry_changed=False,visibility_changed=False,base_smoothness=base_smoothness,
        lowpass_solvers=solvers,domain='sRGB-linearized display RGB; Reinhard remains applied',
        outside_mixed_pixels_exact_rgb_identity=True,low_band_detail_source_labels_unchanged=True,
        low_band_clipped_channels=int((((low_linear<0)|(low_linear>1))&mixed[None]).sum()))
    return outputs,weights,band,stats
