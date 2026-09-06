"""Smooth train-only gain fields on the target mesh surface, never RGB blending.

Shared observations constrain the log gain relative to the primary train camera.
A depth-separated Laplacian extends that correction into disoccluded surface.
Only the correction field is smoothed; each output retains one source image.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from patchmatch_color_calibration import apply_camera_gain


def solve_surface_field(target,observed,depth,*,smoothness=64.,ridge=1e-4,max_iterations=1536,tolerance=1e-4,
                        edge_guide=None,max_color_jump=None):
    """Batched preconditioned CG for (observations + lambda L + ridge) gain=data."""
    if target.ndim!=4 or observed.shape!=(target.shape[0],1,*target.shape[-2:]) or depth.shape!=target.shape[-2:]:
        raise ValueError('Expected NCHW data, N1HW observations and HW depth')
    if (not math.isfinite(smoothness) or smoothness<=0 or ridge<=0 or tolerance<=0 or max_iterations<1
            or not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(depth).all())):
        raise ValueError('Invalid surface-field inputs or solver parameters')
    if ((edge_guide is None)!=(max_color_jump is None) or (edge_guide is not None and (
            edge_guide.shape!=(3,*depth.shape) or not bool(torch.isfinite(edge_guide).all())
            or not math.isfinite(max_color_jump) or max_color_jump<=0))):
        raise ValueError('Color-edge gate requires finite CHW guide and positive threshold')
    support=depth>0
    logz=depth.clamp_min(1e-7).log()
    dx=(logz[:,:-1]-logz[:,1:]).abs();dy=(logz[:-1]-logz[1:]).abs()
    wx=((dx<.0075)&support[:,:-1]&support[:,1:]).to(target.dtype)/(1+(dx/.002).square())
    wy=((dy<.0075)&support[:-1]&support[1:]).to(target.dtype)/(1+(dy/.002).square())
    color_edges_removed=0
    if edge_guide is not None:
        gx=(edge_guide[:,:,:-1]-edge_guide[:,:,1:]).square().mean(0)<max_color_jump**2
        gy=(edge_guide[:,:-1]-edge_guide[:,1:]).square().mean(0)<max_color_jump**2
        color_edges_removed=int(((wx>0)&~gx).sum()+((wy>0)&~gy).sum())
        wx=wx*gx;wy=wy*gy
    degree=F.pad(wx,(0,1))+F.pad(wx,(1,0))+F.pad(wy,(0,0,0,1))+F.pad(wy,(0,0,1,0))
    weight=observed.to(target.dtype)*support
    diagonal=weight+smoothness*degree+ridge
    def matvec(x):
        result=diagonal*x
        result[...,:-1]-=smoothness*wx*x[...,1:]
        result[...,1:]-=smoothness*wx*x[...,:-1]
        result[...,:-1,:]-=smoothness*wy*x[...,1:,:]
        result[...,1:,:]-=smoothness*wy*x[...,:-1,:]
        return result
    def dot(x,y):return (x*y).sum((-2,-1),keepdim=True)
    b=weight*target
    x=torch.zeros_like(target);r=b.clone();z=r/diagonal;p=z.clone();rz=dot(r,z)
    bnorm=dot(b,b).sqrt().clamp_min(1e-12)
    relative=dot(r,r).sqrt()/bnorm
    iteration=0
    for iteration in range(1,max_iterations+1):
        ap=matvec(p)
        alpha=rz/dot(p,ap).clamp_min(1e-30)
        x=x+alpha*p;r=r-alpha*ap
        z=r/diagonal;next_rz=dot(r,z)
        p=z+(next_rz/rz.clamp_min(1e-30))*p;rz=next_rz
        if iteration%16==0 or iteration==max_iterations:
            relative=dot(r,r).sqrt()/bnorm
            if bool((relative<tolerance).all()):break
    # Measure the true residual, not just the recursively updated CG residual.
    relative=dot(matvec(x)-b,matvec(x)-b).sqrt()/bnorm
    stats={'iterations':iteration,'max_relative_residual':float(relative.max()),
              'converged':bool((relative<max(tolerance*5,1e-3)).all()),'smoothness':smoothness,
              'ridge':ridge,'max_depth_log_jump':.0075,'depth_edge_scale':.002}
    if edge_guide is not None:stats.update(max_color_jump=max_color_jump,color_edges_removed=color_edges_removed)
    return x,stats


def correct_surface_colors(warped,valid_masks,depth,*,smoothness=64.,holdout=True):
    """Primary RGB unchanged; secondary colors receive a smooth monotonic gain."""
    if len(warped)<2:return warped,{'enabled':False,'reason':'one_source'}
    rgb=torch.stack(warped)
    valid=torch.stack(valid_masks)
    if rgb.ndim!=4 or rgb.shape[1]!=3 or valid.shape!=(rgb.shape[0],*rgb.shape[-2:]):
        raise ValueError('Expected matching RGB sources and geometric visibility')
    # Erosion discards inaccurate or interpolated observations at source occlusions.
    reliable=F.max_pool2d((~valid).float()[:,None],7,stride=1,padding=3)[:,0]==0
    reliable&=(rgb>.08).all(1)&(rgb<.92).all(1)&(depth>0)
    overlap=reliable[1:]&reliable[0]
    yy,xx=torch.meshgrid(torch.arange(depth.shape[0],device=depth.device),torch.arange(depth.shape[1],device=depth.device),indexing='ij')
    held=((((xx//32)*73856093)^((yy//32)*19349663))%5==0) if holdout else torch.zeros_like(depth,dtype=torch.bool)
    observed=overlap&~held
    linear=torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).pow(2.4))
    exposed=linear/(1-linear).clamp_min(1e-6)
    delta=(exposed[0:1].clamp_min(1e-7).log()-exposed[1:].clamp_min(1e-7).log()).clamp(-math.log(2),math.log(2))
    # Unknown RGB values have no influence on the linear system, even under NaNs.
    delta=torch.where(observed[:,None],delta,0)
    fields,stats=solve_surface_field(delta,observed[:,None],depth,smoothness=smoothness)
    fields=fields.clamp(-math.log(2),math.log(2))
    corrected=[warped[0]]+[apply_camera_gain(source,[1,1,1],field) for source,field in zip(warped[1:],fields)]
    rows=[]
    for i,field in enumerate(fields,1):
        check=overlap[i-1]&held
        before=(rgb[i]-rgb[0]).abs().mean(0)[check]
        after=(corrected[i]-rgb[0]).abs().mean(0)[check]
        rows.append({'source_rank':i,'fit_samples':int(observed[i-1].sum()),'held_samples':int(check.sum()),
                     'held_pair_display_l1_before':float(before.median()) if before.numel() else None,
                     'held_pair_display_l1_after':float(after.median()) if after.numel() else None,
                     'gain_min':float(field.exp().min()),'gain_max':float(field.exp().max())})
    stats.update(enabled=True,uses_eval_rgb=False,uses_semantic_masks=False,source_averaging=False,
                 primary_unchanged=True,view_dependent=True,domain='exposed_linear_log_gain',
                 holdout=holdout,holdout_block_pixels=32,camera_fits=rows)
    return corrected,stats
