"""Transport a smooth color base into hard-selected secondary texture patches.

Unlike seam-offset leveling, this replaces the secondary's low-frequency base
instead of preserving its interior shading gradient. Its own detail residual
is retained. Only correction fields are extended; RGB from different source
cameras is never pointwise averaged. This is an opt-in appearance assumption,
not unchanged reprojection or proof of physically correct hidden illumination.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import torch
from surface_color_field import solve_surface_field


def exposed_chromaticity(rgb):
    """Exposure-invariant RGB ratios under the fixed Reinhard/sRGB ingest."""
    linear=torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).pow(2.4))
    exposed=linear/(1-linear).clamp_min(1e-6)
    chroma=exposed/exposed.sum(0,keepdim=True).clamp_min(1e-8)
    reliable=(rgb>.03).all(0)&(rgb<.97).all(0)
    return chroma,reliable


def exposed_log_luminance(rgb):
    linear=torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).pow(2.4))
    exposed=linear/(1-linear).clamp_min(1e-6)
    weights=rgb.new_tensor([.2126,.7152,.0722])[:,None,None]
    return (exposed*weights).sum(0,keepdim=True).clamp_min(1e-7).log()


def bound_rgb_response(original,requested,*,max_gain=2.,max_chroma_shift=.025):
    """Project correction into bounded exposed-linear gain/chromaticity change.

    This transforms ONE selected source, not a mixture of source photographs.
    The chromaticity projection is a convex combination of two gain-bounded
    corrections, so it also preserves the per-channel gain bounds.
    """
    def inverse(rgb):
        linear=torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).pow(2.4))
        return linear/(1-linear).clamp_min(1e-6)
    if not math.isfinite(max_gain) or max_gain<1 or not math.isfinite(max_chroma_shift) or max_chroma_shift<=0:
        raise ValueError('Invalid bounded RGB response parameters')
    original_exposed=inverse(original)
    requested_exposed=inverse(requested.clamp(0,1))
    gain=(requested_exposed/original_exposed.clamp_min(1e-8)).clamp(1/max_gain,max_gain)
    limited=original_exposed*gain
    original_chroma=original_exposed/original_exposed.sum(0,keepdim=True).clamp_min(1e-8)
    total=limited.sum(0,keepdim=True)
    limited_chroma=limited/total.clamp_min(1e-8)
    distance=(limited_chroma-original_chroma).square().mean(0,keepdim=True).sqrt()
    alpha=(max_chroma_shift/distance.clamp_min(1e-8)).clamp_max(1)
    exposed=(original_chroma+alpha*(limited_chroma-original_chroma))*total
    mapped=exposed/(1+exposed)
    rgb=torch.where(mapped<=.0031308,mapped*12.92,1.055*mapped.pow(1/2.4)-.055).clamp(0,1)
    return rgb


def anchored_region(region,depth,seeds,*,guide=None,max_color_jump=None):
    """Only propagate over same-depth graph components with a measured boundary."""
    if region.shape!=depth.shape or region.shape!=seeds.shape or region.ndim!=2:
        raise ValueError('Expected matching region/depth/seed images')
    if ((guide is None)!=(max_color_jump is None) or (guide is not None and (
            guide.shape!=(*depth.shape,3) or not np.isfinite(guide).all()
            or not math.isfinite(max_color_jump) or max_color_jump<=0))):
        raise ValueError('Invalid color-edge anchor guide')
    region=np.asarray(region,bool)&np.isfinite(depth)&(depth>0)
    ids=np.full(region.shape,-1,np.int64);ids[region]=np.arange(region.sum())
    if not region.any():return region
    logz=np.log(np.maximum(depth,1e-9));aa=[];bb=[]
    for a,b in [((slice(None),slice(None,-1)),(slice(None),slice(1,None))),
                ((slice(None,-1),slice(None)),(slice(1,None),slice(None)))]:
        edge=region[a]&region[b]&(np.abs(logz[a]-logz[b])<.0075)
        if guide is not None:edge&=np.mean((guide[a]-guide[b])**2,-1)<max_color_jump**2
        aa.extend(ids[a][edge]);bb.extend(ids[b][edge])
    graph=coo_matrix((np.ones(len(aa)),(aa,bb)),shape=(int(region.sum()),)*2).tocsr()
    _,labels=connected_components(graph,directed=False)
    anchored=np.unique(labels[np.asarray(seeds,bool)[region]])
    result=np.zeros_like(region);result[region]=np.isin(labels,anchored)
    return result


def transport_source_bases(prediction,selection,warped,valid_masks,depth,*,base_smoothness=16.,strength=1.,
                           max_color_jump=None,max_seed_chroma_difference=None,luminance_only=False,bounded_rgb=False):
    """Replace only anchored secondary bases, preserving the primary and labels."""
    if (prediction.ndim!=3 or prediction.shape[0]!=3 or selection.shape!=depth.shape
            or selection.shape!=prediction.shape[1:] or len(warped)!=len(valid_masks) or not warped
            or not math.isfinite(base_smoothness) or base_smoothness<=0
            or not math.isfinite(strength) or not 0<=strength<=1
            or (max_color_jump is not None and (not math.isfinite(max_color_jump) or max_color_jump<=0))
            or (max_seed_chroma_difference is not None and (
                not math.isfinite(max_seed_chroma_difference) or max_seed_chroma_difference<=0))
            or (luminance_only and bounded_rgb)):
        raise ValueError('Invalid source/base transport inputs')
    if (not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(depth).all())
            or bool((selection>=len(warped)).any()) or bool((selection<-1).any())
            or any(w.shape!=prediction.shape or not bool(torch.isfinite(w).all()) for w in warped)
            or any(v.shape!=selection.shape or v.dtype!=torch.bool for v in valid_masks)):
        raise ValueError('Invalid finite RGB/depth/source inventory')
    support=(depth>0)&(selection>=0)
    for rank,valid in enumerate(valid_masks):
        if bool(((selection==rank)&~valid).any()):raise ValueError('Selected source is invisible')
    stats=dict(method='hard_source_harmonic_base_replacement',base_smoothness=base_smoothness,strength=strength,
               uses_eval_rgb=False,uses_semantic_masks=False,source_rgb_averaging=False,
               source_labels_unchanged=True,primary_unchanged=True,geometry_changed=False,visibility_changed=False,
               view_dependent=True,unchanged_pointwise_reprojection=False,
               retained_detail='selected source minus its screened same-source low-pass base',patches=[])
    if max_color_jump is not None:stats['max_color_jump']=max_color_jump
    if max_seed_chroma_difference is not None:stats['max_seed_chroma_difference']=max_seed_chroma_difference
    if luminance_only:
        stats.update(method='hard_source_harmonic_log_luminance_base_replacement',
                     retained_detail='selected source multiplied by a smooth scalar exposed-linear gain',
                     exposed_chromaticity_preserved=True,maximum_gain=2.,minimum_gain=.5)
    if bounded_rgb:
        stats.update(method='hard_source_harmonic_base_bounded_rgb_response',maximum_gain=2.,minimum_gain=.5,
                     maximum_exposed_chroma_rms_shift=.025,
                     retained_detail='one selected source under bounded pointwise RGB response; not an exact additive detail residual')
    output=prediction.clone();offset=torch.zeros_like(prediction)
    if strength==0 or not bool((support&(selection>0)).any()):
        stats.update(correction_abs_max=0.,clipped_channels=0);return output,offset,stats
    bases=[];lowpass_stats=[]
    chroma_sources=[exposed_chromaticity(source) for source in warped] if max_seed_chroma_difference is not None else None
    reference_chroma,reference_reliable=exposed_chromaticity(prediction) if chroma_sources is not None else (None,None)
    for rank,(source,valid) in enumerate(zip(warped,valid_masks)):
        local_depth=torch.where(valid,depth,0)
        # Smoothing is within ONE camera's valid, same-depth observation graph.
        base_source=exposed_log_luminance(source) if luminance_only else source
        data=torch.where(valid[None],base_source,0).double()[None]
        base,solver=solve_surface_field(data,(valid&(depth>0))[None,None].double(),local_depth.double(),
            smoothness=base_smoothness,ridge=1e-6,max_iterations=2048,tolerance=1e-7,
            edge_guide=source if max_color_jump is not None else None,max_color_jump=max_color_jump)
        if solver['max_relative_residual']>5e-7:raise RuntimeError(f'Source low-pass failed: {solver}')
        bases.append(base[0].to(prediction.dtype));lowpass_stats.append(solver)
    selected_base=torch.zeros_like(bases[0])
    for rank,base in enumerate(bases):selected_base=torch.where((selection==rank)[None],base,selected_base)
    logz=depth.clamp_min(1e-9).log();height,width=depth.shape
    clipped=0
    for rank in range(1,len(warped)):
        region=(selection==rank)&support
        if not bool(region.any()):continue
        chosen=torch.full_like(selection,len(warped));data=torch.zeros_like(selected_base)
        rejected_chroma_seeds=0
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            y0,y1=max(0,-dy),min(height,height-dy);x0,x1=max(0,-dx),min(width,width-dx)
            a=(slice(y0,y1),slice(x0,x1));b=(slice(y0+dy,y1+dy),slice(x0+dx,x1+dx))
            lower=selection[b]
            accept=region[a]&(lower>=0)&(lower<rank)&(lower<chosen[a])&valid_masks[rank][b]
            accept&=(depth[b]>0)&((logz[a]-logz[b]).abs()<.0075)
            if max_color_jump is not None:
                accept&=(warped[rank][(slice(None),*a)]-warped[rank][(slice(None),*b)]).square().mean(0)<max_color_jump**2
            if chroma_sources is not None:
                chroma,reliable=chroma_sources[rank]
                compatible=reference_reliable[b]&reliable[b]&((
                    reference_chroma[(slice(None),*b)]-chroma[(slice(None),*b)]).square().mean(0)<max_seed_chroma_difference**2)
                rejected_chroma_seeds+=int((accept&~compatible).sum())
                accept&=compatible
            # One lower-rank neighboring source sets each seed, with the same
            # secondary's local base slope transporting it by one pixel.
            desired=selected_base[(slice(None),*b)]+bases[rank][(slice(None),*a)]-bases[rank][(slice(None),*b)]
            data[(slice(None),*a)]=torch.where(accept[None],desired,data[(slice(None),*a)])
            chosen[a]=torch.where(accept,lower,chosen[a])
        seeds=chosen<len(warped)
        anchored=anchored_region(region.cpu().numpy(),depth.cpu().numpy(),seeds.cpu().numpy(),
            guide=warped[rank].permute(1,2,0).cpu().numpy() if max_color_jump is not None else None,max_color_jump=max_color_jump)
        anchored=torch.as_tensor(anchored,device=depth.device)
        row=dict(rank=rank,pixels=int(region.sum()),seeds=int(seeds.sum()),anchored_pixels=int(anchored.sum()))
        if chroma_sources is not None:row['rejected_chroma_seed_edges']=rejected_chroma_seeds
        if not bool(anchored.any()):
            row['corrected']=False;stats['patches'].append(row);continue
        yy,xx=torch.where(anchored);y0,y1=int(yy.min()),int(yy.max())+1;x0,x1=int(xx.min()),int(xx.max())+1
        local=anchored[y0:y1,x0:x1]
        transported,solver=solve_surface_field(data[None,:,y0:y1,x0:x1].double(),
            seeds[None,None,y0:y1,x0:x1].double()*128,
            torch.where(local,depth[y0:y1,x0:x1],0).double(),
            smoothness=1.,ridge=1e-6,max_iterations=8192,tolerance=1e-9,
            edge_guide=warped[rank][:,y0:y1,x0:x1] if max_color_jump is not None else None,max_color_jump=max_color_jump)
        if solver['max_relative_residual']>5e-9:raise RuntimeError(f'Base transport failed: {solver}')
        new_base=transported[0].to(prediction.dtype)
        old_base=bases[rank][:,y0:y1,x0:x1]
        delta=strength*(new_base-old_base)
        if luminance_only:
            from patchmatch_color_calibration import apply_camera_gain
            delta=delta.clamp(-math.log(2),math.log(2))
            raw=apply_camera_gain(prediction[:,y0:y1,x0:x1],[1,1,1],delta[0])
            display_delta=raw-prediction[:,y0:y1,x0:x1]
            row.update(log_gain_abs_max=float(delta[:,local].abs().max()),
                       gain_min=float(delta[:,local].exp().min()),gain_max=float(delta[:,local].exp().max()))
        else:
            raw=prediction[:,y0:y1,x0:x1]+delta
            if bounded_rgb:
                raw=bound_rgb_response(prediction[:,y0:y1,x0:x1],raw)
                display_delta=raw-prediction[:,y0:y1,x0:x1]
            else:display_delta=delta
        clipped+=int((((raw<0)|(raw>1))&local[None]).sum())
        output[:,y0:y1,x0:x1]=torch.where(local[None],raw.clamp(0,1),output[:,y0:y1,x0:x1])
        offset[:,y0:y1,x0:x1]=torch.where(local[None],display_delta,offset[:,y0:y1,x0:x1])
        # Later ranks must see the APPLIED bounded correction, not its rejected
        # unconstrained request. The luminance branch stores a log-intensity base.
        applied_base_delta=display_delta if bounded_rgb else delta
        selected_base[:,y0:y1,x0:x1]=torch.where(local[None],old_base+applied_base_delta,selected_base[:,y0:y1,x0:x1])
        row.update(corrected=True,solver=solver,correction_abs_max=float(display_delta[:,local].abs().max()))
        stats['patches'].append(row)
    stats.update(lowpass_solvers=lowpass_stats,correction_abs_max=float(offset.abs().max()),clipped_channels=clipped)
    return output,offset,stats
