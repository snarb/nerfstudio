"""Single-source gradient guidance for additive hard-texture color correction.

Each seam edge takes its RGB difference from ONE train camera visible at both
endpoints. Non-seam edges retain the selected source's gradient. A screened
Poisson offset reconciles these constraints without averaging source RGB.
This is explicit gradient-domain color correction, not unchanged reprojection.
"""
from __future__ import annotations
import math
import torch


def level_source_gradients(prediction,selection,warped,valid_masks,depth,*,ridge=1e-6,
                           max_iterations=8192,tolerance=1e-9):
    if (prediction.ndim!=3 or prediction.shape[0]!=3 or selection.shape!=prediction.shape[1:]
            or depth.shape!=selection.shape or len(warped)!=len(valid_masks) or not len(warped)
            or not math.isfinite(ridge) or ridge<=0 or not math.isfinite(tolerance) or tolerance<=0):
        raise ValueError('Invalid RGB/source/depth inventory or solver parameters')
    if (not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(depth).all())
            or any(w.shape!=prediction.shape or not bool(torch.isfinite(w).all()) for w in warped)
            or any(v.shape!=selection.shape for v in valid_masks)
            or bool((selection>=len(warped)).any())):
        raise ValueError('Invalid finite RGB/source/depth inputs')
    device=prediction.device;support=(depth>0)&(selection>=0);logz=depth.clamp_min(1e-9).log()
    directions=[((slice(None),slice(None,-1)),(slice(None),slice(1,None))),
                ((slice(None,-1),slice(None)),(slice(1,None),slice(None)))]
    rhs=torch.zeros_like(prediction,dtype=torch.float64);degree=torch.zeros_like(depth,dtype=torch.float64)
    edges=[];rows=[]
    for a,b in directions:
        dz=(logz[a]-logz[b]).abs()
        geometrical=support[a]&support[b]&(dz<.0075)
        seam=geometrical&(selection[a]!=selection[b])
        chosen=torch.full_like(selection[a],-1)
        guidance=torch.zeros_like(prediction[(slice(None),*a)],dtype=torch.float64)
        for rank,(source,valid) in enumerate(zip(warped,valid_masks)):
            accept=seam&(chosen<0)&valid[a]&valid[b]
            delta=source[(slice(None),*a)].double()-source[(slice(None),*b)].double()
            guidance=torch.where(accept[None],delta,guidance)
            chosen=torch.where(accept,rank,chosen)
        connected=geometrical&(~seam|(chosen>=0))
        weight=connected.double()/(1+(dz.double()/.002).square())
        original=prediction[(slice(None),*a)].double()-prediction[(slice(None),*b)].double()
        target=torch.where((seam&(chosen>=0))[None],guidance-original,0.)
        rhs[(slice(None),*a)]+=weight*target;rhs[(slice(None),*b)]-=weight*target
        degree[a]+=weight;degree[b]+=weight
        edges.append((a,b,weight,target))
        rows.append(dict(seam_edges=int(seam.sum()),guided_edges=int((chosen>=0).sum()),
                         unknown_seam_edges_disconnected=int((seam&(chosen<0)).sum()),
                         guidance_source_edges=[int((chosen==r).sum()) for r in range(len(warped))]))
    diagonal=degree+ridge
    def matvec(value):
        result=diagonal*value
        for a,b,w,_ in edges:
            result[(slice(None),*a)]-=w*value[(slice(None),*b)]
            result[(slice(None),*b)]-=w*value[(slice(None),*a)]
        return result
    def dot(a,b):return (a*b).sum((-2,-1),keepdim=True)
    norm=dot(rhs,rhs).sqrt().clamp_min(1e-12)
    x=torch.zeros_like(rhs);r=rhs.clone();z=r/diagonal;p=z.clone();rz=dot(r,z)
    for iteration in range(1,max_iterations+1):
        ap=matvec(p);alpha=rz/dot(p,ap).clamp_min(1e-30)
        x+=alpha*p;r-=alpha*ap;z=r/diagonal;new_rz=dot(r,z)
        p=z+(new_rz/rz.clamp_min(1e-30))*p;rz=new_rz
        if iteration%16==0 and bool((dot(r,r).sqrt()/norm<tolerance).all()):break
    true=matvec(x)-rhs;relative=dot(true,true).sqrt()/norm
    solver=dict(iterations=iteration,max_relative_residual=float(relative.max()),
                required_true_relative_residual=tolerance*5,converged=bool((relative<tolerance*5).all()),
                dtype='float64',ridge=ridge)
    if not solver['converged']:raise RuntimeError(f'Gradient color solve failed: {solver}')
    raw=prediction.double()+x
    output=torch.where(support[None],raw.clamp(0,1),prediction).to(prediction.dtype)
    stats=dict(method='single_source_guided_screened_poisson_additive_display_correction',
        uses_eval_rgb=False,uses_semantic_masks=False,source_rgb_averaging=False,
        source_labels_unchanged=True,primary_unchanged=False,view_dependent=True,
        geometry_changed=False,visibility_changed=False,solver=solver,edges=rows,
        correction_mean_rgb=x[:,support].mean(-1).tolist() if bool(support.any()) else [0.,0.,0.],
        correction_abs_max=float(x.abs().max()),clipped_channels=int((((raw<0)|(raw>1))&support[None]).sum()))
    return output,x.to(prediction.dtype),stats
