"""Stage-only observed-SfM surface opacity term; no RGB, grid edits or parameters."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch import nn
from staged_native_hash import file_sha
from marshawn_los_objective_frozen import FreeSpaceObjective


def clipped_surface_bands(origins, directions, depths, bounds):
    """Float64 world intervals, clipped to AABB and native .01 near plane."""
    origins, directions, depths, bounds = [np.asarray(v, dtype=np.float64)
                                          for v in (origins,directions,depths,bounds)]
    n=len(depths)
    if origins.shape!=(n,3) or directions.shape!=(n,3) or bounds.shape!=(2,3):
        raise ValueError('Invalid band geometry shapes')
    if not all(np.isfinite(v).all() for v in [origins,directions,depths,bounds]):
        raise ValueError('Nonfinite band geometry')
    if (bounds[1]<=bounds[0]).any() or (depths<=.01).any() or not np.allclose(np.linalg.norm(directions,axis=1),1,rtol=0,atol=1e-7):
        raise ValueError('Invalid world bounds, depth or unit directions')
    parallel=np.abs(directions)<1e-12
    if (parallel & ((origins<bounds[0])|(origins>bounds[1]))).any():
        raise ValueError('Parallel ray outside AABB')
    safe=np.where(parallel,1.,directions)
    t0=(bounds[0]-origins)/safe;t1=(bounds[1]-origins)/safe
    low=np.minimum(t0,t1);high=np.maximum(t0,t1)
    low[parallel]=-np.inf;high[parallel]=np.inf
    enter=low.max(axis=1);exit=high.min(axis=1)
    half=np.maximum(.005,.01*depths)
    starts=np.maximum.reduce([depths-half,enter,np.full(n,.01)])
    ends=np.minimum(depths+half,exit)
    if (ends<=starts).any() or (depths<starts-1e-6).any() or (depths>ends+1e-6).any():
        raise ValueError('Witness outside nonempty clipped surface band')
    clipped=(starts>depths-half+1e-12)|(ends<depths+half-1e-12)
    return starts,ends,half,clipped


def opacity_nll(tau):
    """Density-derived reference for forward checks, not native auxiliary backward."""
    tau=tau.float()
    opacity=-torch.expm1(-tau)
    return -torch.log(opacity.clamp_min(torch.finfo(torch.float32).tiny))


def opacity_nll_from_log_tau(log_tau):
    """Bounded log-tau derivative, safe eager branches at arbitrarily low logits."""
    log_tau=log_tau.float()
    small=-log_tau+.5*torch.exp(log_tau.clamp(max=-10))
    regular_tau=torch.exp(log_tau.clamp(min=-10,max=15))
    regular=-torch.log(-torch.expm1(-regular_tau))
    return torch.where(log_tau < -10,small,regular)


class SurfaceBandObjective(nn.Module):
    def __init__(self,manifest,variant,current_bounds,transforms,preparation,*,
                 weight=.01,rays=128,samples=16,seed=42,device='cpu'):
        super().__init__()
        if not np.isfinite(weight) or weight<0 or rays!=128 or samples!=16:
            raise ValueError('Reviewed candidate requires weight>=0,128rays,16samples')
        # Reuse exact existing artifact/bounds/transforms/preparation validation.
        geometry=FreeSpaceObjective(manifest,variant,current_bounds,transforms,preparation,
                                    weight=0.,rays=rays,samples=1,seed=seed,device='cpu')
        path=Path(manifest);payload=json.loads(path.read_text())
        record=next(v for v in payload['variants']if v['name']==variant)
        archive=path.parent/record['file']
        with np.load(archive,allow_pickle=False)as data:
            origins=data['origins'].copy();directions=data['directions'].copy()
            depths=data['witness_depth'].copy();point=data['point_idx'].copy()
            cameras=data['camera_idx'].copy()
        pairs=np.unique(np.stack([point,cameras],axis=1),axis=0)
        _,views=np.unique(pairs[:,0],return_counts=True)
        if (views<3).any():raise ValueError('Every witness must retain>=3distinct train camera observations')
        starts,ends,half,clipped=clipped_surface_bands(origins,directions,depths,current_bounds)
        self.weight,self.rays,self.samples,self.seed=float(weight),rays,samples,int(seed)
        for name,value,dtype in [('origins',origins,torch.float32),('directions',directions,torch.float32),
            ('starts',starts,torch.float32),('ends',ends,torch.float32),('depths',depths,torch.float32),
            ('half_widths',half,torch.float32),('clipped',clipped,torch.bool)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=dtype,device=device),persistent=False)
        self.receipt=dict(**geometry.receipt,weight=self.weight,batch_rays=rays,samples_per_ray=samples,
            band_half_width='max(.005,.01*witness_depth)worldunits',clipped_bands=int(clipped.sum()),
            minimum_distinct_train_views=int(views.min()),target='opaque band probability1',
            formula='-weight*mean(log(-expm1(-tau_band)))',uses_rgb=False,uses_eval_rgb=False,uses_masks=False,
            native_gradient_path='Native density preactivation +log(average_init_density), logsumexp integral; surface auxiliary bypasses truncated-exp surrogate backward only',
            assumptions=['Observed SfM features are treated as opaque surfaces.',
            'Mirror/reflection, transparency and correspondence/depth errors can violate this assumption.',
            'Low-log-tau asymptotic NLL retains bounded logit gradients; no density epsilon or surface SDF is introduced.'])

    def sample_batch(self,step,indices):
        if not isinstance(step,int)or step<0:raise ValueError('Expected nonnegative integer globalstep')
        if indices.ndim!=1 or len(indices)!=self.rays or indices.dtype!=torch.long:
            raise ValueError('Supply exact128 int64 LOS batch indices')
        if (indices<0).any()or(indices>=len(self.starts)).any():raise ValueError('Band index out of range')
        gen=torch.Generator(device=self.origins.device)
        gen.manual_seed((self.seed+step*1_000_003+0x53555246)%(2**63-1))
        jitter=torch.rand((self.rays,self.samples),generator=gen,device=self.origins.device)
        unit=(torch.arange(self.samples,device=self.origins.device)[None]+jitter)/self.samples
        length=self.ends[indices]-self.starts[indices]
        t=self.starts[indices,None]+length[:,None]*unit
        return dict(indices=indices,t=t,length=length,
                    points=self.origins[indices,None]+self.directions[indices,None]*t[...,None])

    def forward(self,field,step,indices):
        batch=self.sample_batch(step,indices)
        points=batch['points'].reshape(-1,3)
        if not hasattr(field,'average_init_density')or getattr(field,'spatial_distortion',None)is not None:
            raise ValueError('Reviewed native bounded field attributes required')
        if not bool(((points>field.aabb[0])&(points<field.aabb[1])).all()):
            raise ValueError('Every queried band sample must be strictly inside native AABB')
        density=field.density_fn(points)
        if density.numel()!=self.rays*self.samples:raise ValueError('One density per band sample required')
        raw=field._density_before_activation
        if raw.numel()!=density.numel()or not math.isfinite(float(field.average_init_density))or field.average_init_density<=0:
            raise ValueError('Invalid native preactivation/density scale')
        with torch.autocast(device_type=self.origins.device.type,enabled=False):
            log_sigma=raw.reshape(self.rays,self.samples).float()+math.log(float(field.average_init_density))
            log_tau=torch.logsumexp(log_sigma,dim=1)-math.log(self.samples)+torch.log(batch['length'].float())
            nll=opacity_nll_from_log_tau(log_tau);loss=self.weight*nll.mean()
            tau_diagnostic=torch.exp(log_tau.detach().clamp(max=15));alpha=-torch.expm1(-tau_diagnostic)
        return loss,dict(surface_band_tau_capped_mean=tau_diagnostic.mean(),surface_band_opacity_mean=alpha.mean(),
            surface_band_opacity_p10=torch.quantile(alpha,.1),surface_band_opacity_min=alpha.min(),
            surface_band_raw_logit_min=raw.detach().float().min(),surface_band_raw_logit_max=raw.detach().float().max(),
            surface_band_log_tau_min=log_tau.detach().min(),surface_band_log_tau_max=log_tau.detach().max())
