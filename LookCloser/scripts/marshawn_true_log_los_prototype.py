"""Uninstalled true-log-density LOS prototypes; never patches the frozen objective."""
import math
import torch
import torch.nn.functional as F
from marshawn_canonical_hdr_preflight import FrozenLookCloserDensityProxy

def loss_from_log_sigma(log_sigma,length,weight,mode):
    if mode not in('exact_tau','robust_log1p_tau'):raise ValueError('Explicit reviewed mode required')
    if log_sigma.ndim!=2 or length.shape!=(len(log_sigma),)or not torch.isfinite(log_sigma).all()or not torch.isfinite(length).all()or not(length>0).all():raise ValueError('Finite raw log-density and positive world intervals required')
    dtype=torch.float64 if mode=='exact_tau'or log_sigma.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=log_sigma.device.type,enabled=False):
        log_tau=torch.logsumexp(log_sigma.to(dtype),1)-math.log(log_sigma.shape[1])+length.to(dtype).log()
        if mode=='exact_tau':
            value=log_tau.exp()
            if not torch.isfinite(value).all():raise FloatingPointError('Exact true tau exceeds FP64 range; no silent clamp')
        else:value=F.softplus(log_tau)
        return float(weight)*value.mean(),log_tau

class TrueLogFreeSpacePrototype:
    """Plain accessor reusing exact validated samplegeometry/private step sampler."""
    def __init__(self,geometry,field,mode):
        self.geometry=geometry;self.proxy=FrozenLookCloserDensityProxy(field);self.mode=mode
    def __call__(self,step):
        b=self.geometry.sample_batch(step);p=b['points'].reshape(-1,3)
        if not((p>self.proxy.aabb[0])&(p<self.proxy.aabb[1])).all():raise ValueError('Strict inside-AABB points required; cannot replace zero selector by exp')
        sigma=self.proxy.density_fn(p);raw=self.proxy._density_before_activation
        log_sigma=raw.float().reshape(self.geometry.rays,self.geometry.samples)+math.log(self.proxy.average_init_density)
        loss,log_tau=loss_from_log_sigma(log_sigma,b['length'],self.geometry.weight,self.mode)
        return loss,dict(batch=b,raw=raw,log_sigma=log_sigma,log_tau=log_tau,native_sigma=sigma)
