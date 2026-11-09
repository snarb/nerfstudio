"""Optional differentiable free-ray optical-thickness objective; never auto-installed.

Uses observed train SfM line segments only. These do not certify whole voxels or
hidden volume. No RGB, occupancy mutation, new trainable parameters or optimizer
steps. Caller explicitly composes the returned scalar with its training objective.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from staged_native_hash import file_sha


class FreeSpaceObjective(nn.Module):
    def __init__(self, manifest, variant, current_bounds, transforms, preparation, *, weight=.01, rays=128, samples=64, seed=42, device='cpu'):
        super().__init__()
        if not np.isfinite(weight) or weight < 0 or not isinstance(rays,int) or rays<1 or not isinstance(samples,int) or samples<1:
            raise ValueError('Nonnegative finite weight and positive ray/sample counts required')
        self.weight,self.rays,self.samples,self.seed=float(weight),rays,samples,int(seed)
        path=Path(manifest);payload=json.loads(path.read_text())
        if payload.get('kind')!='observed_train_sfm_free_ray_interval_candidates' or payload.get('coordinate_system')!='prepared_world' or payload.get('uses_eval_rgb') is not False or payload.get('uses_masks') is not False:
            raise ValueError('Expected prepared-world observed-train free-ray candidate artifact')
        for name,current in [('transforms_sha256',transforms),('normalization_sha256',preparation)]:
            if payload['bindings'].get(name)!=file_sha(current):raise ValueError('Current normalized dataset hash mismatch: '+name)
        selected=[v for v in payload['variants'] if v['name']==variant]
        if len(selected)!=1:raise ValueError('Expected exactly one artifact variant')
        record=selected[0];archive=(path.parent/record['file']).resolve()
        if archive.parent!=path.parent.resolve() or file_sha(archive)!=record['sha256']:raise ValueError('Artifact path/hash mismatch')
        with np.load(archive,allow_pickle=False) as data:
            bounds=np.array(current_bounds,dtype=np.float64)
            if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or not np.allclose(bounds,data['aabb'],rtol=0,atol=1e-6) or not np.allclose(bounds,record['aabb'],rtol=0,atol=1e-6):raise ValueError('Current normalized AABB mismatch')
            origins=data['origins'].copy();directions=data['directions'].copy();start=data['t_start'].copy();end=data['t_end'].copy();cameras=data['camera_idx'].copy();distance=data['witness_depth'];margin=data['surface_margin']
        n=len(start)
        if n<1 or origins.shape!=(n,3) or directions.shape!=(n,3) or end.shape!=(n,) or cameras.shape!=(n,) or not np.issubdtype(cameras.dtype,np.integer) or not np.isfinite(origins).all() or not np.isfinite(directions).all() or not np.isfinite([start,end,distance,margin]).all():raise ValueError('Invalid free-ray array shapes/values')
        if not np.allclose(np.linalg.norm(directions,axis=1),1,rtol=0,atol=1e-7) or (start<.01).any() or (end<=start).any() or (end>distance-margin+1e-10).any() or not np.allclose(margin,np.maximum(.01,.05*distance),rtol=0,atol=1e-10):raise ValueError('Invalid physical free-interval/margin policy')
        for t in [start,end]:
            xyz=origins+t[:,None]*directions
            if (xyz<bounds[0]-1e-6).any() or (xyz>bounds[1]+1e-6).any():raise ValueError('Free interval extends outside current normalized bounds')
        order=np.argsort(cameras,kind='stable');unique,counts=np.unique(cameras,return_counts=True)
        self.receipt={'manifest_sha256':file_sha(path),'archive_sha256':record['sha256'],'variant':variant,'coordinate_system':'prepared_world','policy':payload['policy'],'rays':n,'cameras':len(unique)}
        for name,value,dtype in [('origins',origins,torch.float32),('directions',directions,torch.float32),('starts',start,torch.float32),('ends',end,torch.float32),('camera_idx',cameras,torch.long),('camera_values',unique,torch.long),('csr_order',order,torch.long),('csr_offsets',np.concatenate(([0],np.cumsum(counts))),torch.long)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=dtype,device=device),persistent=False)

    def sample_batch(self, step):
        if not isinstance(step,int) or step<0:raise ValueError('step must be a nonnegative integer')
        # Seed depends only on declared seed and optimizer step; no global RNG state.
        gen=torch.Generator(device=self.origins.device)
        gen.manual_seed((self.seed+step*1_000_003)%(2**63-1))
        ncam=len(self.camera_values);base=self.rays//ncam;extra=self.rays%ncam
        order=torch.randperm(ncam,generator=gen,device=self.origins.device)
        selected_camera=torch.cat([order.repeat(base),order[:extra]])
        counts=self.csr_offsets[1:]-self.csr_offsets[:-1]
        random=torch.rand(self.rays,generator=gen,device=self.origins.device)
        local=torch.floor(random*counts[selected_camera]).long()
        ids=self.csr_order[self.csr_offsets[selected_camera]+local]
        jitter=torch.rand((self.rays,self.samples),generator=gen,device=self.origins.device,dtype=torch.float32)
        unit=(torch.arange(self.samples,device=self.origins.device,dtype=torch.float32)[None]+jitter)/self.samples
        length=self.ends[ids]-self.starts[ids]
        t=self.starts[ids,None]+length[:,None]*unit
        points=self.origins[ids,None,:]+self.directions[ids,None,:]*t[...,None]
        return {'indices':ids,'camera_idx':self.camera_idx[ids],'points':points,'t':t,'length':length}

    def forward(self,field,step):
        batch=self.sample_batch(step)
        # Field operations stay differentiable; accumulation explicitly uses FP32.
        density=field.density_fn(batch['points'].reshape(-1,3))
        if density.numel()!=self.rays*self.samples:raise ValueError('density_fn must return one density per point')
        with torch.autocast(device_type=self.origins.device.type,enabled=False):
            sigma=density.reshape(self.rays,self.samples).float()
            tau=sigma.mean(dim=1)*batch['length'].float()
            loss=tau.mean()*self.weight
        detached=tau.detach()
        metrics={'free_tau_mean':detached.mean(),'free_tau_p90':torch.quantile(detached,.9),'free_tau_max':detached.max(),'free_tau_min':detached.min(),'free_ray_count':detached.new_tensor(self.rays),'free_camera_count':detached.new_tensor(batch['camera_idx'].unique().numel())}
        return loss,metrics
