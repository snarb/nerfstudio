"""Explicit mask-aware actor geometry; no historical guard is weakened.

Witness LOS has the native forward density and true exponential derivative.
Positive SfM bands use bounded log-tau NLL gradients. Semantic absence is a
separate soft hypothesis, never an alpha target or hard occupancy carve.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from marshawn_true_los import exact_forward_true_exp

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stable_opacity_nll(log_tau):
    x=log_tau.float();small=-x+.5*torch.exp(x.clamp(max=-10))
    tau=torch.exp(x.clamp(min=-10,max=15));regular=-torch.log(-torch.expm1(-tau))
    return torch.where(x< -10,small,regular)

def load_actor_geometry(manifest,variant,bounds,mask_manifest,transforms,preparation):
    path=Path(manifest);m=json.loads(path.read_text());mask=json.loads(Path(mask_manifest).read_text())
    if m.get('kind')!='observed_train_sfm_free_ray_interval_candidates'or m.get('uses_masks')is not True or m.get('uses_eval_rgb')is not False or m.get('coordinate_system')!='prepared_world':
        raise ValueError('Explicit prepared-world actor semantic TRAIN geometry required')
    selection=m['actor_semantic_selection']
    if (selection.get('sha256')!=sha(mask_manifest)or Path(selection['manifest']).resolve()!=Path(mask_manifest).resolve()
        or selection.get('train63_only')is not True or selection.get('minimum_actual_track_camera_votes',0)<3
        or selection.get('soft_band_is_NOT_geometry_truth')is not True):raise ValueError('Actor semantic selection contract mismatch')
    for key,file in [('transforms_sha256',transforms),('normalization_sha256',preparation)]:
        if m['bindings'][key]!=sha(file):raise ValueError('Current actor geometry source mismatch '+key)
    records=[r for r in m['variants']if r['name']==variant]
    if len(records)!=1:raise ValueError('One exact actor geometry variant required')
    record=records[0];archive=path.parent/record['file']
    if archive.resolve().parent!=path.resolve().parent or sha(archive)!=record['sha256']:raise ValueError('Actor geometry archive changed')
    with np.load(archive,allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
    b=np.asarray(bounds,np.float64)
    if b.shape!=(2,3)or not np.allclose(data['aabb'],b,rtol=0,atol=1e-6)or not np.allclose(record['aabb'],b,rtol=0,atol=1e-6):raise ValueError('Actor AABB mismatch')
    n=len(data['t_start']);o,d=data['origins'],data['directions'];start,end=data['t_start'],data['t_end'];depth=data['witness_depth'];margin=data['surface_margin'];ids=data['point_idx'];cams=data['camera_idx']
    if n<1 or o.shape!=(n,3)or d.shape!=(n,3)or any(data[k].shape!=(n,)for k in ['t_start','t_end','witness_depth','surface_margin','point_idx','camera_idx']):raise ValueError('Invalid actor witness geometry shapes')
    if not all(np.isfinite(data[k]).all()for k in ['origins','directions','t_start','t_end','witness_depth','surface_margin','witness_xyz','uv']):raise ValueError('Nonfinite actor geometry')
    if not np.allclose(np.linalg.norm(d,axis=1),1,rtol=0,atol=1e-7)or(start<.052-1e-9).any()or(end<=start).any()or(end>depth-margin+1e-9).any()or not np.allclose(margin,np.maximum(.01,.05*depth),rtol=0,atol=1e-10):raise ValueError('Invalid native-near/witness-margin geometry')
    if not np.allclose(o+d*depth[:,None],data['witness_xyz'],rtol=0,atol=1e-7):raise ValueError('Witness position/ray mismatch')
    pairs=np.unique(np.stack([ids,cams],1),axis=0);_,votes=np.unique(pairs[:,0],return_counts=True)
    if (votes<3).any():raise ValueError('Every actor point needs >=3 actual distinct TRAIN track cameras')
    train={r['camera']:r for r in mask['rows']if r['split'].lower()=='train'}
    names=data['camera_names'].astype(str)
    if len(train)!=63 or set(names)!=set(train)or(cams<0).any()or(cams>=len(names)).any():raise ValueError('TRAIN63 camera identity mismatch')
    # Recheck the actual hard-interior source at each selected observed UV.
    for ci,name in enumerate(names):
        idx=np.where(cams==ci)[0]
        if not len(idx):continue
        row=train[name]
        if sha(row['archive'])!=row['archive_sha256']:raise ValueError('Selected semantic source changed')
        with np.load(row['archive'],allow_pickle=False)as z:inside=(z['hard']>=.99999)&~z['source_ineligible']
        interior=distance_transform_edt(inside)>=2
        xy=np.floor(data['uv'][idx]).astype(int)
        if (xy[:,0]<0).any()or(xy[:,0]>=inside.shape[1]).any()or(xy[:,1]<0).any()or(xy[:,1]>=inside.shape[0]).any()or not interior[xy[:,1],xy[:,0]].all():raise ValueError('Witness no longer lies in reviewed actor semantic interior')
    half=np.maximum(.005,.01*depth)
    parallel=np.abs(d)<1e-12;safe=np.where(parallel,1,d);a=(b[0]-o)/safe;c=(b[1]-o)/safe
    enter=np.where(parallel,-np.inf,np.minimum(a,c)).max(1);leave=np.where(parallel,np.inf,np.maximum(a,c)).min(1)
    band_start=np.maximum.reduce([depth-half,enter,np.full(n,.052)]);band_end=np.minimum(depth+half,leave)
    if (band_end<=band_start).any()or(depth<band_start-1e-7).any()or(depth>band_end+1e-7).any():raise ValueError('Witness band excluded by current actor bounds')
    for t in [start,end,band_start,band_end]:
        xyz=o+d*t[:,None]
        if(xyz<b[0]-1e-6).any()or(xyz>b[1]+1e-6).any():raise ValueError('Actor free/band endpoint outside bounds')
    data['band_start']=band_start;data['band_end']=band_end
    receipt=dict(manifest_sha256=sha(path),archive_sha256=sha(archive),mask_manifest_sha256=sha(mask_manifest),
        variant=variant,uses_masks=True,TRAIN_observations=n,unique_points=len(votes),minimum_actual_distinct_TRAIN_votes=int(votes.min()),
        actual_semantic_interiors_rechecked=True,geometry_confidence_independent_of_RGB_policy=True,
        no_fullscene_floor_or_MVS=True,no_alpha_GT=True)
    return data,receipt

class ActorObservedGeometry:
    def __init__(self,manifest,variant,bounds,mask_manifest,transforms,preparation,*,device='cpu',free_weight=.01,band_weight=.01,rays=128,free_samples=64,band_samples=16,seed=42):
        data,self.receipt=load_actor_geometry(manifest,variant,bounds,mask_manifest,transforms,preparation)
        self.rays,self.free_samples,self.band_samples,self.seed=rays,free_samples,band_samples,seed
        self.free_weight,self.band_weight=float(free_weight),float(band_weight)
        if not all(math.isfinite(v)and v>=0 for v in [self.free_weight,self.band_weight]):raise ValueError('Finite nonnegative actor geometry weights required')
        self.origins=torch.tensor(data['origins'],dtype=torch.float32,device=device);self.directions=torch.tensor(data['directions'],dtype=torch.float32,device=device)
        self.starts=torch.tensor(data['t_start'],dtype=torch.float32,device=device);self.ends=torch.tensor(data['t_end'],dtype=torch.float32,device=device)
        self.band_start=torch.tensor(data['band_start'],dtype=torch.float32,device=device);self.band_end=torch.tensor(data['band_end'],dtype=torch.float32,device=device)
        cams=data['camera_idx'];order=np.argsort(cams,kind='stable');_,counts=np.unique(cams,return_counts=True)
        self.order=torch.tensor(order,device=device);self.offsets=torch.tensor(np.r_[0,np.cumsum(counts)],device=device)

    def sample(self,step,band=False):
        gen=torch.Generator(device=self.origins.device);gen.manual_seed((self.seed+int(step)*1000003)%(2**63-1))
        ncam=len(self.offsets)-1;base,extra=divmod(self.rays,ncam);order=torch.randperm(ncam,generator=gen,device=self.origins.device)
        camera=torch.cat([order.repeat(base),order[:extra]]);count=self.offsets[1:]-self.offsets[:-1]
        local=(torch.rand(self.rays,generator=gen,device=self.origins.device)*count[camera]).long()
        ids=self.order[self.offsets[camera]+local]
        if band:gen.manual_seed((self.seed+int(step)*1000003+0x53555246)%(2**63-1))
        samples=self.band_samples if band else self.free_samples
        unit=(torch.arange(samples,device=self.origins.device)[None]+torch.rand((self.rays,samples),generator=gen,device=self.origins.device))/samples
        begin,end=(self.band_start,self.band_end)if band else(self.starts,self.ends);length=end[ids]-begin[ids]
        t=begin[ids,None]+length[:,None]*unit;xyz=self.origins[ids,None]+self.directions[ids,None]*t[...,None]
        return xyz,length,ids

    def __call__(self,field,step):
        losses={};metrics={}
        for band in [False,True]:
            xyz,length,ids=self.sample(step,band);points=xyz.reshape(-1,3)
            if not bool(((points>field.aabb[0])&(points<field.aabb[1])).all()):raise ValueError('Actor auxiliary samples must be strictly inside field')
            native=field.density_fn(points);raw=field._density_before_activation
            if raw.numel()!=native.numel()or not torch.isfinite(raw).all():raise FloatingPointError('Nonfinite/unmatched cached native density logits')
            scale=float(field.average_init_density)
            with torch.autocast(device_type=points.device.type,enabled=False):
                if band:
                    log_tau=torch.logsumexp(raw.reshape(self.rays,self.band_samples).float()+math.log(scale),1)-math.log(self.band_samples)+length.float().log()
                    losses['actor_sfm_surface_band']=self.band_weight*stable_opacity_nll(log_tau).mean()
                    metrics['actor_surface_band_opacity_mean']=(-torch.expm1(-log_tau.detach().clamp(max=15).exp())).mean()
                else:
                    sigma=exact_forward_true_exp(native,raw,scale).reshape(self.rays,self.free_samples).float()
                    tau=sigma.mean(1)*length.float();losses['actor_sfm_true_tau']=self.free_weight*tau.mean()
                    metrics['actor_sfm_free_tau_mean']=tau.detach().mean()
        return losses,metrics
