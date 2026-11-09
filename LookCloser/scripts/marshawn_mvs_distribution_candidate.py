"""TRAIN-only photometric-MVS expected robust first-termination error objective.

No RGB targets, hard carving, occupancy edits or NN parameters. Diagnostic
robustness scale is not a calibrated MVS confidence interval.
"""
from pathlib import Path
import json, hashlib, math
import numpy as np
import torch
from marshawn_packed_distortion_candidate import PackedDistortionCapture

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def conditional_expected_cost(weights, sample_error, ray_indices, opacity):
    """Jensen distribution cost; errors in normal-height world units."""
    sample_cost=torch.sqrt(sample_error.square()+.005**2)-.005
    return torch.zeros_like(opacity).index_add(0,ray_indices,weights*sample_cost)/opacity

class MVSDistributionDepth:
    def __init__(self, manifest, expected_sha, pipeline, weight=.01):
        path=Path(manifest)
        if sha(path)!=expected_sha: raise ValueError('MVS manifest changed')
        m=json.loads(path.read_text())
        for p,h in m['bindings'].items():
            if sha(p)!=h: raise ValueError('MVS source changed: '+p)
        if m['uses_eval_RGB'] or m['bands_are_calibrated_CI']: raise ValueError('Wrong MVS provenance')
        data=Path(pipeline.config.datamanager.dataparser.data)
        if sha(data/'transforms.json')!=m['transforms_sha256']: raise ValueError('Actual calibrated source mismatch')
        if not math.isfinite(weight) or weight<=0: raise ValueError('Positive finite weight required')
        z=np.load(m['archive']);n=np.asarray(z['normal'],float)
        if sha(m['archive'])!=m['archive_sha256'] or not np.isclose(np.linalg.norm(n),1,atol=1e-9): raise ValueError('Archive/normal mismatch')
        train_names=[Path(p).stem.rsplit('_',1)[-1] for p in pipeline.datamanager.train_dataset.image_filenames]
        self.camera_names=z['cameras'].tolist()
        if self.camera_names!=['MA','NA','NB','LB'] or not set(self.camera_names)<=set(train_names): raise ValueError('Exactly four actual TRAIN source cameras required')
        origins,directions,depth=[np.asarray(z[k]) for k in ['origins','directions','depth']]
        if origins.shape!=(4,52,3) or directions.shape!=origins.shape or depth.shape!=(4,52): raise ValueError('Exactly52candidates×4TRAINrays required')
        if not np.allclose(origins+directions*depth[...,None],z['xyz'][None],atol=3e-7,rtol=0): raise ValueError('Rays must hit actual candidate XYZ')
        if not np.allclose(np.linalg.norm(directions,axis=-1),1,atol=1e-7): raise ValueError('Unit world-distance rays required')
        tf=json.loads((data/'transforms.json').read_text());by={q['physical_camera']:q for q in tf['frames']}
        for c,name in enumerate(self.camera_names):
            if not np.allclose(origins[c],np.array(by[name]['transform_matrix'])[:3,3],atol=1e-7): raise ValueError('World camera origins changed')
        self.device=pipeline.device;self.weight=float(weight)
        for name,value in [('origins',origins),('directions',directions),('depth',depth),('normal',n)]:setattr(self,name,torch.tensor(value,device=self.device,dtype=torch.float32))
        self.camera_indices=torch.tensor([train_names.index(c)for c in self.camera_names],device=self.device,dtype=torch.long)
        self.receipt=dict(kind='photometric_MVS_expected_robust_firsttermination_error',rays=128,candidates=52,clear_train_cameras=self.camera_names,weight=self.weight,robust_normal_height_scale=.005,scale_is_regularizer_not_calibrated_CI=True,RGB_targets=False,eval_RGB=False,hard_carving=False,occupancy_changes=False,head_gradients=False,manifest_sha256=expected_sha,private_ray_selection_and_march_RNG=True)
    def sample(self,step):
        g=torch.Generator(device=self.device);seed=int(np.random.SeedSequence([42,int(step),0x4D565344]).generate_state(1,dtype=np.uint64)[0])%(2**63-1);g.manual_seed(seed)
        ids=torch.randint(52,(4,32),device=self.device,generator=g);cams=torch.arange(4,device=self.device)[:,None].expand(4,32)
        return cams.flatten(),ids.flatten(),seed
    def __call__(self,model,step):
        from nerfstudio.cameras.rays import RayBundle
        cameras,ids,seed=self.sample(step);o=self.origins[cameras,ids];d=self.directions[cameras,ids];truth=self.depth[cameras,ids]
        ray=RayBundle(origins=o,directions=d,pixel_area=torch.ones((128,1),device=self.device),camera_indices=self.camera_indices[cameras,None])
        # The native train sampler may stratify. Isolate this extra forward from
        # the main sampler/RGB random-background stream and from all other losses.
        device_index=torch.device(self.device).index
        if device_index is None:device_index=torch.cuda.current_device()
        with torch.random.fork_rng(devices=[device_index]):
            torch.random.default_generator.manual_seed(seed);torch.cuda.manual_seed(seed)
            with PackedDistortionCapture(model) as capture: model(ray)
        p=capture.packet
        if capture.calls!=1 or p is None:raise ValueError('Expected one extra native graph-weight packet')
        w=p['weights'].reshape(-1).float();ri=p['ray_indices'].long();mid=(p['starts'].reshape(-1).float()+p['ends'].reshape(-1).float())*.5
        if not w.requires_grad or not bool(torch.isfinite(w).all()):raise ValueError('Finite graph-connected native weights required')
        with torch.autocast(device_type='cuda',enabled=False):
            opacity=torch.zeros(128,device=self.device).index_add(0,ri,w);numerator=torch.zeros_like(opacity).index_add(0,ri,w*mid)
            if not bool((opacity.detach()>1e-4).all()):raise ValueError('Conditional depth undefined for empty/transparent source rays; do not silently mask')
            predicted=numerator/opacity;cosine=(d*self.normal).sum(-1);error=cosine*(predicted-truth);sample_error=cosine[ri]*(mid-truth[ri]);per_ray_cost=conditional_expected_cost(w,sample_error,ri,opacity);unit=per_ray_cost.mean();loss=self.weight*unit
        self.last_packet=dict(candidate_indices=ids.detach(),camera_indices=cameras.detach(),origins=o.detach(),directions=d.detach(),truth=truth.detach(),predicted=predicted.detach(),opacity=opacity.detach(),normal_height_error=error.detach(),distribution_expected_cost=per_ray_cost.detach(),starts=p['starts'].detach(),ends=p['ends'].detach(),weights=w.detach(),ray_indices=ri.detach())
        return loss,dict(MVS_depth_unit=unit.detach(),MVS_depth_normal_abs_mean=error.detach().abs().mean(),MVS_depth_opacity_min=opacity.detach().min(),MVS_depth_samples=torch.tensor(len(w),device=self.device))

def __getattr__(name):
    if name!='NativeMVSDistributionPipeline':raise AttributeError(name)
    from marshawn_floor_band_scene_prior import NativeFloorBandPriorPipeline
    class NativeMVSDistributionPipeline(NativeFloorBandPriorPipeline):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            from marshawn_true_los import TrueTauFreeSpaceObjective
            # New object owns the SAME validated observations/private sampling.
            # Replacing only the auxiliary preserves original RGB/NN forward.
            object.__setattr__(self,'_final_free',TrueTauFreeSpaceObjective.from_existing(self._final_free))
            objective=MVSDistributionDepth(self.config.MVS_depth_manifest,self.config.MVS_depth_manifest_sha256,self,weight=self.config.MVS_depth_weight)
            object.__setattr__(self,'_MVS_depth',objective)
        def get_train_loss_dict(self,step):
            outputs,losses,metrics=super().get_train_loss_dict(step)
            # Main distortion capture has EXITED before the independent extra rays.
            loss,diagnostic=self._MVS_depth(self.model,int(step));losses['MVS_distribution_depth']=loss;metrics.update(diagnostic)
            return outputs,losses,metrics
    NativeMVSDistributionPipeline.__name__=name;NativeMVSDistributionPipeline.__qualname__=name;NativeMVSDistributionPipeline.__module__=__name__;globals()[name]=NativeMVSDistributionPipeline;return NativeMVSDistributionPipeline
