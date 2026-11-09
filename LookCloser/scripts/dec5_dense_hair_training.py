"""Opt-in confidence-qualified dense hair surface guidance for LookCloser."""
from pathlib import Path
import json
import numpy as np
import torch
from dec5_hair import unproject,packed_depth_quantile
from nerfstudio.utils.lookcloser_rng import fork_seeded_rng
from run_dec5_hair_campaign import atomic_json,sha


def median_optical_thickness_term(before_band,half_width,band_sigma):
    """Opaque first-hit median has optical thickness ln(2) at its anchor.

    Band samples are even midpoint quadrature; the front half ends exactly at
    the anchor. This is restricted to qualified opaque foreground observations.
    """
    if band_sigma.shape[-1]%2:raise ValueError('Median guidance needs even midpoint quadrature')
    tau=before_band+band_sigma[...,:band_sigma.shape[-1]//2].clamp_min(0).mean(-1)*half_width
    return torch.nn.functional.smooth_l1_loss(tau/np.log(2),torch.ones_like(tau),reduction='none',beta=.5)


def surface_guidance_terms(sigma,starts,ends,indices,target,half_width,band_sigma):
    overlap=(torch.minimum(ends.reshape(-1),target[indices]-half_width[indices])-starts.reshape(-1)).clamp_min(0)
    empty=sigma.reshape(-1).clamp_min(0).mul(overlap)
    empty=target.new_zeros(len(target)).scatter_add(0,indices.long(),empty)
    thickness=band_sigma.clamp_min(0).mean(-1)*(2*half_width)
    surface=-torch.log(-torch.expm1(-thickness)+1e-12)
    return empty,surface


class DenseHairGuidance:
    def __init__(self,trainer,path,output,weight=.01,rays_per_step=256):
        self.trainer=trainer;self.pipe=trainer.pipeline;self.model=self.pipe.model;self.weight=weight;self.rays_per_step=rays_per_step;self.active=False;self.captured=None
        self.model.current_train_step=max(int(trainer._start_step)-1,0)
        if trainer._start_step<int(self.model.config.depth_loss_steps):raise ValueError('Dense late guidance must follow the original early sparse phase')
        receipt=json.loads(path.with_suffix('.json').read_text());arrays=np.load(path)
        data=self.pipe.datamanager.config.data
        if receipt['dataset_sha256']!=sha(data/'transforms.json') or receipt['npz_sha256']!=sha(path) or receipt.get('uses_validation_rgb') is not False or receipt.get('uses_test_rgb') is not False:raise ValueError('Dense anchors must be verified train-only inputs')
        if receipt['anchors']<10000 or receipt['cameras_with_anchors']<30:raise ValueError('Too little dense hair support')
        dataset=self.pipe.datamanager.train_dataset;payload=json.loads((data/'transforms.json').read_text());by={Path(f['file_path']).name:f for f in payload['frames']}
        frames=[by[Path(p).name] for p in dataset.image_filenames];lookup={f['physical_camera']:i for i,f in enumerate(frames)}
        remap=np.asarray([lookup[n] for n in arrays['camera_names']]);device=self.pipe.device
        self.cameras=dataset.cameras.to(device)
        normalized=self.cameras.camera_to_worlds.cpu().numpy();raw=np.asarray([f['transform_matrix'] for f in frames])
        rotation=normalized[0,:3,:3]@raw[0,:3,:3].T
        self.scale=float(np.linalg.norm(normalized[1,:3,3]-normalized[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]))
        translation=normalized[0,:3,3]-self.scale*rotation@raw[0,:3,3]
        self.camera=torch.as_tensor(remap[arrays['camera']],device=device,dtype=torch.long)
        self.uv=torch.as_tensor(arrays['uv'],device=device);self.z=torch.as_tensor(arrays['camera_z'],device=device)
        self.confidence=torch.as_tensor((arrays['support'].astype(np.float32)-2).clip(1,6)/6,device=device)
        validation=arrays['validation'].astype(bool);self.fit=torch.as_tensor(np.flatnonzero(~validation),device=device)
        self.validation=torch.as_tensor(np.flatnonzero(validation),device=device)
        # Validate the complete camera-z -> ray-range / scene-similarity chain.
        selected=np.linspace(0,len(self.z)-1,min(256,len(self.z)),dtype=int)
        rays,target,_=self.rays(torch.as_tensor(selected,device=device))
        xyz=np.stack([unproject(arrays['uv'][k:k+1],arrays['camera_z'][k:k+1],frames[int(self.camera[k])])[0] for k in selected])
        expected=self.scale*xyz@rotation.T+translation
        predicted=(rays.origins+rays.directions*target[:,None]).detach().cpu().numpy()
        error=float(np.linalg.norm(expected-predicted,axis=-1).max())
        if error>2e-5:raise ValueError('Dense anchor camera-z/ray-range transform failed')
        density_only=getattr(trainer,'dec5_density_only',False)
        self.center_median=getattr(trainer,'dec5_center_median',False)
        if density_only:
            field=self.model.field
            if field.density_geometry_encoding is None or field.density_geometry_mlp is None:
                raise ValueError('Density-only refinement requires an explicitly cloned density branch')
            field.requires_grad_(False)
            field.density_geometry_encoding.requires_grad_(True)
            field.density_geometry_mlp.requires_grad_(True)
        elif not getattr(trainer,'dec5_joint_head',False):
            for p in self.model.field.mlp_color.parameters():p.requires_grad_(False)
        def capture(module,args,result):
            if self.active:
                from nerfstudio.field_components.field_heads import FieldHeadNames
                rs=args[0];self.captured=(result[FieldHeadNames.DENSITY],rs.frustums.starts,rs.frustums.ends)
        self.model.field.register_forward_hook(capture)
        atomic_json(output/'dense_hair_guidance.json',dict(anchors_receipt_sha256=sha(path.with_suffix('.json')),anchors=len(self.z),fit_anchors=len(self.fit),validation_anchors=len(self.validation),
            weight=weight,rays_per_step=rays_per_step,surface_samples=16,half_band='two projected pixel footprints along the ray',
            objective=('optical thickness before band + band opacity + opaque first-hit median optical-thickness constraint' if self.center_median else
                       'optical thickness before anchor band + negative log opacity within anchor band'),
            assumption='qualified opaque visible hair surface; no constraints on transparent hair boundary',
            frozen=('all original appearance encoding/geometry/color parameters and frequency grid' if density_only else
                    ('frequency grid only; joint scoped density and RGB continuation' if getattr(trainer,'dec5_joint_head',False) else
                     'appearance decoder weights only; shared hash/geometry features still learn')),
            rng='sparse_depth stream, after the mutually exclusive early sparse phase',
            camera_z_ray_range_roundtrip_max_error=error,normalized_per_calibration_unit=self.scale,
            uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__)))

    def rays(self,selected):
        index=self.camera[selected];uv=self.uv[selected]
        rays=self.cameras.generate_rays(index[:,None],coords=uv[:,[1,0]],keep_shape=False)
        forward=-self.cameras.camera_to_worlds[index,:3,2]
        cosine=(rays.directions*forward).sum(-1)
        target=self.z[selected]*self.scale/cosine
        focal=torch.minimum(self.cameras.fx[index].reshape(-1),self.cameras.fy[index].reshape(-1))
        return rays,target,(2*target/focal).clamp_min(1e-6)

    def install(self):
        original=self.pipe.get_train_loss_dict
        def loss(step):
            outputs,losses,metrics=original(step)
            with fork_seeded_rng(self.pipe.config.training_seed,'sparse_depth',step,self.pipe.device):
                selected=self.fit[torch.randint(len(self.fit),(self.rays_per_step,),device=self.pipe.device)]
            rays,target,half_width=self.rays(selected);rays=self.model.collider(rays)
            valid=(target-half_width>rays.nears.reshape(-1))&(target+half_width<rays.fars.reshape(-1))
            if not bool(valid.any()):return outputs,losses,metrics
            rays=rays[valid];target=target[valid];half_width=half_width[valid];confidence=self.confidence[selected][valid]
            self.active=True;self.captured=None
            try:result=self.model(rays)
            finally:self.active=False
            indices=result['packed_ray_indices'].long()
            if len(indices):sigma,starts,ends=self.captured
            else:sigma=starts=ends=target.new_empty(0)
            sample=(torch.arange(16,device=self.pipe.device,dtype=target.dtype)+.5)/16
            distances=target[:,None]-half_width[:,None]+2*half_width[:,None]*sample
            positions=rays.origins[:,None]+rays.directions[:,None]*distances[...,None]
            band_sigma=self.model.field.density_fn(positions.reshape(-1,3)).reshape(len(target),16)
            empty,surface=surface_guidance_terms(sigma,starts,ends,indices,target,half_width,band_sigma)
            objective=empty+surface
            if self.center_median:objective=objective+median_optical_thickness_term(empty,half_width,band_sigma)
            losses['dense_hair_surface_loss']=self.weight*(objective*confidence).mean()
            extra=len(indices)+positions.numel()//3;self.pipe.cumulative_point_samples.add_(extra)
            metrics['cumulative_point_samples']=self.pipe.cumulative_point_samples.clone()
            metrics['dense_anchor_rays']=torch.tensor(len(target),device=self.pipe.device)
            metrics['dense_expected_relative']=(result['depth'].reshape(-1).detach()-target).abs().div(target).mean()
            self.captured=None
            return outputs,losses,metrics
        self.pipe.get_train_loss_dict=loss

    def evaluate(self):
        self.pipe.eval()
        try:
            relative_rows=[];valid_rows=[]
            with torch.no_grad():
                for selected in self.validation.split(512):
                    rays,target,_=self.rays(selected);result=self.model(rays)
                    values=[result[k].float().cpu().numpy() for k in ['packed_spacing_starts','packed_spacing_ends','packed_weights']]
                    depth=packed_depth_quantile(*values,result['packed_ray_indices'].cpu().numpy(),rays.nears.cpu().numpy(),rays.fars.cpu().numpy())
                    target=target.cpu().numpy();valid=(depth>0)&np.isfinite(depth)
                    relative_rows.append(np.where(valid,np.abs(depth-target)/target,1.));valid_rows.append(valid)
                relative=np.concatenate(relative_rows);valid=np.concatenate(valid_rows)
                return dict(eval_dense_anchor_median_relative=float(np.median(relative)),eval_dense_anchor_p90_relative=float(np.quantile(relative,.9)),eval_dense_anchor_valid_fraction=float(valid.mean()))
        finally:self.pipe.train()
