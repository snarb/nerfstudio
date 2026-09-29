"""Opt-in LookCloser mesh-teacher pipeline; stock methods retain their behavior."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from copy import copy
import json
from pathlib import Path
from typing import Type, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig
from nerfstudio.model_components.lookcloser_grid import FrequencyGridManager
from nerfstudio.model_components.mesh_distillation import (SeparateBackground, camera_z_to_distance,
    compose_actor_background, projected_frequency, tensor_digest, weighted_charbonnier, weighted_tail_mean, matte_tail_objective, ray_plane_slab)
from nerfstudio.models.lookcloser import LookCloserModel, LookCloserModelConfig
from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline, LookCloserPipelineConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig


@dataclass
class DistillationParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda: DistillationParser)


class DistillationParser(Nerfstudio):
    def _generate_dataparser_outputs(self, split='train'):
        cfg=self.config
        if cfg.auto_scale_poses or cfg.orientation_method!='none' or cfg.center_method!='none' or cfg.scale_factor!=1:
            raise ValueError('Distillation requires the immutable camera coordinate gauge')
        if cfg.depth_unit_scale_factor!=1 or cfg.downscale_factor!=1 or cfg.load_3D_points:
            raise ValueError('Distillation requires depth units=1, native images, and no implicit point initialization')
        out=super()._generate_dataparser_outputs(split)
        root=cfg.data if cfg.data.is_dir() else cfg.data.parent
        meta=json.loads((root/'transforms.json').read_text())
        if 'distillation' not in meta:raise ValueError('Missing distillation manifest')
        out.scene_box.aabb=torch.tensor(meta['distillation']['actor_bounds'],dtype=torch.float32)
        by_path={str(root/r['file_path']):r for r in meta['frames']}
        out.metadata.update(distillation=meta['distillation'],distillation_root=str(root),
                            distillation_rows=[by_path[str(p)] for p in out.image_filenames],distillation_split=split)
        return out


class DistillationDataset(InputDataset):
    exclude_batch_keys_from_device=InputDataset.exclude_batch_keys_from_device+['confidence','depth_image','evaluation_mask','empty_mask','detail_mask','foreground_target','alpha_target','alpha_valid']

    @property
    def image_filenames(self):
        return self._dataparser_outputs.image_filenames

    def get_metadata(self,data):
        i=int(data['image_idx']);row=self.metadata['distillation_rows'][i]
        root=Path(self.metadata['distillation_root']);h,w=data['image'].shape[:2]
        def load_mask(name,default):
            if name not in row:return torch.full((h,w,1),default,dtype=torch.float32)
            a=np.asarray(Image.open(root/row[name])).copy()
            if a.shape!=(h,w):raise ValueError(f'{name} dimensions do not match native RGB')
            return torch.from_numpy(a).float()[...,None]/255
        depth=torch.zeros((h,w,1))
        if 'depth_file_path' in row:
            depth=get_depth_image_from_path(root/row['depth_file_path'],h,w,1.)
        foreground=data['image'];alpha=load_mask('alpha_file_path',1.)
        has_matte='foreground_file_path' in row and 'alpha_file_path' in row
        if ('foreground_file_path' in row)!=('alpha_file_path' in row):raise ValueError('Matte needs both foreground color and alpha')
        if has_matte:
            if self.metadata['distillation_split']!='train':raise ValueError('Matte targets are training-only; evaluation uses original RGB')
            foreground=torch.from_numpy(np.array(Image.open(root/row['foreground_file_path']).convert('RGB'))).float()/255
            if foreground.shape!=data['image'].shape:raise ValueError('Matte foreground dimensions mismatch')
        return dict(confidence=load_mask('confidence_file_path',1.),depth_image=depth,
                    evaluation_mask=load_mask('evaluation_mask_path',1.),empty_mask=load_mask('empty_mask_path',0.),
                    detail_mask=load_mask('detail_mask_path',0.),
                    foreground_target=foreground*alpha,alpha_target=alpha,
                    alpha_valid=torch.full((h,w,1),float(has_matte)))


@dataclass
class MaskedFrequencySamplerConfig(PixelSamplerConfig):
    _target: Type = field(default_factory=lambda: MaskedFrequencySampler)
    frequency_map_dir: str='lookcloser_frequencies'
    enable_fas: bool=True
    fas_warmup_steps: int=1000
    fas_ramp_steps: int=3000
    empty_fraction: float=0.0
    detail_fraction: float=0.0
    boundary_fraction: float=0.0
    boundary_radius: int=3


class MaskedFrequencySampler(PixelSampler):
    """Exact valid-pixel buckets: FAS cannot draw an unknown pixel.

    All-images caching is intentional. Global flattened indices are int32 for
    this 300-view pilot; larger datasets fail explicitly rather than overflow.
    """
    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        self.dataset=kwargs['dataset'];self.sample_count=0;self.buckets=None

    def initialize(self,batch):
        shape=batch['image'].shape[:3];n,h,w=shape
        if n!=len(self.dataset) or n*h*w>=2**31:raise ValueError('Sampler requires all-images cache below int32 index limit')
        self.cache_order=batch['image_idx'].detach().cpu().long().clone()
        if not torch.equal(self.cache_order.sort().values,torch.arange(n)):
            raise ValueError('Cache must contain every training image exactly once')
        parts=defaultdict(list);root=Path(self.dataset.metadata['distillation_root'])
        for i in range(n):
            valid=batch['mask'][i,...,0].cpu().numpy().astype(bool)
            valid &= batch['confidence'][i,...,0].cpu().numpy()>0
            path=root/self.config.frequency_map_dir/(self.dataset.image_filenames[int(self.cache_order[i])].stem+'.pt')
            if self.config.enable_fas and self.dataset.metadata['distillation_split']=='train':
                scalar=torch.load(path,map_location='cpu',weights_only=True).numpy()
                levels=np.rint(np.log(scalar/16)/np.log(8192/16)*15).clip(0,15).astype('uint8')
                levels=np.repeat(np.repeat(levels,8,0),8,1)[:h,:w]
            else:levels=np.zeros((h,w),'uint8')
            for level in np.unique(levels[valid]):
                idx=np.flatnonzero(valid & (levels==level)).astype('int32')+i*h*w
                parts[int(level)].append(torch.from_numpy(idx))
        self.buckets={k:torch.cat(v).cuda() for k,v in parts.items() if v}
        if not self.buckets:raise ValueError('No valid supervision pixels')
        self.levels=sorted(self.buckets);self.shape=shape
        empty=batch.get('empty_mask')
        self.empty_indices=None
        if self.config.empty_fraction>0 and empty is not None:
            flat=torch.nonzero(empty.reshape(-1)>0).flatten().to(torch.int32)
            if len(flat):self.empty_indices=flat.cuda()
        fractions=(self.config.empty_fraction,self.config.detail_fraction,self.config.boundary_fraction)
        if any(not 0<=v<=1 for v in fractions) or sum(fractions)>1:
            raise ValueError('Detail, boundary and empty sampling fractions must fit in the batch')
        self.detail_indices=None
        if self.config.detail_fraction>0 and 'detail_mask' in batch:
            detail=(batch['detail_mask'].cpu()>0)&(batch['mask'].cpu()>0)&(batch['confidence'].cpu()>0)
            flat=torch.nonzero(detail.reshape(-1)).flatten().to(torch.int32)
            if len(flat):self.detail_indices=flat.cuda()
        self.boundary_indices=None
        if self.config.boundary_fraction>0:
            if 'alpha_target' not in batch or 'alpha_valid' not in batch:
                raise ValueError('Boundary sampling requires matte targets')
            radius=self.config.boundary_radius
            if radius<1:raise ValueError('Boundary radius must be positive')
            boundary=[]
            for i in range(n):
                alpha=batch['alpha_target'][i].permute(2,0,1).float()
                hi=F.max_pool2d(alpha,2*radius+1,1,radius)
                lo=-F.max_pool2d(-alpha,2*radius+1,1,radius)
                edge=(hi-lo)[0]>.02
                for key in ('mask','confidence','alpha_valid'):
                    edge &= batch[key][i,...,0].to(edge.device)>0
                flat=torch.nonzero(edge.reshape(-1)).flatten().to(torch.int32)+i*h*w
                if len(flat):boundary.append(flat.cuda())
            if boundary:self.boundary_indices=torch.cat(boundary)

    def sample(self,batch,**kwargs):
        if self.buckets is None or not torch.equal(batch['image_idx'].cpu(),self.cache_order):
            self.initialize(batch)
        n,h,w=self.shape;size=self.num_rays_per_batch
        empty_count=int(size*self.config.empty_fraction) if self.empty_indices is not None else 0
        detail_count=int(size*self.config.detail_fraction) if self.detail_indices is not None else 0
        boundary_count=int(size*self.config.boundary_fraction) if self.boundary_indices is not None else 0
        strength=min(max((self.sample_count-self.config.fas_warmup_steps)/max(self.config.fas_ramp_steps,1),0),1)
        uniform=torch.tensor([len(self.buckets[k]) for k in self.levels],device='cuda',dtype=torch.float32)
        uniform/=uniform.sum()
        fas=torch.tensor([1+2*k/15 for k in self.levels],device='cuda');fas/=fas.sum()
        probs=uniform*(1-strength)+fas*strength if self.config.enable_fas else uniform
        remainder=size-empty_count-detail_count-boundary_count
        chosen=(torch.multinomial(probs,remainder,replacement=True) if remainder else torch.empty(0,device='cuda',dtype=torch.long));flat=[]
        for j,k in enumerate(self.levels):
            count=int((chosen==j).sum())
            if count:flat.append(self.buckets[k][torch.randint(len(self.buckets[k]),(count,),device='cuda')])
        if empty_count:
            flat.append(self.empty_indices[torch.randint(len(self.empty_indices),(empty_count,),device='cuda')])
        if detail_count:
            flat.append(self.detail_indices[torch.randint(len(self.detail_indices),(detail_count,),device='cuda')])
        if boundary_count:
            flat.append(self.boundary_indices[torch.randint(len(self.boundary_indices),(boundary_count,),device='cuda')])
        flat=torch.cat(flat).long();flat=flat[torch.randperm(size,device='cuda')]
        indices=torch.stack([flat//(h*w),(flat//w)%h,flat%w],-1)
        cpu=indices.cpu();out={}
        for key,value in batch.items():
            if isinstance(value,torch.Tensor) and value.ndim>=3 and tuple(value.shape[:3])==(n,h,w):
                idx=indices if value.is_cuda else cpu
                out[key]=value[idx[:,0],idx[:,1],idx[:,2]]
        indices[:,0]=self.cache_order.to(indices.device)[indices[:,0]]
        # VanillaDataManager's ray generator keeps its pixel-coordinate lookup
        # on CPU; camera-ray generation subsequently moves the rays as needed.
        out['indices']=indices.cpu();self.sample_count+=1
        return out


class NativeDistillationDataManager(VanillaDataManager[DistillationDataset]):
    """Optional subpixel native-color supervision, independent of output views."""
    def next_train(self, step):
        rays, batch = super().next_train(step)
        stereo=getattr(self.config,'stereo_depth_receipt',None)
        if stereo is not None:
            if not hasattr(self,'_stereo_depth'):
                from nerfstudio.model_components.stereo_depth_evidence import StereoDepthTargets
                self._stereo_depth=StereoDepthTargets(stereo,self.train_dataset)
            batch=self._stereo_depth.apply(batch)
        observed = getattr(self.config, 'observed_background_path', None)
        if observed is not None:
            if not hasattr(self, '_observed_background'):
                from nerfstudio.model_components.observed_background import ObservedBackgroundTargets
                self._observed_background = ObservedBackgroundTargets(observed, self.train_dataset,
                    getattr(self.config,'observed_trimap_path',None))
            batch = self._observed_background.apply(batch)
        path = getattr(self.config, 'native_training_manifest', None)
        if path is None:
            return rays, batch
        if not hasattr(self, '_native_targets'):
            from nerfstudio.model_components.native_training import NativeTrainingTargets
            excluded=self._observed_background.opaque_override if observed is not None else None
            self._native_targets = NativeTrainingTargets(path, self.train_dataset, self.device,exclude=excluded)
        native_observed=getattr(self.config,'native_observed_background',False)
        if native_observed and observed is None:raise ValueError('Native composites require observed background')
        return self._native_targets.apply(self.train_ray_generator.cameras, batch,
            observed=self._observed_background if native_observed else None)


@dataclass
class NativeDistillationDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: NativeDistillationDataManager)
    native_training_manifest: Optional[Path] = None
    observed_background_path: Optional[Path] = None
    observed_trimap_path: Optional[Path] = None
    native_observed_background: bool = False
    stereo_depth_receipt: Optional[Path] = None


class ObservedFrequencyGrid(FrequencyGridManager):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer('observed',torch.zeros_like(self.grid,dtype=torch.bool))
        self.register_buffer('initialized',torch.tensor(False))
        self.frozen=False

    def query(self,positions):
        index=self.grid_to_indices(positions);idx=tuple(index.T)
        return torch.where(self.observed[idx],self.grid[idx],float(self.num_levels-1))[:,None]

    def update_max(self,positions,new_levels):
        if self.frozen:raise RuntimeError('Attempted mutation of frozen Frequency Grid')
        valid=((positions>=self.aabb_min_buf)&(positions<=self.aabb_max_buf)).all(-1)&torch.isfinite(new_levels.reshape(-1))
        positions,new_levels=positions[valid],new_levels.reshape(-1)[valid]
        super().update_max(positions,new_levels)
        if len(positions):self.observed[tuple(self.grid_to_indices(positions).T)]=True


@dataclass
class DistillationModelConfig(LookCloserModelConfig):
    _target: Type=field(default_factory=lambda: DistillationModel)
    background_checkpoint: Optional[Path]=None
    equipment_slab: Optional[Tuple[float,float]]=None
    density_slab_plane: Optional[Tuple[float,float,float,float]]=None
    freeze_frequency_grid: bool=False
    correct_sh_directions: bool=True
    view_independent_color: bool=False
    normalized_exponential_density: bool=False
    depth_distribution_weight: float=0.0
    depth_distribution_sigma: float=.0015
    stereo_depth_weight: float=0.0
    foreground_support_path: Optional[Path]=None
    empty_opacity_weight: float=0.0
    matte_opacity_weight: float=0.0
    matte_boundary_only: bool=True
    matte_soft_target_weight: float=1.0
    opacity_hard_fraction: float=1.0
    optimize_training_cameras: bool=False
    freeze_decoders: bool=False


class DistillationModel(LookCloserModel):
    def populate_modules(self):
        super().populate_modules()
        self.field.correct_sh_directions=self.config.correct_sh_directions
        self.field.view_independent_color=self.config.view_independent_color
        self.field.normalized_exponential_density=self.config.normalized_exponential_density
        if self.config.freeze_decoders:
            self.field.mlp_geo.requires_grad_(False);self.field.mlp_color.requires_grad_(False)
        if self.config.optimize_training_cameras:
            self.training_camera_optimizer=CameraOptimizerConfig(mode='SO3xR3').setup(num_cameras=self.num_train_data,device='cpu')
        if self.config.foreground_support_path:
            self.field.register_buffer('foreground_support',torch.load(self.config.foreground_support_path,map_location='cpu',weights_only=True))
        grid=ObservedFrequencyGrid(self.scene_box,self.config.grid_resolution,self.config.num_frequency_levels,
                                   self.config.min_res,float(self.config.max_res))
        self.freq_grid=grid;self.field.freq_grid=grid;self.adaptive_sampler.freq_grid=grid
        grid.frozen=self.config.freeze_frequency_grid
        if self.config.background_color!='black':raise ValueError('Separate background requires premultiplied black actor RGB')
        self.background=None
        if self.config.background_checkpoint:
            state=torch.load(self.config.background_checkpoint,map_location='cpu',weights_only=False)
            resolution=state['model']['texture'].shape[-1] if state['kind']=='plane' else None
            self.background=SeparateBackground(state['kind'],state['geometry'],texture_resolution=resolution)
            self.background.load_state_dict(state['model']);self.background.requires_grad_(False)
        if self.config.equipment_slab is not None:
            if self.config.density_slab_plane is not None:
                plane=torch.tensor(self.config.density_slab_plane,dtype=torch.float32)
                if plane.shape!=(4,) or not torch.isfinite(plane).all() or plane[:3].norm()<1e-8:
                    raise ValueError('Expected a finite nondegenerate world plane')
            elif self.background is not None and self.background.kind=='plane':
                plane=self.background.plane.detach().clone()
            else:raise ValueError('Density slab requires an explicit plane or frozen planar background')
            lo,hi=self.config.equipment_slab
            if not lo < hi or not np.isfinite([lo,hi]).all():
                raise ValueError('Density slab requires finite minimum < maximum signed distance')
            if self.background is not None and lo<=0:
                raise ValueError('A separate wall component requires a positive equipment gap')
            if self.config.ray_sampling_mode!='fixed':
                raise ValueError('Equipment slab currently requires fixed ray integration')
            plane/=plane[:3].norm()
            self.field.register_buffer('density_slab',torch.cat([plane,plane.new_tensor([lo,hi])]))

    def get_outputs(self,ray_bundle):
        if self.training and self.config.optimize_training_cameras:
            ray_bundle=copy(ray_bundle)
            self.training_camera_optimizer.apply_to_raybundle(ray_bundle)
            if self.collider is not None:self.collider.set_nears_and_fars(ray_bundle)
        if self.config.equipment_slab is not None:
            ray_bundle=copy(ray_bundle)
            near,far,hit=ray_plane_slab(ray_bundle.origins,ray_bundle.directions,self.field.density_slab)
            near=torch.maximum(near,ray_bundle.nears[...,0]);far=torch.minimum(far,ray_bundle.fars[...,0])
            hit &= far>near
            ray_bundle.nears=torch.where(hit,near,0.)[...,None]
            ray_bundle.fars=torch.where(hit,far,1e-6)[...,None]
        out=super().get_outputs(ray_bundle)
        if self.config.equipment_slab is not None:
            # Numerical placeholder intervals on missed rays are never content.
            for key in ('rgb','depth','accumulation','optical_thickness'):
                if key in out:out[key]=torch.where(hit[...,None],out[key],0.)
        if self.training and ((self.current_train_step<self.config.depth_loss_steps and self.config.depth_distribution_weight>0) or self.config.stereo_depth_weight>0):
            if self.config.ray_sampling_mode!='fixed':
                raise ValueError('Initial depth distribution probe currently requires fixed sampling')
            if 'sample_distances' not in out:
                samples=out['loss_ray_samples']
                t=(samples.spacing_starts+samples.spacing_ends)*.5
                out['sample_distances']=ray_bundle.nears[:,None]+(ray_bundle.fars-ray_bundle.nears)[:,None]*t
        out['actor_rgb']=out['rgb'];out['actor_depth']=out['depth']
        if self.background is not None:
            with torch.no_grad():bg=self.background(ray_bundle.origins,ray_bundle.directions)
            out['background_rgb']=bg['rgb'];out['background_depth']=bg['depth']
            out['rgb']=compose_actor_background(out['actor_rgb'],out['accumulation'],bg['rgb'])
        else:out['background_rgb']=torch.zeros_like(out['rgb'])
        out['directions_norm']=ray_bundle.metadata['directions_norm']
        return out

    def get_loss_dict(self,outputs,batch,metrics_dict=None):
        # Call the stock distortion path without inadvertently applying its
        # unscaled, unscheduled depth loss or its unweighted reconstruction.
        clean={k:v for k,v in batch.items() if k!='depth_image'}
        losses=super().get_loss_dict(outputs,clean,metrics_dict)
        observed=batch.get('observed_background_valid') if self.training else None
        if observed is not None:
            if self.config.matte_opacity_weight<=0 or self.config.optimize_training_cameras:
                raise ValueError('Observed compositing requires matte targets and fixed training cameras')
            observed=observed.to(self.device)
        weight=batch['confidence'].to(self.device)*batch['mask'].to(self.device)
        if observed is not None:weight=weight*(1-observed)
        losses['rgb_loss']=weighted_charbonnier(outputs['rgb'],batch['image'].to(self.device),weight)
        if self.training and self.config.stereo_depth_weight>0:
            if self.config.optimize_training_cameras:raise ValueError('Stereo evidence requires fixed calibrated cameras')
            from nerfstudio.model_components.stereo_depth_evidence import conditional_depth_interval
            target=camera_z_to_distance(batch['stereo_depth'].to(self.device),outputs['directions_norm'])
            sigma=camera_z_to_distance(batch['stereo_depth_sigma'].to(self.device),outputs['directions_norm'])
            losses['stereo_depth']=self.config.stereo_depth_weight*conditional_depth_interval(
                outputs['loss_weights'],outputs['sample_distances'],target,sigma,batch['stereo_depth_valid'].to(self.device))
        if self.training and self.config.optimize_training_cameras:
            adjustment=self.training_camera_optimizer.pose_adjustment
            losses['camera_prior']=.0001*(adjustment/.0002).square().mean()+.01*(adjustment.mean(0)/.0002).square().mean()
        if self.config.matte_opacity_weight>0 and self.training:
            valid=batch['alpha_valid'].to(self.device)*weight
            if not bool((valid>0).any()) and observed is None:raise ValueError('Matte supervision enabled without training targets')
            if 'optical_thickness' not in outputs:raise ValueError('Matte opacity requires an optical-thickness renderer')
            # The extracted foreground is premultiplied by alpha. Original RGB
            # includes wall color and must not supervise the isolated actor edge.
            losses['rgb_loss']=weighted_charbonnier(outputs['actor_rgb'],batch['foreground_target'].to(self.device),valid)
            tau=outputs['optical_thickness'].float().clamp_min(1e-6)
            alpha=batch['alpha_target'].to(self.device)
            cross_entropy=-alpha*torch.log(-torch.expm1(-tau))+(1-alpha)*tau
            opacity_valid=valid*(alpha<.999) if self.config.matte_boundary_only else valid
            soft_weight = getattr(self.config, 'matte_soft_target_weight', 1.0)
            if not 0 <= soft_weight <= 1:
                raise ValueError('Matte soft-target confidence must lie in [0,1]')
            # Extracted non-opaque alpha (including matting-only zeros) is an
            # uncertain estimate, not directly observed transmittance. Explicit
            # known-empty rays retain separate supervision. Opaque RGB stays real.
            opacity_valid = opacity_valid*torch.where(alpha >= .999, 1.0, soft_weight)
            losses['matte_opacity']=self.config.matte_opacity_weight*matte_tail_objective(
                cross_entropy, alpha, opacity_valid, getattr(self.config, 'opacity_hard_fraction', 1.))
        if self.config.empty_opacity_weight>0:
            empty=batch['empty_mask'].to(self.device)
            if observed is not None:empty=empty*(1-observed)
            if 'optical_thickness' not in outputs:raise ValueError('Empty-ray supervision requires an optical-thickness renderer')
            losses['known_empty']=self.config.empty_opacity_weight*weighted_tail_mean(
                outputs['optical_thickness'], empty, getattr(self.config, 'opacity_hard_fraction', 1.))
        if self.training and self.current_train_step<self.config.depth_loss_steps and self.config.depth_loss_mult>0:
            z=batch['depth_image'].to(self.device);valid=(z>0)&torch.isfinite(z)&(weight>0)
            target=camera_z_to_distance(z,outputs['directions_norm'])
            losses['depth_loss']=self.config.depth_loss_mult*weighted_charbonnier(outputs['depth'],target,valid.float())
        if self.training and self.current_train_step<self.config.depth_loss_steps and self.config.depth_distribution_weight>0:
            z=batch['depth_image'].to(self.device);valid=(z>0)&torch.isfinite(z)&(weight>0)
            target=camera_z_to_distance(torch.where(valid,z,0.),outputs['directions_norm'])
            agreement=torch.exp(-.5*((outputs['sample_distances']-target[:,None])/self.config.depth_distribution_sigma).square())
            mass=(outputs['loss_weights']*agreement).sum(1)
            support=weight*valid
            losses['depth_distribution']=self.config.depth_distribution_weight*(-mass.clamp_min(1e-8).log()*support).sum()/support.sum().clamp_min(1e-8)
        if observed is not None:
            from nerfstudio.model_components.observed_background import observed_composite_objective
            remaining=(weight*batch['alpha_valid'].to(self.device)).sum()
            count=observed.sum();total=(remaining+count).clamp_min(1e-8)
            losses['rgb_loss']=losses['rgb_loss']*remaining/total
            losses['observed_composite']=observed_composite_objective(outputs['actor_rgb'],outputs['accumulation'],
                batch['observed_background_rgb'].to(self.device),batch['image'].to(self.device),
                observed,batch['observed_background_error'].to(self.device))*count/total
        return losses

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self,camera_ray_bundle):
        h,w=camera_ray_bundle.origins.shape[:2];out=defaultdict(list)
        for start in range(0,len(camera_ray_bundle),self.config.eval_num_rays_per_chunk):
            rays=camera_ray_bundle.get_row_major_sliced_ray_bundle(start,start+self.config.eval_num_rays_per_chunk).to(self.device)
            values=self(rays)
            for key in ('rgb','actor_rgb','actor_depth','depth','accumulation','background_rgb','background_depth'):
                if key in values:out[key].append(values[key].detach())
        return {k:torch.cat(v).reshape(h,w,-1) for k,v in out.items()}

    def get_image_metrics_and_images(self,outputs,batch):
        from torchmetrics.functional import structural_similarity_index_measure
        pred=outputs['rgb'];gt=batch['image'].to(self.device);mask=batch['evaluation_mask'].to(self.device)[...,0]>0
        if not mask.any():raise ValueError('Empty independent evaluation region')
        yy,xx=torch.where(mask);y0,y1=int(yy.min()),int(yy.max())+1;x0,x1=int(xx.min()),int(xx.max())+1
        a=torch.where(mask[...,None],pred,0)[y0:y1,x0:x1].permute(2,0,1)[None]
        b=torch.where(mask[...,None],gt,0)[y0:y1,x0:x1].permute(2,0,1)[None]
        metrics=dict(psnr=float(-10*torch.log10((pred[mask]-gt[mask]).square().mean().clamp_min(1e-12))),
            ssim=float(structural_similarity_index_measure(a,b,data_range=1.)),lpips=float(self.lpips(a,b)))
        images=dict(img=torch.cat([gt,pred],1),actor=outputs['actor_rgb'],background=outputs['background_rgb'],
                    accumulation=outputs['accumulation'].expand_as(pred))
        return metrics,images


@dataclass
class DistillationPipelineConfig(LookCloserPipelineConfig):
    _target: Type=field(default_factory=lambda: DistillationPipeline)
    datamanager: VanillaDataManagerConfig=field(default_factory=lambda: NativeDistillationDataManagerConfig(
        dataparser=DistillationParserConfig(),
        pixel_sampler=MaskedFrequencySamplerConfig()))
    model: DistillationModelConfig=field(default_factory=DistillationModelConfig)


class DistillationPipeline(LookCloserPipeline):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._frozen_grid_digest=None
        if len(self.cached_freq_maps)!=len(self.datamanager.train_dataset):
            raise ValueError('Every training image requires an audited frequency map')
        if self.config.model.appearance_embedding_dim!=0:
            raise ValueError('Initial distillation comparison requires shared appearance without image embeddings')
        ds=self.datamanager.train_dataset;root=Path(ds.metadata['distillation_root'])
        self.cached_valid_maps={}
        for i,path in enumerate(ds.image_filenames):
            valid=torch.load(root/self.config.frequency_map_dir/(path.stem+'.valid.pt'),map_location='cpu',weights_only=True)
            if valid.dtype!=torch.bool or valid.shape!=self.cached_freq_maps[i].shape:
                raise ValueError('Frequency validity sidecar must be Boolean and match its map')
            self.cached_valid_maps[i]=valid.to(self.device)
        if self.config.model.freeze_frequency_grid and self.config.grid_update_interval!=0:
            raise ValueError('Frozen grid requires periodic updates disabled')

    def load_pipeline(self,state,step):
        super().load_pipeline(state,step)
        if self.config.model.freeze_frequency_grid:
            if not bool(self.model.freq_grid.initialized):raise ValueError('Source checkpoint has uninitialized Frequency Grid')
            self._frozen_grid_digest=self.frequency_digest()

    def frequency_digest(self):
        return {k:tensor_digest(v) for k,v in self.model.freq_grid.state_dict().items()}

    def get_train_loss_dict(self,step):
        if self.config.model.freeze_frequency_grid and self._frozen_grid_digest is None:
            raise RuntimeError('Frozen fine-tune must load a full initialized checkpoint')
        result=super().get_train_loss_dict(step)
        if self._frozen_grid_digest is not None and step%100==0:
            if self.frequency_digest()!=self._frozen_grid_digest:raise RuntimeError('Frequency Grid changed during fine-tune')
        return result

    @torch.no_grad()
    def _update_frequency_grid(self,step):
        if self.config.model.freeze_frequency_grid:raise RuntimeError('Frozen grid update requested')
        rays,batch=self.datamanager.next_train(step)
        index=batch['indices'].to(self.device);c,y,x=index.T
        cameras=self.datamanager.train_dataset.cameras
        # Camera tensors may be CPU while sampled indices are CUDA.
        ci=c.to(cameras.fx.device)
        f2d=torch.empty(len(index),device=self.device)
        valid_patch=torch.zeros(len(index),device=self.device,dtype=torch.bool)
        for i in c.unique().tolist():
            take=c==i
            f2d[take]=self.cached_freq_maps[i].to(self.device)[y[take]//8,x[take]//8]
            valid_patch[take]=self.cached_valid_maps[i][y[take]//8,x[take]//8]
        z=batch['depth_image'].to(self.device).flatten()
        # During real-only initialization only predicted actor depth is available.
        outputs=self.model(rays);norm=rays.metadata['directions_norm'].flatten()
        predicted=outputs['depth'].flatten()/norm
        trusted=(z>0)&torch.isfinite(z)
        use_depth=torch.where(trusted,z,predicted)
        valid=(batch['confidence'].to(self.device).flatten()>0)&(trusted|(outputs['accumulation'].flatten()>.5))
        valid &= torch.isfinite(use_depth)&(use_depth>0)
        # The metadata marks valid frequency patches independently from RGB.
        valid &= valid_patch
        fx=cameras.fx[ci].flatten().to(self.device);fy=cameras.fy[ci].flatten().to(self.device)
        width=cameras.width[ci].flatten().to(self.device);height=cameras.height[ci].flatten().to(self.device)
        resolution=projected_frequency(f2d,fx,fy,width,height,use_depth,self.model.freq_grid.aabb_size_buf)
        # Model depth was rendered with refined training poses. Project it with
        # those same rays, rather than writing frequencies at the old cameras'
        # surface positions. The input bundle remains unchanged for the model.
        projection_rays = rays
        if self.model.training and self.model.config.optimize_training_cameras:
            projection_rays = copy(rays)
            self.model.training_camera_optimizer.apply_to_raybundle(projection_rays)
        position=projection_rays.origins+projection_rays.directions*(use_depth*norm)[:,None]
        self.model.freq_grid.update_max(position[valid],self.model.freq_grid.freq_to_level(resolution[valid]))
        self.model.freq_grid.initialized.copy_(self.model.freq_grid.observed.any())
