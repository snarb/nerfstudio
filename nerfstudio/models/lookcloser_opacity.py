"""Experimental opacity controls and hard first-intersection LookCloser."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import math
import hashlib
import json
import numpy as np
from typing import Literal, Type
import torch
import torch.nn.functional as F
import nerfacc
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.lookcloser_field import LookCloserField
from nerfstudio.fields.lookcloser_surface_field import LookCloserSurfaceField
from nerfstudio.model_components.lookcloser_opacity import first_crossing, sdf_weights, opacity_profile_loss, depth_interval_loss, density_band_loss, cord_termination_mass_loss, density_element_loss, floor_ray_geometry_loss
from nerfstudio.model_components.lookcloser_occupancy import admit_occupancy_points_
from nerfstudio.models.lookcloser import LookCloserModel, LookCloserModelConfig
from nerfstudio.utils.lookcloser_rng import fork_seeded_rng


@dataclass
class LookCloserOpacityModelConfig(LookCloserModelConfig):
    _target: Type = field(default_factory=lambda: LookCloserOpacityModel)
    representation: Literal['volume','opaque_surface','semantic_surface'] = 'volume'
    opacity_regularization: float = 0.
    opacity_ramp_steps: int = 4096
    depth_prior_strength: float = 0.
    depth_prior_loss_mult: float = .01
    depth_prior_objective: Literal['distance','interval_mass','density_band'] = 'distance'
    depth_prior_opacity_target: float = .99
    density_element_regularization: float = 0.
    density_element_samples: int = 1024
    density_element_sparse_bias: float = .99
    floor_ray_prior_path: str | None = None
    floor_ray_transforms_sha256: str | None = None
    floor_ray_regularization: float = 0.
    floor_rays_per_batch: int = 64
    floor_free_sampling: str = 'uniform'
    surface_num_samples: int = 256
    surface_refine_steps: int = 8
    surface_inverse_width: float = 400.
    surface_geometry_gradient_mode: Literal['cdf','implicit'] = 'cdf'
    surface_sdf_features: Literal['reweighted','continuous'] = 'reweighted'
    surface_geometry_head: Literal['shared','independent'] = 'shared'
    eikonal_loss_mult: float = .1
    semantic_grid_path: str | None = None
    semantic_transparency_mult: float = .01
    skip_invalid_depth_updates: bool = True
    allow_unmarked_volume_checkpoint: bool = False
    volume_depth_guided_sampling: bool = True
    cord_sampling_fit_path: str | None = None
    cord_sampling_fit_sha256: str | None = None
    cord_sampling_transforms_sha256: str | None = None
    cord_sampling_radius: float = .002
    cord_sampling_step_size: float = .001
    cord_sampling_max_steps_per_ray: int = 4096
    cord_ray_regularization: float = 0.
    cord_ray_transforms_path: str | None = None
    cord_ray_selection_path: str | None = None
    cord_ray_selection_sha256: str | None = None
    cord_ray_color_regularization: float = 0.
    cord_rays_per_batch: int = 32
    cord_ray_tolerance: float = .003
    cord_ray_tube_radius: float = 0.
    cord_ray_opacity_target: float = .99
    cord_ray_objective: Literal['density_band','termination_mass'] = 'density_band'
    cord_occupancy_radius: float = 0.


class LookCloserOpacityModel(LookCloserModel):
    config: LookCloserOpacityModelConfig

    def populate_modules(self):
        if self.config.representation not in {'volume','opaque_surface','semantic_surface'}:
            raise ValueError('Unknown LookCloser representation')
        if self.config.surface_geometry_gradient_mode not in {'cdf','implicit'}:
            raise ValueError('Unknown surface geometry gradient mode')
        if self.config.surface_sdf_features not in {'reweighted','continuous'}:
            raise ValueError('Unknown signed geometry feature mode')
        if self.config.surface_geometry_head not in {'shared','independent'}:
            raise ValueError('Unknown geometry head mode')
        if self.config.separate_density_encoding and self.config.representation!='volume':
            raise ValueError('Independent density encoding currently supports volume experiments only')
        if self.config.depth_prior_objective not in {'distance','interval_mass','density_band'}:
            raise ValueError('Unknown depth interval objective')
        if self.config.cord_ray_objective not in {'density_band','termination_mass'}:
            raise ValueError('Unknown cord ray objective')
        if not 0 < self.config.depth_prior_opacity_target < 1:
            raise ValueError('Depth-band opacity target must be between zero and one')
        if self.config.depth_prior_objective=='density_band' and (
                self.config.representation!='volume' or self.config.volume_depth_guided_sampling):
            raise ValueError('Density-band prior requires volume and unchanged native RGB sampling')
        if self.config.density_element_samples < 1 or not 0 <= self.config.density_element_sparse_bias < 1:
            raise ValueError('Invalid density element sampling/bias')
        if self.config.density_element_regularization > 0 and (
                self.config.representation != 'volume' or self.config.occupancy_grid_levels != 1):
            raise ValueError('Density elements currently require bounded single-level volume')
        controls=[self.config.opacity_regularization,self.config.depth_prior_strength,
                  self.config.depth_prior_loss_mult,self.config.eikonal_loss_mult,
                  self.config.density_element_regularization,self.config.floor_ray_regularization,
                  self.config.cord_ray_regularization,self.config.cord_ray_color_regularization,
                  self.config.cord_occupancy_radius,self.config.cord_ray_tube_radius]
        if any(not math.isfinite(value) or value<0 for value in controls):
            raise ValueError('Regularization must be nonnegative')
        if self.config.cord_ray_tube_radius > 0 and (
                self.config.cord_ray_regularization <= 0
                or self.config.cord_ray_tube_radius >= self.config.cord_sampling_radius):
            raise ValueError('Cord front-surface radius requires active ray prior inside the fine-sampling tube')
        if self.config.cord_ray_color_regularization>0 and (
                self.config.cord_ray_regularization<=0
                or self.config.cord_ray_selection_path is None
                or self.config.appearance_embedding_dim!=0):
            raise ValueError('Cord band RGB anchors require selected geometry rays and no appearance embedding')
        if self.config.surface_num_samples < 2 or self.config.surface_refine_steps < 0 or self.config.surface_inverse_width <= 0:
            raise ValueError('Invalid surface integrator configuration')
        if self.config.representation!='volume' and (
            self.config.appearance_embedding_dim!=0 or self.config.background_color not in {'black','white'}
            or self.config.rgb_output_parameterization!='sigmoid' or self.config.reconstruction_loss_type!='charbonnier'):
            raise ValueError('Surface experiments require sigmoid RGB/Charbonnier, black or white background, and no appearance embedding')
        self.field_class = LookCloserField if self.config.representation == 'volume' else LookCloserSurfaceField
        super().populate_modules()
        if self.config.cord_sampling_fit_path is not None:
            if (self.config.representation != 'volume'
                    or not math.isfinite(self.config.cord_sampling_radius)
                    or not math.isfinite(self.config.cord_sampling_step_size)
                    or self.config.cord_sampling_radius <= 0
                    or self.config.cord_sampling_step_size <= 0
                    or self.config.cord_sampling_max_steps_per_ray < 1):
                raise ValueError('Local cord sampling requires volume and positive controls')
            fit_path = Path(self.config.cord_sampling_fit_path)
            if hashlib.sha256(fit_path.read_bytes()).hexdigest() != self.config.cord_sampling_fit_sha256:
                raise ValueError('Cord landmark fit changed since request creation')
            fit = json.loads(fit_path.read_text())
            if (fit.get('train_only') is not True or fit.get('coordinate_system') != 'train_normalized_opengl'
                    or fit.get('transforms_sha256') != self.config.cord_sampling_transforms_sha256):
                raise ValueError('Cord geometry must be train-only and match scene calibration')
            fit_ids = set().union(*(set(ids) for ids in fit['fit_camera_ids'].values()))
            if fit['validation_camera_id'] in fit_ids:
                raise ValueError('Held-out camera was used to fit the cord')
            endpoints = torch.tensor([fit['fits'][key]['normalized_world_xyz']
                                      for key in ('junction', 'first_bead')],
                                     dtype=torch.float32, device=self.occupancy_grid.aabbs.device)
            if (not torch.isfinite(endpoints).all() or (endpoints[1]-endpoints[0]).norm() <= 0
                    or not ((endpoints >= self.scene_box.aabb[0].to(endpoints))
                            & (endpoints <= self.scene_box.aabb[1].to(endpoints))).all()):
                raise ValueError('Cord segment is invalid or outside the scene')
            self.register_buffer('cord_sampling_endpoints', endpoints, persistent=False)
        if self.config.cord_occupancy_radius > 0:
            if (self.config.cord_sampling_fit_path is None
                    or not math.isfinite(self.config.cord_occupancy_radius)):
                raise ValueError('Cord occupancy admission requires a pinned fit and finite radius')
            start, end = self.cord_sampling_endpoints
            axis = F.normalize(end-start,dim=0)
            reference = start.new_tensor([1.,0.,0.])
            if torch.abs(torch.dot(axis,reference)) > .9:
                reference = start.new_tensor([0.,1.,0.])
            tangent = F.normalize(torch.linalg.cross(axis,reference),dim=0)
            bitangent = torch.linalg.cross(axis,tangent)
            angle = torch.arange(8,device=start.device,dtype=start.dtype)* (2*math.pi/8)
            offsets = self.config.cord_occupancy_radius*(
                angle.cos()[:,None]*tangent + angle.sin()[:,None]*bitangent)
            offsets = torch.cat([offsets,start.new_zeros(1,3)],dim=0)
            center = start[None] + torch.linspace(0,1,257,device=start.device)[:,None]*(end-start)[None]
            positions = (center[:,None]+offsets[None]).reshape(-1,3)
            admission = torch.zeros_like(self.occupancy_grid.binaries)
            admit_occupancy_points_(admission,self.occupancy_grid.aabbs,positions)
            if not admission.any():
                raise ValueError('Cord occupancy support misses the scene grid')
            self.register_buffer('cord_occupancy_admission',admission,persistent=False)
        if self.config.cord_ray_regularization > 0:
            if (self.config.cord_sampling_fit_path is None or not self.config.cord_ray_transforms_path
                    or self.config.cord_rays_per_batch < 1
                    or not math.isfinite(self.config.cord_ray_tolerance)
                    or self.config.cord_ray_tolerance <= 0
                    or not 0 < self.config.cord_ray_opacity_target < 1):
                raise ValueError('Cord ray prior requires a pinned fit, calibration and positive controls')
            transforms_path = Path(self.config.cord_ray_transforms_path)
            if hashlib.sha256(transforms_path.read_bytes()).hexdigest() != self.config.cord_sampling_transforms_sha256:
                raise ValueError('Cord ray calibration changed since request creation')
            transforms = json.loads(transforms_path.read_text())
            train_names = set(transforms['train_filenames'])
            # Only cameras with independently annotated endpoints constrain the
            # segment. A projected line in an unrelated view is not evidence
            # that the cord is visible or sharp there.
            camera_ids = sorted(set.intersection(*(set(fit['fit_camera_ids'][key])
                                                   for key in ('junction','first_bead'))))
            selected_pairs = None
            if self.config.cord_ray_selection_path is not None:
                selection_path = Path(self.config.cord_ray_selection_path)
                if hashlib.sha256(selection_path.read_bytes()).hexdigest()!=self.config.cord_ray_selection_sha256:
                    raise ValueError('Cord ray selection changed since request creation')
                selection=json.loads(selection_path.read_text())
                if (selection.get('train_only') is not True or selection.get('masks_used') is not False
                        or selection.get('fit_sha256')!=self.config.cord_sampling_fit_sha256
                        or selection.get('transforms_sha256')!=self.config.cord_sampling_transforms_sha256):
                    raise ValueError('Cord ray selection does not match train-only geometry')
                raw_pairs=selection.get('selected_camera_fraction_indices',[])
                if (not raw_pairs or any(not isinstance(pair,list) or len(pair)!=2
                                         or type(pair[0]) is not int or type(pair[1]) is not int
                                         or pair[0] not in camera_ids or not 0<=pair[1]<12
                                         for pair in raw_pairs)):
                    raise ValueError('Invalid selected cord ray indices')
                selected_pairs={tuple(pair) for pair in raw_pairs}
                if len(selected_pairs)!=len(raw_pairs):
                    raise ValueError('Repeated selected cord ray')
            frames = {int(frame['camera_id']): frame for frame in transforms['frames']}
            points = endpoints[0][None] + torch.linspace(.08,.92,12,device=endpoints.device)[:,None] * (endpoints[1]-endpoints[0])[None]
            if selected_pairs is not None:
                overrides=selection.get('stereo_points',{})
                if not isinstance(overrides,dict):
                    raise ValueError('Stereo cord points must be indexed by fraction')
                selected_fractions={index for _,index in selected_pairs}
                for key,value in overrides.items():
                    if not key.isdecimal() or int(key) not in selected_fractions:
                        raise ValueError('Stereo cord point is not a selected train fraction')
                    index=int(key)
                    candidate=torch.as_tensor(value,dtype=points.dtype,device=points.device)
                    if (candidate.shape!=(3,) or not torch.isfinite(candidate).all()
                            or (candidate-points[index]).norm()>.01
                            or not ((candidate>=self.scene_box.aabb[0].to(candidate))
                                    &(candidate<=self.scene_box.aabb[1].to(candidate))).all()):
                        raise ValueError('Stereo cord point is invalid or too far from the landmark fit')
                    points[index]=candidate
            origins, directions, depths = [], [], []
            ray_pairs=[]
            for camera_id in camera_ids:
                frame = frames[camera_id]
                if frame['file_path'] not in train_names:
                    raise ValueError('Cord ray prior contains an evaluation camera')
                pose = torch.tensor(frame['transform_matrix'],dtype=torch.float32,device=endpoints.device)
                origin = pose[:3,3]
                local = (points-origin) @ pose[:3,:3]
                forward = -local[:,2]
                px = frame['fl_x']*local[:,0]/forward + frame['cx']
                py = -frame['fl_y']*local[:,1]/forward + frame['cy']
                valid = (forward>0) & (px>=0) & (px<frame['w']) & (py>=0) & (py<frame['h'])
                ray_pairs.extend((camera_id,int(index)) for index in torch.nonzero(valid).flatten())
                displacement = points[valid]-origin
                origins.append(origin.expand_as(displacement))
                depths.append(displacement.norm(dim=-1))
                directions.append(F.normalize(displacement,dim=-1))
            if not origins or sum(len(value) for value in depths)<1:
                raise ValueError('No valid train-only cord prior rays')
            origins = torch.cat(origins)
            directions = torch.cat(directions)
            depths = torch.cat(depths)
            if selected_pairs is not None:
                if not selected_pairs.issubset(set(ray_pairs)):
                    raise ValueError('Selected cord ray is outside a training camera')
                keep=torch.tensor([pair in selected_pairs for pair in ray_pairs],
                                  dtype=torch.bool,device=depths.device)
                origins,directions,depths=origins[keep],directions[keep],depths[keep]
                if self.config.cord_ray_color_regularization>0:
                    target_by_pair={}
                    for row in selection.get('reviewed',[]):
                        pair=(row.get('camera_id'),row.get('fraction_index'))
                        if pair in selected_pairs:
                            if pair in target_by_pair:
                                raise ValueError('Repeated RGB target for selected cord ray')
                            target_by_pair[pair]=row.get('source_gt')
                    if set(target_by_pair)!=selected_pairs:
                        raise ValueError('Missing selected cord RGB target')
                    targets=torch.tensor([target_by_pair[pair] for pair in ray_pairs if pair in selected_pairs],
                                         dtype=torch.float32,device=depths.device)
                    if (targets.shape!=(len(depths),3) or not torch.isfinite(targets).all()
                            or (targets<0).any() or (targets>1).any()):
                        raise ValueError('Invalid selected cord RGB targets')
                    self.register_buffer('cord_ray_target_rgb',targets,persistent=False)
            if self.config.cord_ray_tube_radius > 0:
                if selected_pairs is not None and selection.get('stereo_points'):
                    raise ValueError('Tube front surface cannot be combined with stereo point overrides')
                tangent=F.normalize(endpoints[1]-endpoints[0],dim=0)
                transverse=(1-(directions*tangent).sum(-1).square()).clamp_min(1e-6).sqrt()
                depths=depths-self.config.cord_ray_tube_radius/transverse
            if (not torch.isfinite(depths).all()
                    or (depths-self.config.cord_ray_tolerance <= self.config.near_plane).any()):
                raise ValueError('Cord prior geometry invalid or too close to near plane')
            self.register_buffer('cord_ray_origins',origins,persistent=False)
            self.register_buffer('cord_ray_directions',directions,persistent=False)
            self.register_buffer('cord_ray_depths',depths,persistent=False)
        if self.config.floor_ray_regularization > 0:
            if self.config.floor_free_sampling not in {'uniform','stratified'}:
                raise ValueError('Unknown floor free-space sampling mode')
            if self.config.representation!='volume' or not self.config.floor_ray_prior_path or self.config.floor_rays_per_batch<1:
                raise ValueError('Floor rays require volume, an explicit prior and a positive ray batch')
            path=Path(self.config.floor_ray_prior_path);manifest=json.loads(path.read_text())
            if not manifest.get('train_only') or manifest.get('transforms_sha256')!=self.config.floor_ray_transforms_sha256:
                raise ValueError('Floor prior must match train-only data calibration')
            archive=path.parent/manifest['archive']
            if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['archive_sha256']:
                raise ValueError('Floor ray archive hash mismatch')
            with np.load(archive,allow_pickle=False) as arrays:
                for key in ['origins','directions','depth','tolerance']:
                    value=torch.tensor(arrays[key],dtype=torch.float32,device=self.occupancy_grid.aabbs.device)
                    if not torch.isfinite(value).all():raise ValueError('Nonfinite floor ray data')
                    self.register_buffer('floor_'+key,value,persistent=False)
            if (self.floor_depth.ndim!=1 or len(self.floor_depth)<1 or self.floor_origins.shape!=(len(self.floor_depth),3)
                    or self.floor_directions.shape!=self.floor_origins.shape or self.floor_tolerance.shape!=self.floor_depth.shape
                    or (self.floor_tolerance<=0).any() or (self.floor_depth-self.floor_tolerance<=self.config.near_plane).any()
                    or not torch.allclose(self.floor_directions.norm(dim=-1),torch.ones_like(self.floor_depth),atol=1e-5)):
                raise ValueError('Invalid floor ray geometry')
            offsets=torch.linspace(-1,1,9,device=self.floor_origins.device)
            distances=self.floor_depth[:,None]+self.floor_tolerance[:,None]*offsets
            positions=self.floor_origins[:,None]+self.floor_directions[:,None]*distances[...,None]
            if not ((positions>=self.scene_box.aabb[0])&(positions<=self.scene_box.aabb[1])).all():
                raise ValueError('Floor uncertainty band outside scene box')
            admission=torch.zeros_like(self.occupancy_grid.binaries)
            admit_occupancy_points_(admission,self.occupancy_grid.aabbs,positions.reshape(-1,3))
            self.register_buffer('floor_admission',admission,persistent=False)
        if self.config.representation!='volume':
            self.field.continuous_sdf = self.config.surface_sdf_features == 'continuous'
            if self.config.surface_geometry_head == 'independent':
                self.field.enable_independent_geometry()
        self.freq_grid.skip_invalid_depth_updates = self.config.skip_invalid_depth_updates
        self.register_buffer('opacity_representation',torch.tensor(
            ['volume','opaque_surface','semantic_surface'].index(self.config.representation),dtype=torch.int64))
        self.register_buffer('semantic_permissions',torch.zeros(1,1,1,dtype=torch.bool))
        if self.config.representation == 'semantic_surface':
            if self.config.semantic_grid_path is None:
                raise ValueError('semantic_surface requires reviewed train-only spatial permissions')
            payload = torch.load(Path(self.config.semantic_grid_path),map_location='cpu',weights_only=True)
            if payload.get('train_only') is not True or payload.get('reviewed') is not True:
                raise ValueError('Semantic permissions must be train-only and reviewed')
            if not torch.allclose(payload['aabb'].float(),self.scene_box.aabb.cpu(),atol=1e-6):
                raise ValueError('Semantic grid AABB differs from the field')
            self.semantic_permissions = payload['permissions'].bool()
            if self.semantic_permissions.ndim!=3 or min(self.semantic_permissions.shape)<1:
                raise ValueError('Semantic permissions must be a nonempty 3D grid')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        key = prefix+'opacity_representation'
        if (key not in state_dict and self.config.allow_unmarked_volume_checkpoint
                and self.config.representation=='volume'):
            # A standard volume field has the identical parameter/optimizer
            # layout. Initialize only the extension's bookkeeping buffers;
            # strict loading still checks every actual field/grid parameter.
            state_dict[key]=self.opacity_representation.detach().clone()
            state_dict[prefix+'semantic_permissions']=self.semantic_permissions.detach().clone()
        if key in state_dict and int(state_dict[key]) != int(self.opacity_representation):
            error_msgs.append('Checkpoint surface/volume representation is incompatible with config')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def get_training_callbacks(self,attributes):
        if self.config.representation != 'volume':
            return []  # Density occupancy updates are not valid for an SDF.
        return super().get_training_callbacks(attributes)

    def _refine_adaptive_samples(self, ray_bundle, ray_samples, ray_indices, stats):
        if self.config.cord_sampling_fit_path is None:
            return ray_samples, ray_indices, stats
        with torch.no_grad():
            start, end = self.cord_sampling_endpoints.to(ray_bundle.origins)
            o, d = ray_bundle.origins, ray_bundle.directions
            direction_norm_sq = (d*d).sum(-1).clamp_min(1e-12)
            segment = end-start
            projected_segment = segment - d*((segment*d).sum(-1)/direction_norm_sq)[:, None]
            denominator = (projected_segment*projected_segment).sum(-1).clamp_min(1e-12)
            fraction = -((start-o)*projected_segment).sum(-1)/denominator
            fraction = fraction.clamp(0,1)
            closest = start+fraction[:, None]*segment
            ray_t = ((closest-o)*d).sum(-1)/direction_norm_sq
            distance_sq = ((closest-o-ray_t[:, None]*d)**2).sum(-1)
            near = ray_bundle.nears.reshape(-1) if ray_bundle.nears is not None else torch.full_like(ray_t,self.config.near_plane)
            far = ray_bundle.fars.reshape(-1) if ray_bundle.fars is not None else torch.full_like(ray_t,self.config.far_plane)
            selected = ((distance_sq <= self.config.cord_sampling_radius**2)
                        & (ray_t >= near) & (ray_t <= far))
            selected_ids = torch.nonzero(selected,as_tuple=False).flatten()
        if selected_ids.numel() == 0:
            return ray_samples, ray_indices, stats
        # Sampling every ray once with the ordinary recipe preserves exact
        # unaffected-ray outputs. Replace only selected rays with a finer native
        # nerfacc traversal; never concatenate overlapping sample intervals.
        fine_bundle = ray_bundle[selected_ids]
        with fork_seeded_rng(self.config.training_seed,'cord_local_sampling',self.current_train_step,self.device):
            fine_samples, fine_local_indices, _ = self.adaptive_sampler(
                ray_bundle=fine_bundle,
                render_step_size=float(self.config.cord_sampling_step_size),
                near_plane=float(self.config.near_plane),
                far_plane=float(self.config.far_plane),
                alpha_thre=float(self.config.alpha_thre),
                early_stop_eps=float(self.config.transmittance_threshold),
                cone_angle=float(self.config.cone_angle),
                adaptive_min_step_size=float(self.config.adaptive_min_step_size),
                adaptive_max_step_size=float(self.config.adaptive_max_step_size),
                adaptive_min_frequency_level=float(self.config.adaptive_min_frequency_level),
                adaptive_max_frequency_level=(None if self.config.adaptive_max_frequency_level is None
                                             else float(self.config.adaptive_max_frequency_level)),
                adaptive_interval_level_mode=self.config.adaptive_interval_level_mode,
                max_steps_per_ray=int(self.config.cord_sampling_max_steps_per_ray),
                corrected_allocator=bool(self.config.corrected_arm_allocator),
            )
        fine_indices = selected_ids[fine_local_indices]
        if fine_indices.numel() == 0:
            return ray_samples, ray_indices, stats
        replaced = torch.zeros_like(selected)
        replaced[fine_indices] = True
        kept = ~replaced[ray_indices]
        standard_samples, standard_indices = ray_samples[kept], ray_indices[kept]
        merged, merged_indices = self._concat_packed_ray_samples(
            standard_samples, standard_indices, fine_samples, fine_indices
        )
        packed = nerfacc.pack_info(merged_indices, len(ray_bundle))
        counts = packed[:, 1]
        stats.num_samples = torch.tensor(merged_indices.numel(),device=merged_indices.device)
        stats.mean_samples_per_ray = counts.float().mean()
        stats.max_samples_per_ray = counts.max().float()
        cap = torch.where(replaced,int(self.config.cord_sampling_max_steps_per_ray),int(self.config.max_steps_per_ray))
        stats.saturation_rate = (counts >= cap).float().mean()
        stats.packed_info = packed
        return merged, merged_indices, stats

    def _admit_depth_occupancy(self) -> None:
        super()._admit_depth_occupancy()
        if self.config.cord_occupancy_radius > 0:
            self.occupancy_grid.binaries.logical_or_(self.cord_occupancy_admission)

    def _semantic_allowed(self,positions):
        aabb = self.scene_box.aabb.to(positions)
        uv = (positions-aabb[0])/(aabb[1]-aabb[0])
        shape = torch.tensor(self.semantic_permissions.shape,device=positions.device)
        ijk = (uv*shape).long()
        inside = ((uv>=0)&(uv<1)).all(-1)
        ijk = torch.minimum(ijk.clamp_min(0),shape-1)
        return inside & self.semantic_permissions[ijk[:,0],ijk[:,1],ijk[:,2]]

    def _surface_outputs(self,ray_bundle):
        o,d = ray_bundle.origins,ray_bundle.directions
        n = len(o)
        count = self.config.surface_num_samples
        near,far = ray_bundle.nears,ray_bundle.fars
        edges = torch.linspace(0,1,count+1,device=o.device)
        t = near+(far-near)*edges[None,:]
        t = self._guided_edges(ray_bundle,t)
        count=t.shape[1]-1
        points = o[:,None]+d[:,None]*t[...,None]
        implicit = self.training and self.config.surface_geometry_gradient_mode == 'implicit'
        if implicit:
            with torch.no_grad():
                sdf = self.field.query_surface(points.reshape(-1,3)).reshape(n,count+1)
        else:
            sdf = self.field.query_surface(points.reshape(-1,3)).reshape(n,count+1)
        first,hit = first_crossing(sdf.detach())
        crossing = (sdf.detach()[:,:-1]*sdf.detach()[:,1:]<=0) & (
            (sdf.detach()[:,:-1]!=0)|(sdf.detach()[:,1:]!=0))
        if self.config.representation == 'opaque_surface':
            ray_ids = torch.arange(n,device=o.device)
            brackets = first
            valid = hit
        else:
            # Spatially permitted transparency may reveal subsequent surfaces.
            ray_ids,brackets = torch.nonzero(crossing,as_tuple=True)
            valid = torch.ones_like(ray_ids,dtype=torch.bool)
        with torch.no_grad():
            lo,hi = t[ray_ids,brackets],t[ray_ids,brackets+1]
            s_lo = sdf.detach()[ray_ids,brackets]
            for _ in range(self.config.surface_refine_steps):
                mid = (lo+hi)*.5
                s_mid = self.field.query_surface(o[ray_ids]+d[ray_ids]*mid[:,None]).flatten()
                same = s_mid*s_lo>0
                lo = torch.where(same,mid,lo)
                hi = torch.where(same,hi,mid)
                s_lo = torch.where(same,s_mid,s_lo)
            root = (lo+hi)*.5
            root_points = o[ray_ids]+d[ray_ids]*root[:,None]
        differentiable_root = root
        query_points = root_points
        if implicit:
            # dt/dtheta = -S_theta / (grad(S).direction), evaluated at the
            # actual first intersection. The subtraction leaves the hard
            # forward root exactly unchanged. Guard grazing derivatives.
            with torch.no_grad():
                denominator = (self.field.finite_gradients(root_points)*d[ray_ids]).sum(-1)
                sign = torch.where(denominator>=0,1.,-1.)
                denominator = sign*denominator.abs().clamp_min(.05)
            root_sdf = self.field.query_surface(root_points).flatten()
            differentiable_root = root-(root_sdf-root_sdf.detach())/denominator
            query_points = o[ray_ids]+d[ray_ids]*differentiable_root[:,None]
        _,color,alpha = self.field.query_surface(query_points,d[ray_ids])
        if self.config.representation == 'opaque_surface':
            hard_rgb = color*valid[:,None]
            hard_acc = valid[:,None].float()
            hard_depth = differentiable_root[:,None]*valid[:,None]
            transparency = color.sum()*0
        else:
            allowed = self._semantic_allowed(root_points)
            # Nerfacc scans require FP32 even under mixed-precision training.
            # Accumulate colors in the same dtype; casting retains head grads.
            alpha = alpha.float()
            alpha = torch.where(allowed[:,None],alpha,torch.ones_like(alpha))
            # Packed ray IDs are sorted; opaque crossings set all later weights to zero.
            import nerfacc
            packing = nerfacc.pack_info(ray_ids,n)
            w,_ = nerfacc.render_weight_from_alpha(alpha[:,0],packed_info=packing)
            hard_rgb = w.new_zeros(n,3).index_add(0,ray_ids,w[:,None]*color.float())
            hard_acc = w.new_zeros(n,1).index_add(0,ray_ids,w[:,None])
            expected_depth = w.new_zeros(n,1).index_add(0,ray_ids,w[:,None]*differentiable_root[:,None])/hard_acc.clamp_min(1e-6)
            # Surface geometry/frequency correspondence uses an actual surface,
            # never the mean location between hair and background layers.
            has_crossing=packing[:,1]>0
            hard_depth=w.new_zeros(n,1)
            hard_depth[has_crossing]=differentiable_root[packing[has_crossing,0],None]
            front_allowed=torch.zeros(n,dtype=torch.bool,device=o.device)
            front_allowed[has_crossing]=allowed[packing[has_crossing,0]]
            # RGB is averaged over the whole ray batch. Use that same domain
            # so a tiny permission region is not overwhelmed by its penalty.
            transparency = ((1-alpha[allowed]).sum()/max(n,1) if allowed.any() else color.sum()*0)
        if self.config.background_color == 'white':
            hard_rgb = hard_rgb+1-hard_acc
        output = dict(rgb=hard_rgb,depth=hard_depth,accumulation=hard_acc,
                      num_samples_per_ray=torch.full((n,1),count+self.config.surface_refine_steps+1,device=o.device),
                      surface_hit=hit[:,None].float(),semantic_transparency=transparency)
        if self.config.representation == 'semantic_surface':
            output['expected_depth']=expected_depth
            output['prior_opaque']=~front_allowed[:,None]
        if not self.training and self.config.representation == 'opaque_surface':
            gradients = self.field.finite_gradients(root_points)
            output['normals'] = F.normalize(gradients,dim=-1)*valid[:,None]
        needs_surrogate = ((self.training and not implicit) or getattr(self,'_surrogate_eval',False)
                           or (implicit and not bool(hit.all())))
        if needs_surrogate:
            if implicit:
                # Only missed rays need a CDF fallback to form a surface.
                sdf = self.field.query_surface(points.reshape(-1,3)).reshape(n,count+1)
            mids = (t[:,:-1]+t[:,1:])*.5
            sample_points = o[:,None]+d[:,None]*mids[...,None]
            _,rgb,_ = self.field.query_surface(sample_points.reshape(-1,3),d[:,None].expand(-1,count,-1).reshape(-1,3))
            inverse_width = self.config.surface_inverse_width*min(1.,.1+self.current_train_step/4096.)
            weights = sdf_weights(sdf,inverse_width)
            soft_rgb = (weights[...,None]*rgb.reshape(n,count,3)).sum(1)
            soft_acc = weights.sum(-1,keepdim=True)
            soft_depth = (weights*mids).sum(-1,keepdim=True)/soft_acc.clamp_min(1e-6)
            if self.config.background_color == 'white':
                soft_rgb = soft_rgb+1-soft_acc
            if self.training:
                # Hard forward, surrogate backward, including rays without a hit.
                # Appearance/opacity learn at actual intersections. Only geometry
                # uses the surrogate, so semantic alpha receives RGB gradients.
                geometry_rgb = (weights[...,None]*rgb.detach().reshape(n,count,3)).sum(1)
                if self.config.background_color == 'white':
                    geometry_rgb = geometry_rgb+1-soft_acc
                if implicit:
                    missing = (~hit)[:,None]
                    output['rgb'] = hard_rgb+(geometry_rgb-geometry_rgb.detach())*missing
                    output['depth'] = hard_depth+(soft_depth-soft_depth.detach())*missing
                else:
                    output['rgb'] = hard_rgb+geometry_rgb-geometry_rgb.detach()
                    output['depth'] = hard_depth.detach()+soft_depth-soft_depth.detach()
            output.update(prior_weights=weights.reshape(-1,1),prior_mids=mids.reshape(-1,1),
                          prior_indices=torch.arange(n,device=o.device).repeat_interleave(count),
                          soft_rgb=soft_rgb,soft_accumulation=soft_acc,
                          surrogate_rgb_gap=(hard_rgb-soft_rgb).abs().mean())
        if self.training and self.config.eikonal_loss_mult>0:
            # A failed bracket is not a surface. Semantic back surfaces hidden
            # by a fully opaque first hit should not change this regularizer.
            visible_roots = valid if self.config.representation == 'opaque_surface' else w.detach()>1e-4
            surface_points = root_points.detach()[visible_roots]
            sample = surface_points[::max(1,len(surface_points)//256)]
            aabb = self.scene_box.aabb.to(o)
            uniform = aabb[0]+torch.rand(256,3,device=o.device)*(aabb[1]-aabb[0])
            sample = torch.cat([sample,uniform])
            metadata=ray_bundle.metadata or {}
            if implicit and self.config.depth_prior_strength>0 and 'depth_valid' in metadata:
                trusted=metadata['depth_valid'].reshape(-1)
                if self.config.representation == 'semantic_surface':
                    trusted=trusted&~front_allowed
                measured_points=(o+d*metadata['depth_prior'])[trusted].detach()
                measured_points=measured_points[::max(1,len(measured_points)//256)][:256]
                sample=torch.cat([sample,measured_points])
            if len(sample):
                gradients = self.field.finite_gradients(sample)
                output['eikonal'] = (gradients.norm(dim=-1)-1).square().mean()
        metadata=ray_bundle.metadata or {}
        if self.training and self.config.depth_prior_strength>0 and 'depth_prior' in metadata:
            gt=metadata['depth_prior']
            output['prior_sdf']=self.field.query_surface(o+d*gt)
            fractions=torch.tensor([.25,.5,.75,.95],device=o.device)
            probes=o[:,None]+d[:,None]*(gt*fractions[None,:])[...,None]
            output['prior_free_sdf']=self.field.query_surface(probes.reshape(-1,3)).reshape(n,4)
            if implicit:
                tolerance=torch.maximum(2*metadata['depth_uncertainty'],.01*gt/self.config.depth_prior_strength)
                bracket=torch.cat([gt-2*tolerance,gt+2*tolerance],-1)
                bracket=torch.minimum(torch.maximum(bracket,near),far)
                probes=o[:,None]+d[:,None]*bracket[...,None]
                output['prior_bracket_sdf']=self.field.query_surface(probes.reshape(-1,3)).reshape(n,2)
        return output

    def _guided_edges(self,ray_bundle,t):
        metadata=ray_bundle.metadata or {}
        if self.config.depth_prior_strength<=0 or not self.training or 'depth_prior' not in metadata:
            return t
        valid=metadata['depth_valid'].reshape(-1)
        if not valid.any():
            return t
        depth=metadata['depth_prior']
        tolerance=torch.maximum(2*metadata['depth_uncertainty'],.01*depth/self.config.depth_prior_strength)
        offsets=torch.linspace(-2,2,65,device=t.device)
        fine=depth+tolerance*offsets[None,:]
        fine=torch.minimum(torch.maximum(fine,ray_bundle.nears),ray_bundle.fars)
        fine=torch.where(valid[:,None],fine,ray_bundle.nears.expand_as(fine))
        return torch.sort(torch.cat([t,fine],-1),-1).values

    def _guided_volume_outputs(self,ray_bundle):
        """Partition, rather than overlap, intervals around trusted depths.

        The auxiliary samples query the actual learned field. No GT depth/RGB
        is substituted into the render, and eval uses the normal volume sampler.
        """
        o,d=ray_bundle.origins,ray_bundle.directions
        n=len(o)
        near,far=ray_bundle.nears,ray_bundle.fars
        edges=torch.linspace(0,1,self.config.fixed_num_samples_per_ray+1,device=o.device)
        t=self._guided_edges(ray_bundle,near+(far-near)*edges[None,:])
        starts,ends=t[:,:-1],t[:,1:]
        mids=(starts+ends)*.5
        count=mids.shape[1]
        positions=o[:,None]+d[:,None]*mids[...,None]
        sigma,rgb=self.field.query_points(positions.reshape(-1,3),d[:,None].expand(-1,count,-1).reshape(-1,3))
        sigma=sigma.float().reshape(n,count)
        alpha=1-torch.exp(-sigma*(ends-starts))
        trans=torch.cumprod(torch.cat([torch.ones_like(alpha[:,:1]),1-alpha+1e-10],-1),-1)[:,:-1]
        weights=alpha*trans
        acc=weights.sum(-1,keepdim=True)
        color=(weights[...,None]*rgb.reshape(n,count,3)).sum(1)
        if self.config.background_color=='white':
            color=color+1-acc
        spacing_starts=(starts-near)/(far-near).clamp_min(1e-6)
        spacing_ends=(ends-near)/(far-near).clamp_min(1e-6)
        return dict(rgb=color,accumulation=acc,depth=(weights*mids).sum(-1,keepdim=True)/acc.clamp_min(1e-6),
                    num_samples_per_ray=torch.full((n,),count,device=o.device),
                    packed_weights=weights.reshape(-1,1),packed_spacing_starts=spacing_starts.reshape(-1,1),
                    packed_spacing_ends=spacing_ends.reshape(-1,1),
                    packed_ray_indices=torch.arange(n,device=o.device).repeat_interleave(count))

    def _volume_outputs(self,ray_bundle):
        metadata=ray_bundle.metadata or {}
        if (not self.config.volume_depth_guided_sampling or self.config.depth_prior_strength<=0
                or not self.training or 'depth_valid' not in metadata or not metadata['depth_valid'].any()):
            return super().get_outputs(ray_bundle)
        trusted=metadata['depth_valid'].reshape(-1)
        groups=[]
        for selected,guided in [(trusted,True),(~trusted,False)]:
            original=torch.nonzero(selected,as_tuple=True)[0]
            if not len(original):
                continue
            local=ray_bundle[original]
            result=self._guided_volume_outputs(local) if guided else super().get_outputs(local)
            if 'packed_weights' not in result:
                count=result['loss_weights'].shape[1]
                result.update(packed_weights=result['loss_weights'].reshape(-1,1),
                              packed_spacing_starts=result['loss_ray_samples'].spacing_starts.reshape(-1,1),
                              packed_spacing_ends=result['loss_ray_samples'].spacing_ends.reshape(-1,1),
                              packed_ray_indices=torch.arange(len(original),device=original.device).repeat_interleave(count))
            result['packed_ray_indices']=original[result['packed_ray_indices']]
            groups.append((original,result))
        output={}
        for key in ['rgb','depth','accumulation']:
            value=groups[0][1][key]
            merged=value.new_zeros(len(ray_bundle),value.shape[-1])
            for original,result in groups:
                merged=merged.index_copy(0,original,result[key])
            output[key]=merged
        for key in ['packed_weights','packed_spacing_starts','packed_spacing_ends','packed_ray_indices']:
            output[key]=torch.cat([result[key] for _,result in groups])
        order=torch.argsort(output['packed_ray_indices'],stable=True)
        for key in ['packed_weights','packed_spacing_starts','packed_spacing_ends','packed_ray_indices']:
            output[key]=output[key][order]
        output['num_samples_per_ray']=torch.bincount(output['packed_ray_indices'],minlength=len(ray_bundle))
        return output

    def get_outputs(self,ray_bundle):
        if self.config.representation != 'volume':
            return self._surface_outputs(ray_bundle)
        if self.config.floor_ray_regularization > 0:
            self.occupancy_grid.binaries.logical_or_(self.floor_admission)
        outputs = self._volume_outputs(ray_bundle)
        if self.training and self.config.floor_ray_regularization > 0:
            with fork_seeded_rng(self.config.training_seed,'floor_rays',self.current_train_step,self.device):
                ids=torch.randint(len(self.floor_depth),(self.config.floor_rays_per_batch,),device=self.device)
                if self.config.floor_free_sampling=='stratified':
                    fraction=(torch.arange(65,device=self.device)[None]+
                              torch.rand(len(ids),65,device=self.device))/65
                else:
                    fraction=torch.linspace(0,1,65,device=self.device)[None]
            o,d=self.floor_origins[ids],self.floor_directions[ids]
            center,eps=self.floor_depth[ids,None],self.floor_tolerance[ids,None]
            near=self.config.near_plane
            length=(center-eps-near).flatten()
            distances=near+length[:,None]*fraction
            points=o[:,None]+d[:,None]*distances[...,None]
            outputs['floor_free_sigma']=self.field.density_fn(points.reshape(-1,3)).reshape(len(ids),65)
            distances=center+eps*torch.linspace(-1,1,9,device=self.device)
            points=o[:,None]+d[:,None]*distances[...,None]
            outputs['floor_band_sigma']=self.field.density_fn(points.reshape(-1,3)).reshape(len(ids),9)
            outputs['floor_free_length']=length
            outputs['floor_tolerance']=eps.flatten()
            outputs['floor_free_monte_carlo']=self.config.floor_free_sampling=='stratified'
        if self.training and self.config.cord_ray_regularization > 0:
            with fork_seeded_rng(self.config.training_seed,'cord_rays',self.current_train_step,self.device):
                ids=torch.randint(len(self.cord_ray_depths),(self.config.cord_rays_per_batch,),device=self.device)
            o,d=self.cord_ray_origins[ids],self.cord_ray_directions[ids]
            center=self.cord_ray_depths[ids,None]
            eps=self.config.cord_ray_tolerance
            if self.config.cord_ray_objective=='density_band':
                distances=center+eps*torch.linspace(-1,1,17,device=self.device)[None,:]
                points=o[:,None]+d[:,None]*distances[...,None]
                outputs['cord_band_sigma']=self.field.density_fn(points.reshape(-1,3)).reshape(len(ids),17)
            else:
                edges=torch.linspace(-5,5,61,device=self.device)
                distances=center+eps*((edges[:-1]+edges[1:])/2)[None,:]
                points=o[:,None]+d[:,None]*distances[...,None]
                outputs['cord_shell_sigma']=self.field.density_fn(points.reshape(-1,3)).reshape(len(ids),60)
            fractions=o.new_tensor([.25,.5,.75])
            margins=o.new_tensor([2.,4.,8.])
            free_distances=torch.cat([center*fractions[None,:],center-eps*margins[None,:]],-1)
            free_distances=free_distances.clamp_min(self.config.near_plane)
            points=o[:,None]+d[:,None]*free_distances[...,None]
            outputs['cord_free_sigma']=self.field.density_fn(points.reshape(-1,3)).reshape(len(ids),6)
            outputs['cord_tolerance']=center.new_full((len(ids),),eps)
            if self.config.cord_ray_color_regularization>0:
                count=7
                step=2*eps/count
                mid=center+step*(torch.arange(count,device=self.device)-(count-1)/2)[None,:]
                starts,ends=mid-step/2,mid+step/2
                frustums=Frustums(origins=o[:,None,:].expand(-1,count,-1),
                                  directions=d[:,None,:].expand(-1,count,-1),
                                  starts=starts[...,None],ends=ends[...,None],
                                  pixel_area=torch.ones_like(starts)[...,None])
                samples=RaySamples(frustums=frustums,deltas=(ends-starts)[...,None])
                with torch.no_grad():
                    band_density,band_geo=self.field.get_density(samples)
                    band_weights=samples.get_weights(band_density)
                band_rgb=self.field.get_outputs(samples,density_embedding=band_geo)[FieldHeadNames.RGB]
                outputs['cord_color_rgb']=(band_weights*band_rgb).sum(1) / band_weights.sum(1).clamp_min(1e-6)
                outputs['cord_color_target']=self.cord_ray_target_rgb[ids]
        if 'packed_weights' in outputs:
            weights,indices = outputs['packed_weights'],outputs['packed_ray_indices']
            starts,ends = outputs['packed_spacing_starts'],outputs['packed_spacing_ends']
        else:
            weights = outputs['loss_weights'].reshape(-1,1)
            starts = outputs['loss_ray_samples'].spacing_starts.reshape(-1,1)
            ends = outputs['loss_ray_samples'].spacing_ends.reshape(-1,1)
            indices = torch.arange(len(ray_bundle),device=weights.device).repeat_interleave(outputs['loss_weights'].shape[1])
        if self.training and self.config.opacity_regularization>0:
            outputs['opacity_profile'] = opacity_profile_loss(weights,starts,ends,indices,len(ray_bundle))
        if self.training and self.config.density_element_regularization > 0:
            active=torch.nonzero(self.occupancy_grid.binaries[0],as_tuple=False)
            if len(active):
                with fork_seeded_rng(self.config.training_seed,'density_elements',self.current_train_step,self.device):
                    ids=torch.randint(len(active),(self.config.density_element_samples,),device=self.device)
                    offsets=torch.rand(self.config.density_element_samples,3,device=self.device)
                box=self.occupancy_grid.aabbs[0]
                resolution=self.occupancy_grid.resolution.to(box)
                cell_size=(box[3:]-box[:3])/resolution
                points=box[:3]+(active[ids].to(box)+offsets)*cell_size
                outputs['density_element_sigma']=self.field.density_fn(points)
                outputs['density_element_length']=float(cell_size.norm())
            else:
                outputs['density_element_sigma']=weights.new_empty(0)
                outputs['density_element_length']=1.
        if self.training and self.config.depth_prior_strength>0:
            if self.config.depth_prior_objective=='density_band':
                metadata=ray_bundle.metadata or {}
                if 'depth_prior' in metadata:
                    depth=metadata['depth_prior'].reshape(-1)
                    uncertainty=metadata['depth_uncertainty'].reshape(-1)
                    tolerance=torch.maximum(2*uncertainty,.01*depth/self.config.depth_prior_strength)
                    valid=metadata['depth_valid'].reshape(-1).bool()
                    near,far=ray_bundle.nears.reshape(-1),ray_bundle.fars.reshape(-1)
                    valid=valid&torch.isfinite(depth)&torch.isfinite(tolerance)&(tolerance>0)
                    valid=valid&(depth-tolerance>=near)&(depth+tolerance<=far)
                    ids=torch.nonzero(valid).flatten()
                    outputs['density_band_tolerance']=tolerance[ids]
                    if len(ids):
                        o,d=ray_bundle.origins[ids],ray_bundle.directions[ids]
                        center,eps=depth[ids,None],tolerance[ids,None]
                        offsets=torch.linspace(-1,1,9,device=o.device)
                        distances=center+eps*offsets[None,:]
                        points=o[:,None]+d[:,None]*distances[...,None]
                        outputs['density_band_sigma']=self.field.density_fn(points.reshape(-1,3)).reshape(len(ids),9)
                        fractions=o.new_tensor([.25,.5,.75])
                        margins=o.new_tensor([2.,4.,8.])
                        free_distances=torch.cat([center*fractions[None,:],center-eps*margins[None,:]],-1)
                        free_distances=torch.maximum(free_distances,near[ids,None])
                        points=o[:,None]+d[:,None]*free_distances[...,None]
                        outputs['density_free_sigma']=self.field.density_fn(points.reshape(-1,3)).reshape(len(ids),6)
                    else:
                        outputs['density_band_sigma']=weights.new_empty(0,9)
                        outputs['density_free_sigma']=weights.new_empty(0,6)
            else:
                near = ray_bundle.nears[indices]
                span = ray_bundle.fars[indices]-near
                outputs.update(prior_weights=weights,prior_mids=near+span*(starts+ends)*.5,prior_indices=indices)
        return outputs

    def get_loss_dict(self,outputs,batch,metrics_dict=None):
        if self.config.representation == 'volume':
            losses = super().get_loss_dict(outputs,batch,metrics_dict)
        else:
            image = batch['image'].to(self.device)
            losses = {'rgb_loss':torch.sqrt((outputs['rgb']-image).square()+1e-4).mean()}
            if 'eikonal' in outputs:
                losses['eikonal_loss'] = self.config.eikonal_loss_mult*outputs['eikonal']
            if self.config.representation == 'semantic_surface':
                losses['semantic_transparency_loss'] = self.config.semantic_transparency_mult*outputs['semantic_transparency']
        if self.config.opacity_regularization>0 and 'opacity_profile' in outputs:
            ramp = min(1.,max(0.,(self.current_train_step-self.config.adaptive_warmup_steps)/max(self.config.opacity_ramp_steps,1)))
            deficit = -torch.log(outputs['accumulation'].clamp_min(1e-6)).mean()
            losses['opacity_loss'] = self.config.opacity_regularization*ramp*(outputs['opacity_profile']+deficit)
        if self.config.density_element_regularization > 0 and 'density_element_sigma' in outputs:
            losses['density_element_loss']=self.config.density_element_regularization*density_element_loss(
                outputs['density_element_sigma'],outputs['density_element_length'],self.config.density_element_sparse_bias)
        if self.config.floor_ray_regularization > 0 and 'floor_free_sigma' in outputs:
            losses['floor_geometry_loss']=self.config.floor_ray_regularization*floor_ray_geometry_loss(
                outputs['floor_free_sigma'],outputs['floor_free_length'],outputs['floor_band_sigma'],outputs['floor_tolerance'],
                free_monte_carlo=outputs.get('floor_free_monte_carlo',False))
        if self.config.cord_ray_regularization > 0:
            if 'cord_band_sigma' in outputs:
                value=density_band_loss(outputs['cord_band_sigma'],outputs['cord_tolerance'],
                                        outputs['cord_free_sigma'],self.config.cord_ray_opacity_target)
            elif 'cord_shell_sigma' in outputs:
                value=cord_termination_mass_loss(outputs['cord_shell_sigma'],outputs['cord_tolerance'],
                                                 outputs['cord_free_sigma'],self.config.cord_ray_opacity_target)
            else:
                value=None
            if value is not None:
                losses['cord_ray_geometry_loss']=self.config.cord_ray_regularization*value
        if 'cord_color_rgb' in outputs:
            losses['cord_ray_color_loss']=self.config.cord_ray_color_regularization*torch.sqrt(
                (outputs['cord_color_rgb']-outputs['cord_color_target']).square()+1e-4).mean()
        if self.config.depth_prior_strength>0 and 'depth_prior' in batch:
            strength = self.config.depth_prior_strength
            if self.config.depth_prior_objective=='density_band':
                if 'density_band_sigma' in outputs:
                    losses['trusted_depth_loss']=self.config.depth_prior_loss_mult*strength*density_band_loss(
                        outputs['density_band_sigma'],outputs['density_band_tolerance'],
                        outputs['density_free_sigma'],self.config.depth_prior_opacity_target)
                return losses
            depth = batch['depth_prior'].to(self.device)
            uncertainty = batch['depth_uncertainty'].to(self.device)
            valid = batch['depth_valid'].to(self.device)
            if 'prior_opaque' in outputs:
                valid=valid&outputs['prior_opaque']
            tolerance = torch.maximum(2*uncertainty,.01*depth/strength)
            physical_surface = (self.config.representation!='volume'
                                and self.config.surface_geometry_gradient_mode=='implicit')
            # A hard surface already has a Dirac termination distribution.
            # Penalizing the CDF training surrogate's width would conflict
            # with unit SDF gradients and the requested arbitrarily tight prior.
            prior = (outputs['rgb'].sum()*0 if physical_surface else
                     depth_interval_loss(outputs['prior_weights'],outputs['prior_mids'],outputs['prior_indices'],
                                         len(depth),depth,tolerance,valid,
                                         objective=self.config.depth_prior_objective))
            if self.config.representation != 'volume' and valid.any():
                distance = ((outputs['depth']-depth).abs()-tolerance).clamp_min(0)/tolerance.clamp_min(1e-6)
                prior = prior+F.smooth_l1_loss(distance[valid],torch.zeros_like(distance[valid]))
                signed=outputs['prior_sdf'][valid]/tolerance[valid].clamp_min(1e-6)
                prior = prior+F.smooth_l1_loss(signed,torch.zeros_like(signed))
                free=(tolerance-outputs['prior_free_sdf']).clamp_min(0)/tolerance.clamp_min(1e-6)
                free=free[valid.reshape(-1)]
                prior = prior+F.smooth_l1_loss(free,torch.zeros_like(free))
                if physical_surface:
                    brackets=outputs['prior_bracket_sdf']
                    margin=.01*tolerance
                    signs=torch.cat([margin-brackets[:,:1],margin+brackets[:,1:]],-1).clamp_min(0)/tolerance.clamp_min(1e-6)
                    signs=signs[valid.reshape(-1)]
                    prior=prior+F.smooth_l1_loss(signs,torch.zeros_like(signs))
                    prior=prior+(1-outputs['surface_hit'][valid]).mean()
            losses['trusted_depth_loss'] = self.config.depth_prior_loss_mult*strength*prior
        return losses
