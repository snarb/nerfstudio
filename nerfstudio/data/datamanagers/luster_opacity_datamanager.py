"""Attach measured camera-z depth to sampled train rays without pseudo depth."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
import json
import hashlib
import math
from pathlib import Path
from typing import Type
import numpy as np
import torch
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager,VanillaDataManagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.lookcloser_rng import fork_seeded_rng


@dataclass
class LusterOpacityDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda:LusterOpacityDataManager)
    depth_prior_manifest: str | None = None
    train_camera_ids: tuple[int, ...] | None = None
    """Optional physical-camera allowlist for training sampling only."""
    cord_rgb_fit_path: str | None = None
    cord_rgb_fit_sha256: str | None = None
    cord_rgb_camera_ids: tuple[int, ...] | None = None
    cord_rgb_rays_per_batch: int = 0
    cord_rgb_jitter_px: float = 6.
    cord_rgb_seed: int = 42
    """Opt-in RGB sampling near a train-only fitted cord line; no masks."""


def filter_camera_image_batch(batch, allowed_dataset_indices):
    """Keep global dataset indices while presenting only allowed RGB images."""
    ids = batch['image_idx']
    allowed = torch.as_tensor(allowed_dataset_indices, device=ids.device, dtype=ids.dtype)
    keep = torch.nonzero(torch.isin(ids, allowed)).flatten()
    if not len(keep):
        raise ValueError('Cached image batch contains no allowed training cameras')
    positions = keep.cpu().tolist()
    result = {}
    for key, values in batch.items():
        if key == 'image_idx':
            result[key] = values[keep]
        elif values is None:
            result[key] = None
        elif isinstance(values, (list, tuple)) and len(values) == len(ids):
            result[key] = [values[i] for i in positions]
        elif isinstance(values, torch.Tensor) and values.shape[0] == len(ids):
            # Force the variable-resolution FAS path. Its subset lookup maps
            # global frequency-bucket IDs correctly back to these local images.
            result[key] = [values[i] for i in positions]
        else:
            raise ValueError(f'Unsupported cached image-batch entry: {key}')
    return result


class LusterOpacityDataManager(VanillaDataManager):
    @cached_property
    def dataset_type(self):
        return InputDataset

    def setup_train(self):
        super().setup_train()
        ids = self.config.train_camera_ids
        if ids is None:
            if self.config.cord_rgb_rays_per_batch:
                raise ValueError('Cord RGB sampling requires a training-camera allowlist')
            return
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(i,int) or i<1 for i in ids):
            raise ValueError('Require unique positive physical camera IDs')
        if self.config.cpu_fas_prefetch or not self.train_image_dataloader.cache_all_images:
            raise ValueError('Camera filtering requires synchronous sampling and caching all images')
        lookup = {int(path.stem.split('_')[1]):i for i,path in enumerate(self.train_dataset.image_filenames)}
        if set(ids)-lookup.keys():
            raise ValueError('Camera allowlist includes an unknown or held-out camera')
        self._allowed_train_dataset_indices = tuple(lookup[i] for i in ids)
        self._filtered_source_batch = None
        self._filtered_image_batch = None
        if self.config.cord_rgb_rays_per_batch:
            self._setup_cord_rgb_sampling()

    def _setup_cord_rgb_sampling(self):
        count = self.config.cord_rgb_rays_per_batch
        if (count < 1 or count >= self.train_pixel_sampler.num_rays_per_batch
                or not math.isfinite(self.config.cord_rgb_jitter_px)
                or self.config.cord_rgb_jitter_px < 0
                or not self.config.cord_rgb_fit_path or not self.config.cord_rgb_fit_sha256):
            raise ValueError('Invalid cord RGB sampling controls')
        fit_path = Path(self.config.cord_rgb_fit_path)
        if hashlib.sha256(fit_path.read_bytes()).hexdigest() != self.config.cord_rgb_fit_sha256:
            raise ValueError('Cord RGB fit changed since request creation')
        fit = json.loads(fit_path.read_text())
        source = Path(self.train_dataset.metadata.get('opacity_source_data',self.config.dataparser.data))
        transforms_path = source/'transforms.json'
        if (fit.get('train_only') is not True
                or fit.get('coordinate_system') != 'train_normalized_opengl'
                or fit.get('transforms_sha256') != hashlib.sha256(transforms_path.read_bytes()).hexdigest()):
            raise ValueError('Cord RGB fit must match train-only scene calibration')
        transforms = json.loads(transforms_path.read_text())
        common = set.intersection(*(set(fit['fit_camera_ids'][key]) for key in ('junction','first_bead')))
        requested = (set(self.config.cord_rgb_camera_ids) if self.config.cord_rgb_camera_ids is not None
                     else common & set(self.config.train_camera_ids))
        if (not requested or requested - common or requested - set(self.config.train_camera_ids)
                or fit['validation_camera_id'] in requested):
            raise ValueError('Cord RGB cameras must be annotated training cameras only')
        frames = {int(frame['camera_id']):frame for frame in transforms['frames']}
        lookup = {int(path.stem.split('_')[1]):i for i,path in enumerate(self.train_dataset.image_filenames)}
        train_names = {path.name for path in self.train_dataset.image_filenames}
        endpoints = torch.tensor([fit['fits'][key]['normalized_world_xyz']
                                  for key in ('junction','first_bead')],dtype=torch.float64)
        lines, indices, dimensions = [], [], []
        for camera_id in sorted(requested):
            frame=frames[camera_id]
            if Path(frame['file_path']).name not in train_names:
                raise ValueError('Cord RGB camera is not in the training split')
            pose=torch.tensor(frame['transform_matrix'],dtype=torch.float64)
            local=(endpoints-pose[:3,3])@pose[:3,:3]
            forward=-local[:,2]
            xy=torch.stack([frame['fl_x']*local[:,0]/forward+frame['cx'],
                            -frame['fl_y']*local[:,1]/forward+frame['cy']],dim=-1)
            dataset_index=lookup[camera_id]
            height=int(self.train_dataset.cameras.height[dataset_index])
            width=int(self.train_dataset.cameras.width[dataset_index])
            if (not torch.isfinite(xy).all() or not (forward>0).all()
                    or (xy[1]-xy[0]).norm() < 1
                    or not ((xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)).all()):
                raise ValueError(f'Cord RGB line is outside camera {camera_id}')
            lines.append(xy.float())
            indices.append(dataset_index)
            dimensions.append([width,height])
        self._cord_rgb_lines=torch.stack(lines)
        self._cord_rgb_dataset_indices=torch.tensor(indices,dtype=torch.long)
        self._cord_rgb_dimensions=torch.tensor(dimensions,dtype=torch.long)

    def _replace_cord_rgb_rays(self, step, batch, image_batch):
        count=self.config.cord_rgb_rays_per_batch
        device=batch['indices'].device
        lines=self._cord_rgb_lines.to(device)
        global_indices=self._cord_rgb_dataset_indices.to(device)
        dimensions=self._cord_rgb_dimensions.to(device)
        with fork_seeded_rng(self.config.cord_rgb_seed,'cord_rgb',step,device):
            selection=torch.randint(len(lines),(count,),device=device)
            fraction=.08+.84*torch.rand(count,1,device=device)
            transverse=(2*torch.rand(count,1,device=device)-1)*self.config.cord_rgb_jitter_px
        chosen=lines[selection]
        axis=chosen[:,1]-chosen[:,0]
        perpendicular=torch.stack([-axis[:,1],axis[:,0]],dim=-1)/axis.norm(dim=-1,keepdim=True)
        xy=(chosen[:,0]+fraction*axis+transverse*perpendicular).round().long()
        limits=dimensions[selection]-1
        xy=torch.minimum(xy.clamp_min(0),limits)
        indices=torch.stack([global_indices[selection],xy[:,1],xy[:,0]],dim=-1)
        cached={int(value):i for i,value in enumerate(image_batch['image_idx'].tolist())}
        color=torch.empty((count,3),device=batch['image'].device,dtype=batch['image'].dtype)
        for global_index in global_indices.tolist():
            positions=torch.nonzero(indices[:,0]==global_index).flatten()
            if not len(positions):
                continue
            image=image_batch['image'][cached[global_index]]
            local=indices[positions].to(image.device)
            color[positions.to(color.device)]=image[local[:,1],local[:,2],:3].to(color)
        batch['indices'][-count:]=indices.to(batch['indices'].device)
        batch['image'][-count:]=color

    def _next_camera_filtered_train(self,step):
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        if image_batch is not self._filtered_source_batch:
            self._filtered_image_batch = filter_camera_image_batch(image_batch,self._allowed_train_dataset_indices)
            self._filtered_source_batch = image_batch
        batch = self.train_pixel_sampler.sample(self._filtered_image_batch)
        if self.config.cord_rgb_rays_per_batch:
            self._replace_cord_rgb_rays(step,batch,self._filtered_image_batch)
        allowed = torch.tensor(self._allowed_train_dataset_indices,device=batch['indices'].device)
        if not torch.isin(batch['indices'][:,0],allowed).all():
            raise ValueError('Pixel sampler leaked an excluded camera')
        return self.train_ray_generator(batch['indices']),batch

    def next_train(self,step):
        ray_bundle,batch = (super().next_train(step) if self.config.train_camera_ids is None else
                            self._next_camera_filtered_train(step))
        manifest_path = self.config.depth_prior_manifest
        if manifest_path is None:
            return ray_bundle,batch
        if not hasattr(self,'_depth_prior_arrays'):
            self._depth_prior_arrays = {}
            self._depth_manifest = json.loads(Path(manifest_path).read_text())
            if self._depth_manifest['train_only'] is not True:
                raise ValueError('Depth prior must use train-only observations')
            train_names = {p.name for p in self.train_dataset.image_filenames}
            if set(self._depth_manifest['maps'])-train_names:
                raise ValueError('Depth manifest includes held-out images')
            source=Path(self.train_dataset.metadata.get('opacity_source_data',self.config.dataparser.data))
            expected_hash=hashlib.sha256((source/'transforms.json').read_bytes()).hexdigest()
            if self._depth_manifest['transforms_sha256']!=expected_hash:
                raise ValueError('Depth manifest calibration/split differs from RGB')
        indices = batch['indices'].detach().cpu().numpy()
        values = np.zeros((len(indices),3),dtype=np.float32)
        for camera in np.unique(indices[:,0]):
            name = self.train_dataset.image_filenames[int(camera)].name
            row = self._depth_manifest['maps'].get(name)
            if row is None:
                continue
            if name not in self._depth_prior_arrays:
                path = Path(manifest_path).parent/row['path']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                    raise ValueError(f'Depth prior file hash mismatch: {name}')
                with np.load(path,allow_pickle=False) as loaded:
                    arrays = {k:loaded[k] for k in ['depth','uncertainty','valid']}
                expected = (int(self.train_dataset.cameras.height[int(camera)]),int(self.train_dataset.cameras.width[int(camera)]))
                if any(v.shape!=expected for v in arrays.values()):
                    raise ValueError(f'Depth shape mismatch: {name}')
                self._depth_prior_arrays[name] = arrays
            arrays = self._depth_prior_arrays[name]
            selected = indices[:,0]==camera
            y,x = indices[selected,1],indices[selected,2]
            values[selected] = np.stack([arrays['depth'][y,x],arrays['uncertainty'][y,x],arrays['valid'][y,x]],-1)
        tensor = torch.from_numpy(values).to(ray_bundle.origins.device)
        directions_norm = ray_bundle.metadata['directions_norm']
        scale = float(self._depth_manifest['normalization_scale'])
        batch['depth_prior'] = tensor[:,:1]*directions_norm*scale
        batch['depth_uncertainty'] = tensor[:,1:2]*directions_norm*scale
        batch['depth_valid'] = (tensor[:,2:3]>0)&torch.isfinite(batch['depth_prior'])&(batch['depth_prior']>0)
        points=ray_bundle.origins+ray_bundle.directions*batch['depth_prior']
        aabb=self.train_dataset.scene_box.aabb.to(points)
        batch['depth_valid'] &= ((points>=aabb[0])&(points<=aabb[1])).all(-1,keepdim=True)
        ray_bundle.metadata.update(depth_prior=batch['depth_prior'],depth_uncertainty=batch['depth_uncertainty'],
                                   depth_valid=batch['depth_valid'])
        return ray_bundle,batch
