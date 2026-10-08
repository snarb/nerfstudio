"""Attach measured camera-z depth to sampled train rays without pseudo depth."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
import json
import hashlib
from pathlib import Path
from typing import Type
import numpy as np
import torch
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager,VanillaDataManagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset


@dataclass
class LusterOpacityDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda:LusterOpacityDataManager)
    depth_prior_manifest: str | None = None
    train_camera_ids: tuple[int, ...] | None = None
    """Optional physical-camera allowlist for training sampling only."""


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

    def _next_camera_filtered_train(self,step):
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        if image_batch is not self._filtered_source_batch:
            self._filtered_image_batch = filter_camera_image_batch(image_batch,self._allowed_train_dataset_indices)
            self._filtered_source_batch = image_batch
        batch = self.train_pixel_sampler.sample(self._filtered_image_batch)
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
