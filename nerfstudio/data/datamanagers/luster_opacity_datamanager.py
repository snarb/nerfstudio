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


class LusterOpacityDataManager(VanillaDataManager):
    @cached_property
    def dataset_type(self):
        return InputDataset

    def next_train(self,step):
        ray_bundle,batch = super().next_train(step)
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
