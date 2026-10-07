"""Explicit train-normalized, mask-free Luster parser for opacity experiments."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from typing import Type
import torch
from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio,NerfstudioDataParserConfig
from nerfstudio.data.scene_box import SceneBox


@dataclass
class LusterOpacityDataParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda:LusterOpacityDataParser)
    orientation_method: str = 'none'
    center_method: str = 'none'
    auto_scale_poses: bool = False
    downscale_factor: int | None = 1
    load_3D_points: bool = False
    eval_mode: str = 'filename'


class LusterOpacityDataParser(Nerfstudio):
    def _generate_dataparser_outputs(self,split='train'):
        meta = json.loads((self.config.data/'transforms.json').read_text())
        if any('mask_path' in r or 'depth_file_path' in r for r in meta['frames']):
            raise ValueError('This campaign accepts raw RGB and explicit train-only prior manifests only')
        if self.config.auto_scale_poses or self.config.center_method!='none' or self.config.orientation_method!='none':
            raise ValueError('Dataset already has train-only normalization; do not normalize again')
        if self.config.scale_factor!=1 or self.config.downscale_factor!=1:
            raise ValueError('Calibrated HD pixels/coordinates must not be rescaled by the dataparser')
        output = super()._generate_dataparser_outputs(split)
        bounds = torch.tensor(meta['opacity_aabb'],dtype=torch.float32)
        if bounds.shape!=(2,3) or not torch.isfinite(bounds).all() or not (bounds[1]>bounds[0]).all():
            raise ValueError('Invalid explicit scene bounds')
        output.scene_box = SceneBox(aabb=bounds)
        output.metadata['opacity_source_data'] = str(self.config.data.resolve())
        return output
