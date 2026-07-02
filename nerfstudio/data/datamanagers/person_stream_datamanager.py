# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Streaming data manager that oversamples the person region during pixel sampling.

Subclasses :class:`ParallelDataManager` (the WINNER's fast ``load_from_disk`` setup). The only
change vs the leader is the train pixel sampler: a :class:`PersonWeightedPixelSampler` that draws
``person_frac`` of rays from inside the YOLO person masks and the rest uniformly. The sampler runs
inside the dataloader workers, so the masks (path + per-image stems) are attached to the
RayBatchStream's ``pixel_sampler_config`` BEFORE the worker iterator is created.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Optional, Type

from torch.utils.data import DataLoader

from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager, ParallelDataManagerConfig
from nerfstudio.data.person_weighted_pixel_sampler import PersonWeightedPixelSamplerConfig
from nerfstudio.data.utils.data_utils import identity_collate
from nerfstudio.data.utils.dataloaders import RayBatchStream, variable_res_collate
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class PersonStreamDataManagerConfig(ParallelDataManagerConfig):
    """Config for :class:`PersonStreamDataManager`."""

    _target: Type = field(default_factory=lambda: PersonStreamDataManager)
    pixel_sampler: PersonWeightedPixelSamplerConfig = field(default_factory=PersonWeightedPixelSamplerConfig)
    """Person-weighted pixel sampler (exposes ``--pipeline.datamanager.pixel-sampler.person-frac``)."""
    motion_map_dir: str = "person_masks"
    """Sub-directory (under the dataset path) holding ``person_masks.pt`` when no explicit path is given."""
    motion_map_path: Optional[Path] = None
    """Explicit path to the cached ``person_masks.pt`` (overrides ``motion_map_dir``)."""


class PersonStreamDataManager(ParallelDataManager, Generic[TDataset]):
    """Streaming data manager with person-oversampled pixel sampling."""

    config: PersonStreamDataManagerConfig

    def _resolve_mask_path(self) -> Optional[Path]:
        if self.config.motion_map_path is not None:
            return Path(self.config.motion_map_path)
        data = self.config.data if self.config.data is not None else self.config.dataparser.data
        if data is None:
            return None
        return Path(data) / self.config.motion_map_dir / "person_masks.pt"

    def setup_train(self):
        # Mirror ParallelDataManager.setup_train, but build the person-weighted pixel-sampler config
        # (mask path + per-image stems) and attach it to the RayBatchStream BEFORE the worker
        # iterator is created, so the (single) set of workers picks it up.
        ps_cfg = self.config.pixel_sampler
        path = self._resolve_mask_path()
        ps_cfg.person_masks_path = str(path) if path is not None else None
        ps_cfg.stems = [Path(fn).stem for fn in self.train_dataparser_outputs.image_filenames]
        ps_cfg.num_rays_per_batch = self.config.train_num_rays_per_batch
        if path is None or not Path(path).exists():
            CONSOLE.print(
                f"[yellow]PersonStreamDataManager: person masks not found at {path}; "
                "falling back to uniform sampling."
            )
        else:
            CONSOLE.print(f"PersonStreamDataManager: person_frac={ps_cfg.person_frac} masks={path}")

        self.train_raybatchstream = RayBatchStream(
            input_dataset=self.train_dataset,
            num_rays_per_batch=self.config.train_num_rays_per_batch,
            num_images_to_sample_from=(
                50
                if self.config.load_from_disk and self.config.train_num_images_to_sample_from == float("inf")
                else self.config.train_num_images_to_sample_from
            ),
            num_times_to_repeat_images=(
                10
                if self.config.load_from_disk and self.config.train_num_times_to_repeat_images == float("inf")
                else self.config.train_num_times_to_repeat_images
            ),
            device=self.device,
            collate_fn=variable_res_collate,
            load_from_disk=self.config.load_from_disk,
            custom_ray_processor=self.custom_ray_processor,
        )
        self.train_raybatchstream.pixel_sampler_config = ps_cfg
        self.train_ray_dataloader = DataLoader(
            self.train_raybatchstream,
            batch_size=1,
            num_workers=self.config.dataloader_num_workers,
            prefetch_factor=self.config.prefetch_factor,
            shuffle=False,
            collate_fn=identity_collate,
        )
        self.iter_train_raybundles = iter(self.train_ray_dataloader)
