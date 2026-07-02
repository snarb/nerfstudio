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
Parallel (streaming) data manager that attaches a per-ray person label.

Subclasses :class:`ParallelDataManager` (the WINNER's streaming setup: load_from_disk). The
person masks (a single cached dict {image_stem: float16 [H/4, W/4]}) are loaded ONCE in the
MAIN process and NEVER pickled into the dataloader workers. ``next_train`` looks up the mask
value per sampled ray using ``batch['indices']`` (img_idx, row, col) and attaches
``batch['is_person']`` ([N, 1]) on device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generic, Optional, Tuple, Type

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager, ParallelDataManagerConfig
from nerfstudio.utils.rich_utils import CONSOLE

# Downsample factor between full-res images (1080x1920) and the cached masks (270x480).
MASK_DOWNSAMPLE = 4


@dataclass
class TemporalDecompDataManagerConfig(ParallelDataManagerConfig):
    """Config for :class:`TemporalDecompDataManager`."""

    _target: Type = field(default_factory=lambda: TemporalDecompDataManager)
    motion_map_dir: str = "person_masks"
    """Sub-directory (under the dataset path) holding ``person_masks.pt`` when no explicit
    path is given."""
    motion_map_path: Optional[Path] = None
    """Explicit path to the cached ``person_masks.pt`` (overrides ``motion_map_dir``)."""
    mask_downsample: int = MASK_DOWNSAMPLE
    """Integer downsample factor from image pixels to mask cells (image is 4x the mask)."""


class TemporalDecompDataManager(ParallelDataManager, Generic[TDataset]):
    """Streaming data manager that attaches per-ray person labels.

    Missing-mask policy: if a sampled ray's image stem (or mask cell) is unavailable, the ray
    is labeled ``is_person = 1`` ("unknown -> do NOT penalize"). This is the safe choice: we
    never wrongly suppress dynamics on rays we cannot label.
    """

    config: TemporalDecompDataManagerConfig

    def __init__(self, config: TemporalDecompDataManagerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._person_masks: Dict[str, torch.Tensor] = {}
        self._idx_to_stem: Dict[int, str] = {}
        self._load_person_masks()
        self._build_idx_to_stem()

    def _resolve_mask_path(self) -> Optional[Path]:
        if self.config.motion_map_path is not None:
            return Path(self.config.motion_map_path)
        data = self.config.data if self.config.data is not None else self.config.dataparser.data
        if data is None:
            return None
        return Path(data) / self.config.motion_map_dir / "person_masks.pt"

    def _load_person_masks(self) -> None:
        path = self._resolve_mask_path()
        if path is None or not Path(path).exists():
            CONSOLE.print(f"[yellow]TemporalDecompDataManager: person masks not found at {path}; "
                          "all rays will be labeled is_person=1 (no dynamic suppression).")
            self._person_masks = {}
            return
        raw = torch.load(path, map_location="cpu")
        # Keep masks on CPU in the MAIN process; cast to float32 for lookups.
        self._person_masks = {str(k): v.float() for k, v in raw.items()}
        CONSOLE.print(f"TemporalDecompDataManager: loaded {len(self._person_masks)} person masks from {path}")

    def _build_idx_to_stem(self) -> None:
        filenames = self.train_dataparser_outputs.image_filenames
        self._idx_to_stem = {i: Path(fn).stem for i, fn in enumerate(filenames)}

    def _lookup_is_person(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [N, 3] = (img_idx, row, col) in FULL-res pixel coords. Returns [N, 1] float."""
        ds = self.config.mask_downsample
        n = indices.shape[0]
        idx_cpu = indices.detach().cpu().long()
        out = torch.ones(n, 1, dtype=torch.float32)  # default = 1 (unknown -> no penalty)
        for i in range(n):
            img_idx = int(idx_cpu[i, 0].item())
            row = int(idx_cpu[i, 1].item())
            col = int(idx_cpu[i, 2].item())
            stem = self._idx_to_stem.get(img_idx)
            if stem is None:
                continue
            mask = self._person_masks.get(stem)
            if mask is None:
                continue
            mr = min(row // ds, mask.shape[0] - 1)
            mc = min(col // ds, mask.shape[1] - 1)
            out[i, 0] = float(mask[mr, mc].item())
        return out

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_train(step)
        if "indices" in batch:
            is_person = self._lookup_is_person(batch["indices"])
            batch["is_person"] = is_person.to(self.device)
        return ray_bundle, batch
