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
Person-weighted pixel sampler for the STREAMING (RayBatchStream / ParallelDataManager) path.

Goal (no frequency, no FAS): sample a fixed fraction ``person_frac`` of the rays from inside
the person region (YOLO person masks), and the rest uniformly over the whole image. This is the
"uniform + oversample person" idea on the WINNER's fast disk-streaming setup, so it avoids the
VanillaDataManager RAM/setup penalty that hurt the earlier FAS+motion experiments.

Efficiency: the cached person masks (a dict ``{image_stem: float16 [H/4, W/4]}``) are loaded ONCE
per dataloader worker (lazily, on first sample). For each rotating batch of images the worker
builds a flat pool of person mask-cells (cheap nonzero over ~50 small masks, refreshed only when
the image set changes), then samples with two vectorized ``torch.randint`` calls. No per-ray Python
loops; sampling cost is negligible vs ray marching, so train throughput matches the leader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type, Union

import torch
from jaxtyping import Int
from torch import Tensor

from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig


@dataclass
class PersonWeightedPixelSamplerConfig(PixelSamplerConfig):
    """Config for :class:`PersonWeightedPixelSampler`."""

    _target: Type = field(default_factory=lambda: PersonWeightedPixelSampler)
    person_frac: float = 0.3
    """Fraction of rays drawn from inside the person region; the rest are uniform over the image."""
    mask_downsample: int = 4
    """Integer downsample factor from full-res image pixels to mask cells (image is 4x the mask)."""
    person_masks_path: Optional[str] = None
    """Path to the cached ``person_masks.pt`` ({stem: float16 [H/4, W/4]}). Set by the datamanager."""
    stems: Optional[List[str]] = None
    """Global-image-index -> filename stem, used to look up each image's mask. Set by the datamanager."""


class PersonWeightedPixelSampler(PixelSampler):
    """Streaming pixel sampler that oversamples the person region.

    Falls back to uniform sampling for any image without a mask (and for the person quota when no
    person pixels are present in the current image batch), so it can never crash on missing masks.
    """

    config: PersonWeightedPixelSamplerConfig

    def __init__(self, config: PersonWeightedPixelSamplerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._masks_loaded = False
        self._masks: dict = {}
        self._cur_image_idx: Optional[Tensor] = None
        self._pool_sig: Optional[tuple] = None
        self._pool: Optional[Tuple[Tensor, Tensor, Tensor]] = None

    def _ensure_masks(self) -> None:
        if self._masks_loaded:
            return
        self._masks_loaded = True
        path = self.config.person_masks_path
        if path is None:
            return
        try:
            raw = torch.load(path, map_location="cpu")
        except FileNotFoundError:
            return
        # Store as bool cell-occupancy on CPU; nonzero() over these is cheap.
        self._masks = {str(k): (v > 0) for k, v in raw.items()}

    def _build_pool(self, image_idx_cpu: Tensor) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """Flatten the person mask-cells of the current image batch into (local_c, row, col) pools."""
        self._ensure_masks()
        stems = self.config.stems
        if not self._masks or stems is None:
            return None
        cs: List[Tensor] = []
        rs: List[Tensor] = []
        cols: List[Tensor] = []
        for local_c, g in enumerate(image_idx_cpu.tolist()):
            stem = stems[g] if 0 <= g < len(stems) else None
            mask = self._masks.get(stem) if stem is not None else None
            if mask is None:
                continue
            nz = mask.nonzero(as_tuple=False)  # [P_i, 2] -> (cell_row, cell_col)
            if nz.numel() == 0:
                continue
            cs.append(torch.full((nz.shape[0],), local_c, dtype=torch.long))
            rs.append(nz[:, 0].long())
            cols.append(nz[:, 1].long())
        if not cs:
            return None
        return torch.cat(cs), torch.cat(rs), torch.cat(cols)

    def collate_image_dataset_batch(self, batch, num_rays_per_batch: int, keep_full_image: bool = False):
        # Stash the (local-order) global image indices so sample_method can look up masks.
        self._cur_image_idx = batch["image_idx"]
        return super().collate_image_dataset_batch(batch, num_rays_per_batch, keep_full_image)

    def _uniform(self, n: int, num_images: int, h: int, w: int, device) -> Tensor:
        return (
            torch.rand((n, 3), device=device) * torch.tensor([num_images, h, w], device=device)
        ).long()

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        frac = float(self.config.person_frac)
        n_person = int(round(frac * batch_size))
        n_uniform = batch_size - n_person

        if n_person <= 0 or self._cur_image_idx is None:
            return self._uniform(batch_size, num_images, image_height, image_width, device)

        image_idx_cpu = self._cur_image_idx.detach().cpu().long()
        sig = tuple(image_idx_cpu.tolist())
        if sig != self._pool_sig:
            self._pool = self._build_pool(image_idx_cpu)
            self._pool_sig = sig

        uni = self._uniform(n_uniform, num_images, image_height, image_width, device)
        if self._pool is None:
            # No person pixels available in this batch -> fill the quota uniformly.
            extra = self._uniform(n_person, num_images, image_height, image_width, device)
            return torch.cat([uni, extra], dim=0)

        pc, pr, pcol = self._pool
        P = pc.shape[0]
        ds = int(self.config.mask_downsample)
        sel = torch.randint(0, P, (n_person,))
        off_r = torch.randint(0, ds, (n_person,))
        off_c = torch.randint(0, ds, (n_person,))
        rows = (pr[sel] * ds + off_r).clamp_(max=image_height - 1)
        cols = (pcol[sel] * ds + off_c).clamp_(max=image_width - 1)
        person = torch.stack([pc[sel], rows, cols], dim=-1).to(device).long()
        return torch.cat([uni, person], dim=0)
