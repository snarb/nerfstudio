"""
LookCloser (FA-NeRF) Pipeline.
Extends VanillaPipeline to handle periodic "Side-Channel" updates of the frequency grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.models.lookcloser import LookCloserModel
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class LookCloserPipelineConfig(VanillaPipelineConfig):
    """Configuration for LookCloser Pipeline."""

    _target: Type = field(default_factory=lambda: LookCloserPipeline)

    frequency_map_dir: str = "lookcloser_frequencies"
    """Name of the directory inside the data folder containing pre-computed frequency maps."""

    enable_frequency_grid: bool = True
    """Whether to load 2D maps and run periodic 3D frequency-grid updates."""

    grid_update_interval: int = 1024
    """Step interval for updating the 3D frequency grid using dense depth rendering."""

    grid_update_batch_size: int = 2048
    """Number of rays to sample for the grid update step."""

    frequency_patch_size: int = 32
    """Fallback patch size for legacy frequency maps without sidecar metadata."""

    frequency_stride: int = 32
    """Fallback frequency-map stride for legacy maps without sidecar metadata."""


class LookCloserPipeline(VanillaPipeline):
    """
    LookCloser Pipeline.

    In addition to the standard training loop, this pipeline performs a periodic
    "maintenance" step where it samples random patches from the training set,
    renders their depth using the current model state, and updates the 3D
    frequency grid (Part 2 of the LookCloser method).
    """

    config: LookCloserPipelineConfig
    model: LookCloserModel

    def __init__(
            self,
            config: LookCloserPipelineConfig,
            device: str,
            test_mode: Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        if not self.config.enable_frequency_grid and hasattr(self.model, "freq_grid"):
            self.model.freq_grid.enabled = False

        # Cache for frequency maps (Index -> Tensor)
        # We load them lazily or upfront. For simplicity/speed during training, we load upfront.
        self.cached_freq_maps: Dict[int, Tensor] = {}
        self.cached_freq_patch_sizes: Dict[int, int] = {}
        self.cached_freq_strides: Dict[int, int] = {}
        self.cached_freq_image_shapes: Dict[int, Tuple[int, int]] = {}
        if self.config.enable_frequency_grid:
            self._load_frequency_maps()

    def _read_frequency_metadata(self, map_path: Path) -> Optional[Dict]:
        metadata_path = map_path.with_suffix(".json")
        if not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        value_type = metadata.get("value_type")
        if value_type != "scalar_resolution":
            raise ValueError(
                f"Frequency map {map_path} has value_type={value_type!r}; "
                "LookCloser downstream expects scalar_resolution values, not level indices."
            )
        return metadata

    def _validate_frequency_schedule(self, map_path: Path, metadata: Dict) -> None:
        grid = self.model.freq_grid
        min_res = float(metadata.get("min_res", grid.min_res))
        max_res = float(metadata.get("max_res", grid.max_res))
        n_levels = int(metadata.get("n_levels", grid.num_levels))
        if (
            abs(min_res - float(grid.min_res)) > 1e-6
            or abs(max_res - float(grid.max_res)) > 1e-6
            or n_levels != int(grid.num_levels)
        ):
            raise ValueError(
                f"Frequency metadata for {map_path} does not match the model frequency grid: "
                f"metadata(min_res={min_res}, max_res={max_res}, n_levels={n_levels}) vs "
                f"grid(min_res={grid.min_res}, max_res={grid.max_res}, n_levels={grid.num_levels})."
            )

    @staticmethod
    def _expected_map_shape(image_shape: Tuple[int, int], patch_size: int, stride: int) -> Tuple[int, int]:
        image_h, image_w = image_shape
        if image_h < patch_size or image_w < patch_size:
            raise ValueError(f"Image shape {image_shape} is smaller than patch_size={patch_size}.")
        return (
            ((image_h - patch_size) // stride) + 1,
            ((image_w - patch_size) // stride) + 1,
        )

    def _load_frequency_maps(self):
        """Loads pre-computed 2D frequency maps from disk into CPU memory."""
        # Access data directory from the DataManager
        # Note: DataManager interface is slightly abstract, usually has get_datapath()
        # or we check config.datamanager.data

        # Try to find the data path
        data_path = None
        if hasattr(self.datamanager, "get_datapath"):
            data_path = self.datamanager.get_datapath()
        elif hasattr(self.datamanager.config, "data"):
            data_path = self.datamanager.config.data

        if data_path is None:
            CONSOLE.print(
                "[yellow]LookCloserPipeline: Could not determine data path. Grid updates might fail.[/yellow]")
            return

        freq_dir = data_path / self.config.frequency_map_dir
        if not freq_dir.exists():
            CONSOLE.print(
                f"[red]LookCloserPipeline: Frequency map directory not found at {freq_dir}. Please run preprocessing script first.[/red]")
            return

        CONSOLE.print(f"LookCloserPipeline: Loading frequency maps from {freq_dir}...")

        # We need to map image indices to filenames.
        # The dataset stores filenames.
        train_dataset = self.datamanager.train_dataset

        count = 0
        for idx in range(len(train_dataset)):
            # Get filename for this index
            # This depends on dataset structure, but usually dataset.image_filenames exists
            if hasattr(train_dataset, "image_filenames"):
                filepath = train_dataset.image_filenames[idx]
                stem = filepath.stem

                map_path = freq_dir / f"{stem}.pt"
                if map_path.exists():
                    # Load to CPU to save VRAM
                    freq_map = torch.load(map_path, map_location="cpu").float()
                    if freq_map.ndim != 2:
                        raise ValueError(f"Frequency map {map_path} must be a 2D tensor, got shape {tuple(freq_map.shape)}.")
                    if not torch.isfinite(freq_map).all() or torch.min(freq_map).item() <= 0.0:
                        raise ValueError(f"Frequency map {map_path} must contain finite positive scalar resolution values.")

                    metadata = self._read_frequency_metadata(map_path)
                    if metadata is not None:
                        self._validate_frequency_schedule(map_path, metadata)
                        self.cached_freq_patch_sizes[idx] = int(
                            metadata.get("patch_size", self.config.frequency_patch_size)
                        )
                        self.cached_freq_strides[idx] = int(
                            metadata.get("stride", metadata.get("patch_size", self.config.frequency_stride))
                        )
                        image_shape = metadata.get("image_shape")
                        if image_shape is not None:
                            image_shape_tuple = (int(image_shape[0]), int(image_shape[1]))
                            expected_shape = self._expected_map_shape(
                                image_shape_tuple,
                                self.cached_freq_patch_sizes[idx],
                                self.cached_freq_strides[idx],
                            )
                            if tuple(freq_map.shape) != expected_shape:
                                raise ValueError(
                                    f"Frequency map {map_path} shape {tuple(freq_map.shape)} does not match "
                                    f"metadata-derived patch grid {expected_shape} for image_shape={image_shape_tuple}, "
                                    f"patch_size={self.cached_freq_patch_sizes[idx]}, "
                                    f"stride={self.cached_freq_strides[idx]}."
                                )
                            self.cached_freq_image_shapes[idx] = image_shape_tuple
                    else:
                        if torch.max(freq_map).item() <= float(self.model.freq_grid.num_levels - 1):
                            raise ValueError(
                                f"Frequency map {map_path} has no metadata and looks like level-index data "
                                f"(max={torch.max(freq_map).item():.3f}). Expected scalar resolution values."
                            )
                        self.cached_freq_patch_sizes[idx] = int(self.config.frequency_patch_size)
                        self.cached_freq_strides[idx] = int(self.config.frequency_stride)
                    self.cached_freq_maps[idx] = freq_map
                    count += 1

        CONSOLE.print(f"LookCloserPipeline: Loaded {count} frequency maps.")

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """
        Standard training step + Periodic Grid Update.
        """
        # 1. Standard Training Step
        model_outputs, loss_dict, metrics_dict = super().get_train_loss_dict(step)

        # 2. Side-Channel Grid Update
        # "Every 1024 training steps... render depth... update voxel"
        if (
                self.config.grid_update_interval > 0
                and self.config.enable_frequency_grid
                and step % self.config.grid_update_interval == 0
                and step > 0
        ):
            self._update_frequency_grid(step)

        return model_outputs, loss_dict, metrics_dict

    @torch.no_grad()
    def _update_frequency_grid(self, step: int):
        """
        Performs the 'Runtime Update' logic from LookCloser Part 2.
        Samples random patch centers, renders depth, computes f3d, updates grid.
        """
        if not self.cached_freq_maps:
            return

        # --- 1. Sample Random Locations ---
        # We need specific (image_idx, y, x) tuples to look up f2d.
        num_samples = self.config.grid_update_batch_size
        dataset = self.datamanager.train_dataset
        num_images = len(dataset)

        # Randomly choose images
        # We assume dataset indices 0..N-1 correspond to cached_freq_maps keys
        available_indices = list(self.cached_freq_maps.keys())
        if not available_indices:
            return

        rand_img_indices = torch.tensor(available_indices)[
            torch.randint(0, len(available_indices), (num_samples,))
        ]

        # Randomly choose pixels (y, x)
        # We need image dimensions. Cameras object holds this.
        cameras = dataset.cameras
        H = cameras.height[rand_img_indices].squeeze(-1)  # (N,)
        W = cameras.width[rand_img_indices].squeeze(-1)  # (N,)

        rand_y = (torch.rand(num_samples) * H).long()
        rand_x = (torch.rand(num_samples) * W).long()

        # Clamp to be safe
        rand_y = torch.clamp(rand_y, 0, H - 1)
        rand_x = torch.clamp(rand_x, 0, W - 1)

        # --- 2. Retrieve f_2D ---
        # Since maps vary in size, we can't batch lookup easily.
        # We do a CPU loop or gather. Since maps are on CPU, loop is okay for 2048 items.
        f2d_values = []
        valid_mask = []

        for i in range(num_samples):
            img_idx = rand_img_indices[i].item()
            y, x = rand_y[i].item(), rand_x[i].item()

            # The freq map is patch-wise. Prefer preprocessing metadata over legacy fallback.
            # We must convert pixel coordinates to map coordinates.
            stride = self.cached_freq_strides.get(img_idx, int(self.config.frequency_stride))
            patch_size = self.cached_freq_patch_sizes.get(img_idx, int(self.config.frequency_patch_size))
            freq_map = self.cached_freq_maps[img_idx]
            covered_h = (freq_map.shape[0] - 1) * stride + patch_size
            covered_w = (freq_map.shape[1] - 1) * stride + patch_size
            y_for_map = min(y, covered_h - 1)
            x_for_map = min(x, covered_w - 1)
            map_y = min(y_for_map // stride, freq_map.shape[0] - 1)
            map_x = min(x_for_map // stride, freq_map.shape[1] - 1)

            f = freq_map[map_y, map_x]
            f2d_values.append(f)
            valid_mask.append(True)

        f2d_tensor = torch.tensor(f2d_values, dtype=torch.float32, device=self.device)

        # --- 3. Generate Rays ---
        # Generate rays for these specific pixels
        # coord: (y, x)
        coords = torch.stack([rand_y, rand_x], dim=-1).to(self.device)  # (N, 2)
        camera_indices = rand_img_indices.to(self.device).unsqueeze(-1)  # (N, 1)

        ray_bundle = cameras.generate_rays(
            camera_indices=camera_indices,
            coords=coords,
            keep_shape=False  # Flatten
        )
        ray_bundle = ray_bundle.to(self.device)

        # --- 4. Render Depth (and only depth needed) ---
        # We call model with standard forward, but we can optimize?
        # The model's get_outputs usually renders everything.
        outputs = self.model(ray_bundle)
        depth = outputs["depth"].squeeze(-1)  # (N,)

        # --- 5. Compute f_3D and Update ---
        # f3d = f2d * (focal / depth)

        # Get focals for these rays
        # (fx + fy)/2
        fx = cameras.fx[camera_indices.squeeze()].to(self.device).squeeze()
        fy = cameras.fy[camera_indices.squeeze()].to(self.device).squeeze()
        focals = (fx + fy) / 2.0

        # Get positions (surface intersection)
        # pos = o + d * depth
        positions = ray_bundle.origins + ray_bundle.directions * depth.unsqueeze(-1)

        # Call model's grid update
        self.model.freq_grid.update_step(
            step=step,
            positions=positions,
            rendered_depth=depth,
            focals=focals,
            patch_f2d=f2d_tensor
        )
