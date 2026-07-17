# nerfstudio/data/pixel_samplers/lookcloser_pixel_sampler.py

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import Dataset

from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass(frozen=True)
class LookCloserFASPrefetchSnapshot:
    """Immutable CPU inputs and scalar policy for private-generator FAS."""

    image: Tensor
    image_idx: Tensor
    buckets: Tuple[Tensor, ...]
    probs: np.ndarray
    image_heights: Tensor
    image_widths: Tensor
    num_levels: int
    num_rays_per_batch: int
    patch_size: int
    patch_stride: int
    fas_strength: float
    fas_warmup_steps: int
    fas_ramp_steps: int
    fas_decay_start_steps: int
    fas_decay_steps: int
    fas_patch_group_size: int
    config_signature: Tuple[object, ...]
    data_version: int
    image_order: Tuple[int, ...]

    def _active_fas_strength(self, sample_count: int) -> float:
        target = float(np.clip(self.fas_strength, 0.0, 1.0))
        if sample_count < self.fas_warmup_steps:
            return 0.0
        if self.fas_ramp_steps <= 0:
            strength = target
        else:
            ramp_position = min(
                max(sample_count - self.fas_warmup_steps, 0) / float(self.fas_ramp_steps),
                1.0,
            )
            strength = target * ramp_position
        if self.fas_decay_start_steps >= 0 and sample_count >= self.fas_decay_start_steps:
            if self.fas_decay_steps <= 0:
                return 0.0
            decay_position = min(
                (sample_count - self.fas_decay_start_steps) / float(self.fas_decay_steps),
                1.0,
            )
            strength *= 1.0 - decay_position
        return strength

    def sample(self, generator: torch.Generator, sample_count: int) -> Dict:
        """Pure CPU equivalent of the supported synchronous FAS path."""

        num_images, image_height, image_width, _ = self.image.shape
        batch_size = self.num_rays_per_batch
        fas_batch_size = int(round(batch_size * self._active_fas_strength(sample_count)))
        uniform_batch_size = batch_size - fas_batch_size
        indices_list: List[Tensor] = []
        bounds = torch.tensor([num_images, image_height, image_width])
        if uniform_batch_size > 0:
            uniform = torch.rand((uniform_batch_size, 3), generator=generator) * bounds
            indices_list.append(uniform.long())

        if fas_batch_size <= 0:
            indices = indices_list[0]
            c, y, x = (part.flatten() for part in torch.split(indices, 1, dim=-1))
            collated_batch = {"image": self.image[c, y, x]}
            indices[:, 0] = self.image_idx[c]
            collated_batch["indices"] = indices
            return collated_batch

        if fas_batch_size > 0:
            expected_counts = self.probs * fas_batch_size
            counts = np.floor(expected_counts).astype(int)
            diff = int(fas_batch_size - counts.sum())
            if diff > 0:
                remainders = expected_counts - counts
                for level in np.argsort(-remainders)[:diff]:
                    counts[level] += 1
            elif diff < 0:
                remainders = expected_counts - counts
                for level in np.argsort(remainders)[: -diff]:
                    if counts[level] > 0:
                        counts[level] -= 1

            for level in range(self.num_levels):
                n_samples = int(counts[level])
                if n_samples == 0:
                    continue
                bucket = self.buckets[level]
                if bucket.shape[0] == 0:
                    fallback = torch.rand((n_samples, 3), generator=generator) * bounds
                    indices_list.append(fallback.long())
                    continue

                group_size = max(self.fas_patch_group_size, 1)
                if group_size == 1:
                    rand_idx = torch.randint(0, bucket.shape[0], (n_samples,), generator=generator)
                    selected_patches = bucket[rand_idx].long()
                    y_off = torch.randint(0, self.patch_size, (n_samples,), generator=generator)
                    x_off = torch.randint(0, self.patch_size, (n_samples,), generator=generator)
                else:
                    patches_needed = int(np.ceil(n_samples / float(group_size)))
                    rand_idx = torch.randint(0, bucket.shape[0], (patches_needed,), generator=generator)
                    selected_cells = bucket[rand_idx].long()
                    selected_patches = selected_cells.repeat_interleave(group_size, dim=0)[:n_samples]
                    grid_side = int(np.ceil(np.sqrt(group_size)))
                    local_ids = torch.arange(group_size).repeat(patches_needed)[:n_samples]
                    local_y = local_ids // grid_side
                    local_x = local_ids % grid_side
                    sub_h = max(self.patch_size // grid_side, 1)
                    sub_w = max(self.patch_size // grid_side, 1)
                    y_off = local_y * sub_h + torch.randint(
                        0, sub_h, (n_samples,), generator=generator
                    )
                    x_off = local_x * sub_w + torch.randint(
                        0, sub_w, (n_samples,), generator=generator
                    )
                    y_off = torch.clamp(y_off, 0, self.patch_size - 1)
                    x_off = torch.clamp(x_off, 0, self.patch_size - 1)

                img_idx = selected_patches[:, 0]
                y_coord = selected_patches[:, 1] * self.patch_stride + y_off
                x_coord = selected_patches[:, 2] * self.patch_stride + x_off
                valid = (img_idx >= 0) & (img_idx < self.image_heights.shape[0])
                safe_indices = img_idx.clamp(0, self.image_heights.shape[0] - 1)
                heights = torch.where(
                    valid,
                    self.image_heights[safe_indices],
                    torch.full_like(img_idx, image_height),
                )
                widths = torch.where(
                    valid,
                    self.image_widths[safe_indices],
                    torch.full_like(img_idx, image_width),
                )
                y_coord = torch.minimum(torch.clamp_min(y_coord, 0), heights - 1)
                x_coord = torch.minimum(torch.clamp_min(x_coord, 0), widths - 1)
                indices_list.append(torch.stack([img_idx, y_coord, x_coord], dim=1))

        all_indices = torch.cat(indices_list, dim=0)
        shuffle_mask = torch.randperm(all_indices.shape[0], generator=generator)
        indices = all_indices[shuffle_mask]
        c, y, x = (part.flatten() for part in torch.split(indices, 1, dim=-1))
        collated_batch = {"image": self.image[c, y, x]}
        indices[:, 0] = self.image_idx[c]
        collated_batch["indices"] = indices
        return collated_batch


@dataclass
class LookCloserPixelSamplerConfig(PixelSamplerConfig):
    """Configuration for the LookCloser Frequency-Averaged Pixel Sampler."""

    _target: Type = field(default_factory=lambda: LookCloserPixelSampler)

    frequency_map_dir: str = "lookcloser_frequencies"
    """Name of the directory inside the data_dir where pre-computed frequency maps are stored."""

    enable_fas: bool = True
    """Whether to use Frequency-Averaged Sampling; disabled falls back to uniform PixelSampler behavior."""

    num_levels: int = 16
    """Number of frequency levels to bucket pixels into."""

    min_res: float = 16.0
    """Minimum resolution used during pre-processing (base of geometric progression)."""

    max_res: float = 2048.0
    """Fallback maximum resolution for legacy maps without preprocessing metadata."""

    sampling_ramp_start: float = 1.0
    """Start of the linear probability ramp for sampling."""

    sampling_ramp_end: float = 3.0
    """End of the linear probability ramp for sampling (high-freq gets more samples)."""

    fas_strength: float = 1.0
    """Fraction of each batch sampled with FAS. Remaining rays are sampled uniformly."""

    fas_warmup_steps: int = 0
    """Number of initial sampler calls that use uniform sampling before enabling FAS."""

    fas_ramp_steps: int = 0
    """Number of sampler calls over which FAS strength ramps from zero to fas_strength."""

    fas_decay_start_steps: int = -1
    """Sampler step where FAS strength starts decaying back to uniform; negative disables decay."""

    fas_decay_steps: int = 0
    """Number of sampler calls over which FAS strength decays to zero after fas_decay_start_steps."""

    fas_level_count_alpha: float = 0.0
    """Blend frequency-ramp weights with observed bucket population counts; 0 preserves ramp-only FAS."""

    fas_patch_group_size: int = 1
    """Number of locally distributed pixels to sample per selected frequency-map patch; 1 preserves random offsets."""

    fas_max_sampling_level: int = -1
    """Optional maximum frequency level for FAS buckets; negative preserves all preprocessed levels."""

    fas_consolidate_h2d: bool = False
    """Consolidate per-level selected-cell CPU-to-CUDA copies into one exact transfer."""

    debug_mode: bool = False
    """If true, prints sampling stats."""

    patch_size: int = 32
    """Fallback patch size for legacy frequency maps without sidecar metadata."""

    stride: int = 32
    """Fallback frequency-map stride for legacy maps without sidecar metadata."""


class LookCloserPixelSampler(PixelSampler):
    """
    Frequency-Averaged Sampler (FAS) for LookCloser.

    This sampler buckets all pixels in the training dataset based on their pre-computed
    2D frequency complexity and samples a batch ensuring a specific ratio (default 1:3)
    between low-frequency and high-frequency regions.
    """

    config: LookCloserPixelSamplerConfig

    def __init__(self, config: LookCloserPixelSamplerConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.dataset = kwargs.get("dataset")
        self.buckets: Dict[int, Tensor] = {}
        self.samples_per_level: np.ndarray = np.zeros(self.config.num_levels, dtype=int)
        self.level_counts: np.ndarray = np.zeros(self.config.num_levels, dtype=np.float64)

        # We need to initialize the buckets.
        # Since PixelSampler is initialized with the DataManager, we assume the dataset
        # is available or passed in the first sample call?
        # Standard Nerfstudio architecture doesn't pass dataset to __init__.
        # We will lazy-load on the first call to `sample`.
        self.is_initialized = False
        self.patch_size = int(self.config.patch_size)
        self.patch_stride = int(self.config.stride)
        self.image_shapes: Dict[int, Tuple[int, int]] = {}
        self._image_shape_lut_cache: Dict[Tuple[str, int, int, int], Tuple[Tensor, Tensor]] = {}
        self.sample_count = 0
        self.current_fas_strength = 1.0
        self._prefetch_data_version = 0

    def _image_shapes_for_indices(
        self,
        img_idx: Tensor,
        num_images: int,
        image_height: int,
        image_width: int,
    ) -> Tuple[Tensor, Tensor]:
        """Gather per-image bounds without per-ray device-to-host scalar reads."""
        device = img_idx.device
        cache_key = (str(device), int(num_images), int(image_height), int(image_width))
        cached = self._image_shape_lut_cache.get(cache_key)
        if cached is None:
            max_metadata_index = max(self.image_shapes, default=-1)
            lut_size = max(int(num_images), max_metadata_index + 1, 1)
            heights = torch.full((lut_size,), int(image_height), dtype=torch.long)
            widths = torch.full((lut_size,), int(image_width), dtype=torch.long)
            for image_index, (height, width) in self.image_shapes.items():
                if 0 <= image_index < lut_size:
                    heights[image_index] = int(height)
                    widths[image_index] = int(width)
            cached = (heights.to(device), widths.to(device))
            self._image_shape_lut_cache[cache_key] = cached
        height_lut, width_lut = cached
        valid = (img_idx >= 0) & (img_idx < height_lut.shape[0])
        safe_indices = img_idx.clamp(0, height_lut.shape[0] - 1)
        default_heights = torch.full_like(img_idx, int(image_height))
        default_widths = torch.full_like(img_idx, int(image_width))
        heights = torch.where(valid, height_lut[safe_indices], default_heights)
        widths = torch.where(valid, width_lut[safe_indices], default_widths)
        return heights, widths

    def _read_frequency_metadata(self, freq_file: Path) -> Optional[Dict]:
        metadata_path = freq_file.with_suffix(".json")
        if not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        value_type = metadata.get("value_type")
        if value_type != "scalar_resolution":
            raise ValueError(
                f"Frequency map {freq_file} has value_type={value_type!r}; "
                "LookCloser downstream expects scalar_resolution values, not level indices."
            )
        return metadata

    @staticmethod
    def _expected_map_shape(image_shape: Tuple[int, int], patch_size: int, stride: int) -> Tuple[int, int]:
        image_h, image_w = image_shape
        if image_h < patch_size or image_w < patch_size:
            raise ValueError(f"Image shape {image_shape} is smaller than patch_size={patch_size}.")
        return (
            ((image_h - patch_size) // stride) + 1,
            ((image_w - patch_size) // stride) + 1,
        )

    def _initialize_buckets(self, dataset: Dataset):
        """
        Loads frequency maps and buckets all pixels.
        This is a heavy operation run once at startup.
        """
        if not self.config.enable_fas:
            self.is_initialized = True
            return

        CONSOLE.print("[bold green]LookCloserPixelSampler:[/bold green] Initializing frequency buckets...")

        # 1. Locate Data Directory
        # We assume the dataset has a 'data_parser' or 'image_filenames' attribute to find the path.
        # Standard InputDataset has 'image_filenames'.
        if not hasattr(dataset, "image_filenames"):
            raise ValueError("LookCloserPixelSampler requires a dataset with 'image_filenames'.")

        # Assuming all images are in the same root data dir, we find the frequencies folder
        # relative to the first image or the dataset root.
        # A robust way is checking the parent of the first image.
        first_image_path = Path(dataset.image_filenames[0])
        data_dir = first_image_path.parent
        # Walk up until we find the frequency dir or hit root
        freq_dir = None
        current_dir = data_dir

        # Try to resolve where the "lookcloser_frequencies" folder is relative to data
        # Check standard location: {data_dir}/lookcloser_frequencies
        candidate = current_dir / self.config.frequency_map_dir
        if candidate.exists():
            freq_dir = candidate
        else:
            # Try parent (common structure: data/scene/images vs data/scene/lookcloser_frequencies)
            candidate = current_dir.parent / self.config.frequency_map_dir
            if candidate.exists():
                freq_dir = candidate

        if freq_dir is None:
            raise FileNotFoundError(
                f"Could not find frequency map directory '{self.config.frequency_map_dir}' "
                f"near {data_dir}. Please run the preprocessing script first."
            )

        map_records = []
        patch_sizes: List[int] = []
        patch_strides: List[int] = []
        min_res_values: List[float] = []
        max_res_values: List[float] = []
        num_level_values: List[int] = []

        # We must align with the dataset's image indexing.
        for img_idx, image_path in enumerate(dataset.image_filenames):
            # Load freq map
            freq_file = freq_dir / f"{Path(image_path).stem}.pt"
            if not freq_file.exists():
                CONSOLE.print(
                    f"[yellow]Warning:[/yellow] Frequency map missing for {image_path.name}. Skipping image in sampling.")
                continue

            f_map = torch.load(freq_file, map_location="cpu").float()
            if f_map.ndim != 2:
                raise ValueError(f"Frequency map {freq_file} must be a 2D tensor, got shape {tuple(f_map.shape)}.")
            if not torch.isfinite(f_map).all() or torch.min(f_map).item() <= 0.0:
                raise ValueError(f"Frequency map {freq_file} must contain finite positive scalar resolution values.")

            H_map, W_map = f_map.shape
            metadata = self._read_frequency_metadata(freq_file)
            min_res = float(self.config.min_res)
            max_res = float(self.config.max_res)
            num_levels = int(self.config.num_levels)
            patch_size = int(self.config.patch_size)
            patch_stride = int(self.config.stride)
            image_shape = None
            if metadata is not None:
                patch_size = int(metadata["patch_size"])
                patch_stride = int(metadata.get("stride", metadata["patch_size"]))
                min_res = float(metadata.get("min_res", min_res))
                max_res = float(metadata.get("max_res", max_res))
                num_levels = int(metadata.get("n_levels", num_levels))
                image_shape_raw = metadata.get("image_shape")
                if image_shape_raw is not None:
                    image_shape = (int(image_shape_raw[0]), int(image_shape_raw[1]))
                    expected_shape = self._expected_map_shape(image_shape, patch_size, patch_stride)
                    if (H_map, W_map) != expected_shape:
                        raise ValueError(
                            f"Frequency map {freq_file} shape {(H_map, W_map)} does not match metadata-derived "
                            f"patch grid {expected_shape} for image_shape={image_shape}, "
                            f"patch_size={patch_size}, stride={patch_stride}."
                        )
            else:
                if torch.max(f_map).item() <= float(num_levels - 1):
                    raise ValueError(
                        f"Frequency map {freq_file} has no metadata and looks like level-index data "
                        f"(max={torch.max(f_map).item():.3f}). Expected scalar resolution values."
                    )

            if patch_size <= 0 or patch_stride <= 0:
                raise ValueError(f"Invalid patch metadata for {freq_file}: patch_size={patch_size}, stride={patch_stride}.")
            if min_res <= 0 or max_res <= min_res or num_levels < 2:
                raise ValueError(
                    f"Invalid frequency schedule for {freq_file}: "
                    f"min_res={min_res}, max_res={max_res}, n_levels={num_levels}."
                )

            patch_sizes.append(patch_size)
            patch_strides.append(patch_stride)
            min_res_values.append(min_res)
            max_res_values.append(max_res)
            num_level_values.append(num_levels)
            if image_shape is not None:
                self.image_shapes[img_idx] = image_shape

            map_records.append((img_idx, freq_file, f_map, min_res, max_res, num_levels))

        unique_patch_sizes = sorted(set(patch_sizes))
        if len(unique_patch_sizes) > 1:
            raise ValueError(
                "LookCloserPixelSampler currently requires one patch_size across all frequency maps, "
                f"got {unique_patch_sizes}."
            )
        unique_patch_strides = sorted(set(patch_strides))
        if len(unique_patch_strides) > 1:
            raise ValueError(
                "LookCloserPixelSampler currently requires one stride across all frequency maps, "
                f"got {unique_patch_strides}."
            )
        unique_min_res = sorted(set(min_res_values))
        unique_max_res = sorted(set(max_res_values))
        unique_num_levels = sorted(set(num_level_values))
        if len(unique_min_res) > 1 or len(unique_max_res) > 1 or len(unique_num_levels) > 1:
            raise ValueError(
                "LookCloserPixelSampler requires one frequency schedule across all maps, got "
                f"min_res={unique_min_res}, max_res={unique_max_res}, n_levels={unique_num_levels}."
            )

        self.patch_size = unique_patch_sizes[0] if unique_patch_sizes else int(self.config.patch_size)
        self.patch_stride = unique_patch_strides[0] if unique_patch_strides else int(self.config.stride)
        num_levels_for_buckets = unique_num_levels[0] if unique_num_levels else int(self.config.num_levels)
        if num_levels_for_buckets != int(self.config.num_levels):
            raise ValueError(
                "Frequency-map n_levels does not match LookCloserPixelSamplerConfig.num_levels: "
                f"{num_levels_for_buckets} != {self.config.num_levels}. Use one schedule for preprocessing and training."
            )

        # 2. Iterate and Bucket
        # We bucket patch cells, then sample random pixels inside each selected cell.
        bucket_lists = {l: [] for l in range(self.config.num_levels)}

        for img_idx, freq_file, f_map, min_res, max_res, num_levels in map_records:

            # Compute levels for the map
            # l = log_b(f / min_res)
            b = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
            levels_map = torch.log(f_map / min_res) / np.log(b)
            levels_map = torch.clamp(torch.round(levels_map), 0, self.config.num_levels - 1).long()
            max_sampling_level = int(self.config.fas_max_sampling_level)
            if max_sampling_level >= 0:
                max_sampling_level = min(max_sampling_level, self.config.num_levels - 1)
                levels_map = torch.clamp(levels_map, 0, max_sampling_level)

            # Indices of the map
            ys, xs = torch.meshgrid(
                torch.arange(H_map),
                torch.arange(W_map),
                indexing="ij"
            )

            # Flatten
            flat_levels = levels_map.flatten()
            flat_ys = ys.flatten()
            flat_xs = xs.flatten()

            # Distribute to buckets
            for l in range(self.config.num_levels):
                mask = flat_levels == l
                if mask.any():
                    # Store (img_idx, map_y, map_x)
                    # We repeat img_idx
                    count = mask.sum().item()

                    # Create tensor chunk
                    img_indices = torch.full((count,), img_idx, dtype=torch.int32)
                    y_indices = flat_ys[mask].to(torch.int32)
                    x_indices = flat_xs[mask].to(torch.int32)

                    chunk = torch.stack([img_indices, y_indices, x_indices], dim=1)
                    bucket_lists[l].append(chunk)

        # 4. Consolidate Buckets
        for l in range(self.config.num_levels):
            if bucket_lists[l]:
                self.buckets[l] = torch.cat(bucket_lists[l], dim=0)
            else:
                self.buckets[l] = torch.empty((0, 3), dtype=torch.int32)

            if self.config.debug_mode:
                CONSOLE.print(f"Level {l}: {len(self.buckets[l])} patches")

        # 5. Calculate Sampling Distribution (1:3 Ramp, optionally bucket-count aware)
        ramp = np.linspace(
            self.config.sampling_ramp_start,
            self.config.sampling_ramp_end,
            self.config.num_levels
        )
        level_counts = np.array([self.buckets[l].shape[0] for l in range(self.config.num_levels)], dtype=np.float64)
        non_empty = level_counts > 0
        count_alpha = max(float(self.config.fas_level_count_alpha), 0.0)
        if count_alpha > 0.0:
            weights = ramp * np.where(non_empty, np.power(np.maximum(level_counts, 1.0), count_alpha), 0.0)
        else:
            weights = ramp * np.where(non_empty, 1.0, 0.0)
        if weights.sum() <= 0:
            weights = ramp
        probs = weights / weights.sum()

        # We calculate exact counts per batch later
        self.probs = probs
        self.level_counts = level_counts
        self._image_shape_lut_cache.clear()
        self.is_initialized = True
        self._prefetch_data_version += 1
        CONSOLE.print("[bold green]LookCloserPixelSampler:[/bold green] Initialization complete.")

    def _active_fas_strength(self) -> float:
        target = float(np.clip(self.config.fas_strength, 0.0, 1.0))
        warmup_steps = max(int(self.config.fas_warmup_steps), 0)
        ramp_steps = max(int(self.config.fas_ramp_steps), 0)
        if self.sample_count < warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            strength = target
        else:
            ramp_position = min(max(self.sample_count - warmup_steps, 0) / float(ramp_steps), 1.0)
            strength = target * ramp_position

        decay_start = int(self.config.fas_decay_start_steps)
        decay_steps = max(int(self.config.fas_decay_steps), 0)
        if decay_start >= 0 and self.sample_count >= decay_start:
            if decay_steps <= 0:
                return 0.0
            decay_position = min((self.sample_count - decay_start) / float(decay_steps), 1.0)
            strength *= 1.0 - decay_position
        return strength

    def _sample_levels_consolidated_h2d(
        self,
        counts: np.ndarray,
        num_images: int,
        image_height: int,
        image_width: int,
        device: Union[torch.device, str],
    ) -> List[Tensor]:
        """Sample FAS levels while transferring all selected CPU cells to CUDA once.

        CPU bucket-index draws and CUDA offset/fallback draws remain in the same
        level order as the legacy path. Only the deterministic selected-cell
        transfers are delayed and concatenated. This path is CUDA-only because
        on CPU the bucket and offset draws share one generator and cannot be
        reordered without changing the trajectory.
        """
        selected_cell_chunks: List[Tensor] = []
        # Each entry is either a completed fallback tensor or the information
        # needed to finish a non-empty level after the consolidated transfer.
        level_entries = []
        selected_cell_count = 0

        for level in range(self.config.num_levels):
            n_samples = int(counts[level])
            if n_samples == 0:
                continue

            bucket = self.buckets[level]
            num_in_bucket = bucket.shape[0]
            if num_in_bucket <= 0:
                # Keep fallback draws interleaved with non-empty-level draws in
                # exactly the same order as the legacy implementation.
                fallback = torch.rand((n_samples, 3), device=device) * torch.tensor(
                    [num_images, image_height, image_width], device=device
                )
                level_entries.append((fallback.long(), None))
                continue

            group_size = max(int(self.config.fas_patch_group_size), 1)
            selected_start = selected_cell_count
            if group_size == 1:
                rand_idx = torch.randint(0, num_in_bucket, (n_samples,))
                selected_cells_cpu = bucket[rand_idx]
                cells_for_level = n_samples
                y_off = torch.randint(0, self.patch_size, (n_samples,), device=device)
                x_off = torch.randint(0, self.patch_size, (n_samples,), device=device)
            else:
                patches_needed = int(np.ceil(n_samples / float(group_size)))
                rand_idx = torch.randint(0, num_in_bucket, (patches_needed,))
                selected_cells_cpu = bucket[rand_idx]
                cells_for_level = patches_needed

                grid_side = int(np.ceil(np.sqrt(group_size)))
                local_ids = torch.arange(group_size, device=device).repeat(patches_needed)[:n_samples]
                local_y = local_ids // grid_side
                local_x = local_ids % grid_side
                sub_h = max(self.patch_size // grid_side, 1)
                sub_w = max(self.patch_size // grid_side, 1)
                y_off = local_y * sub_h + torch.randint(0, sub_h, (n_samples,), device=device)
                x_off = local_x * sub_w + torch.randint(0, sub_w, (n_samples,), device=device)
                y_off = torch.clamp(y_off, 0, self.patch_size - 1)
                x_off = torch.clamp(x_off, 0, self.patch_size - 1)

            selected_cell_chunks.append(selected_cells_cpu)
            selected_cell_count += cells_for_level
            level_entries.append(
                (
                    None,
                    (
                        selected_start,
                        selected_cell_count,
                        n_samples,
                        group_size,
                        y_off,
                        x_off,
                    ),
                )
            )

        if selected_cell_chunks:
            selected_cells = torch.cat(selected_cell_chunks, dim=0).to(device).long()
        else:
            selected_cells = torch.empty((0, 3), dtype=torch.long, device=device)

        sampled_levels: List[Tensor] = []
        for fallback, selected_entry in level_entries:
            if fallback is not None:
                sampled_levels.append(fallback)
                continue

            selected_start, selected_end, n_samples, group_size, y_off, x_off = selected_entry
            selected_patches = selected_cells[selected_start:selected_end]
            if group_size != 1:
                selected_patches = selected_patches.repeat_interleave(group_size, dim=0)[:n_samples]

            img_idx = selected_patches[:, 0]
            y_coord = selected_patches[:, 1] * self.patch_stride + y_off
            x_coord = selected_patches[:, 2] * self.patch_stride + x_off
            if self.image_shapes:
                heights, widths = self._image_shapes_for_indices(
                    img_idx,
                    num_images=num_images,
                    image_height=image_height,
                    image_width=image_width,
                )
                y_coord = torch.minimum(torch.clamp_min(y_coord, 0), heights - 1)
                x_coord = torch.minimum(torch.clamp_min(x_coord, 0), widths - 1)
            else:
                y_coord = torch.clamp(y_coord, 0, image_height - 1)
                x_coord = torch.clamp(x_coord, 0, image_width - 1)
            sampled_levels.append(torch.stack([img_idx, y_coord, x_coord], dim=1))

        return sampled_levels

    def sample_method(
            self,
            batch_size: int,
            num_images: int,
            image_height: int,
            image_width: int,
            mask: Optional[Tensor] = None,
            device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Naive pixel sampler override.
        We ignore the standard random sampling and use our buckets.

        Note: The `PixelSampler` base class often calls this.
        However, `sample_method` signature doesn't pass the dataset, only dimensions.
        We rely on `_initialize_buckets` having been called via `sample` override or check here.
        But `sample` calls `sample_method`.
        """
        # This method is purely for returning random indices in the base class.
        # We will override `sample` instead to control the flow better,
        # but if `sample` is not overridden, we need this.

        # Since we need the stored buckets, and `sample_method` is stateless regarding the dataset in the base class signature,
        # we must ensure we have initialized.
        if not self.config.enable_fas:
            return super().sample_method(batch_size, num_images, image_height, image_width, mask, device)

        if not self.is_initialized:
            # We can't initialize here effectively without the dataset object.
            # We'll return random fallback if not initialized (sanity check).
            return super().sample_method(batch_size, num_images, image_height, image_width, mask, device)

        fas_batch_size = int(round(batch_size * self.current_fas_strength))
        uniform_batch_size = batch_size - fas_batch_size
        if fas_batch_size <= 0:
            return super().sample_method(batch_size, num_images, image_height, image_width, mask, device)

        # Determine samples per level for this batch. Assign rounding leftovers
        # by largest fractional remainder so capped/empty high levels do not get
        # accidental fallback-uniform samples.
        expected_counts = self.probs * fas_batch_size
        counts = np.floor(expected_counts).astype(int)
        diff = int(fas_batch_size - counts.sum())
        if diff > 0:
            remainders = expected_counts - counts
            for level in np.argsort(-remainders)[:diff]:
                counts[level] += 1
        elif diff < 0:
            remainders = expected_counts - counts
            for level in np.argsort(remainders)[: -diff]:
                if counts[level] > 0:
                    counts[level] -= 1

        indices_list = []

        if uniform_batch_size > 0:
            indices_list.append(
                super().sample_method(
                    uniform_batch_size,
                    num_images,
                    image_height,
                    image_width,
                    mask,
                    device,
                )
            )

        target_device = torch.device(device)
        if self.config.fas_consolidate_h2d and target_device.type == "cuda":
            indices_list.extend(
                self._sample_levels_consolidated_h2d(
                    counts,
                    num_images=num_images,
                    image_height=image_height,
                    image_width=image_width,
                    device=device,
                )
            )

            all_indices = torch.cat(indices_list, dim=0)
            shuffle_mask = torch.randperm(all_indices.shape[0], device=device)
            return all_indices[shuffle_mask]

        for l in range(self.config.num_levels):
            n_samples = counts[l]
            if n_samples == 0:
                continue

            bucket = self.buckets[l]
            num_in_bucket = bucket.shape[0]

            if num_in_bucket > 0:
                group_size = max(int(self.config.fas_patch_group_size), 1)
                if group_size == 1:
                    rand_idx = torch.randint(0, num_in_bucket, (n_samples,))
                    selected_patches = bucket[rand_idx].to(device).long()  # (N, 3) [img, y_patch, x_patch]
                    y_off = torch.randint(0, self.patch_size, (n_samples,), device=device)
                    x_off = torch.randint(0, self.patch_size, (n_samples,), device=device)
                else:
                    patches_needed = int(np.ceil(n_samples / float(group_size)))
                    rand_idx = torch.randint(0, num_in_bucket, (patches_needed,))
                    selected_cells = bucket[rand_idx].to(device).long()
                    selected_patches = selected_cells.repeat_interleave(group_size, dim=0)[:n_samples]

                    grid_side = int(np.ceil(np.sqrt(group_size)))
                    local_ids = torch.arange(group_size, device=device).repeat(patches_needed)[:n_samples]
                    local_y = local_ids // grid_side
                    local_x = local_ids % grid_side
                    sub_h = max(self.patch_size // grid_side, 1)
                    sub_w = max(self.patch_size // grid_side, 1)
                    y_off = local_y * sub_h + torch.randint(0, sub_h, (n_samples,), device=device)
                    x_off = local_x * sub_w + torch.randint(0, sub_w, (n_samples,), device=device)
                    y_off = torch.clamp(y_off, 0, self.patch_size - 1)
                    x_off = torch.clamp(x_off, 0, self.patch_size - 1)

                img_idx = selected_patches[:, 0]
                y_coord = selected_patches[:, 1] * self.patch_stride + y_off
                x_coord = selected_patches[:, 2] * self.patch_stride + x_off

                # Clamp to the selected image's shape when metadata is available.
                if self.image_shapes:
                    heights, widths = self._image_shapes_for_indices(
                        img_idx,
                        num_images=num_images,
                        image_height=image_height,
                        image_width=image_width,
                    )
                    y_coord = torch.minimum(torch.clamp_min(y_coord, 0), heights - 1)
                    x_coord = torch.minimum(torch.clamp_min(x_coord, 0), widths - 1)
                else:
                    y_coord = torch.clamp(y_coord, 0, image_height - 1)
                    x_coord = torch.clamp(x_coord, 0, image_width - 1)

                indices_list.append(torch.stack([img_idx, y_coord, x_coord], dim=1))
            else:
                # Fallback if bucket empty: Random uniform sample
                # (Rare case where a frequency level doesn't exist in the dataset)
                fallback = torch.rand((n_samples, 3), device=device) * torch.tensor(
                    [num_images, image_height, image_width], device=device
                )
                indices_list.append(fallback.long())

        # Concatenate and Shuffle
        all_indices = torch.cat(indices_list, dim=0)

        # Shuffle to mix frequency levels in the batch
        shuffle_mask = torch.randperm(all_indices.shape[0], device=device)
        return all_indices[shuffle_mask]

    def sample(self, image_batch: Dict, *, commit_sample_count: bool = True):
        """
        Main sampling entry point called by DataManager.
        """
        # Lazy initialization if needed
        # We need access to the dataset. image_batch might be the dataset itself
        # depending on how DataManager calls it.
        # In VanillaDataManager: pixel_sampler.sample(self.train_dataset)

        if not self.config.enable_fas:
            return super().sample(image_batch)

        if not self.is_initialized:
            if self.dataset is not None:
                self._initialize_buckets(self.dataset)
            elif isinstance(image_batch, Dataset):
                self._initialize_buckets(image_batch)
            else:
                CONSOLE.print(
                    "[yellow]LookCloserPixelSampler:[/yellow] Dataset unavailable; falling back to uniform sampling."
                )
                return super().sample(image_batch)

        # Call the standard sample logic which internally calls sample_method
        self.current_fas_strength = self._active_fas_strength()
        batch = super().sample(image_batch)
        if commit_sample_count:
            self.sample_count += 1
        return batch

    def _prefetch_config_signature(self) -> Tuple[object, ...]:
        return (
            bool(self.config.enable_fas),
            int(self.config.num_levels),
            float(self.config.sampling_ramp_start),
            float(self.config.sampling_ramp_end),
            float(self.config.fas_strength),
            int(self.config.fas_warmup_steps),
            int(self.config.fas_ramp_steps),
            int(self.config.fas_decay_start_steps),
            int(self.config.fas_decay_steps),
            float(self.config.fas_level_count_alpha),
            int(self.config.fas_patch_group_size),
            int(self.config.fas_max_sampling_level),
            bool(self.config.fas_consolidate_h2d),
            int(self.patch_size),
            int(self.patch_stride),
            bool(self.config.keep_full_image),
        )

    @staticmethod
    def _prefetch_image_order(image_batch: Dict) -> Tuple[int, ...]:
        image_idx = image_batch.get("image_idx")
        if not isinstance(image_idx, Tensor):
            raise TypeError("CPU FAS prefetch requires tensor image_idx")
        return tuple(int(value) for value in image_idx.detach().cpu().tolist())

    @staticmethod
    def _prefetch_tensor_identity(value: Tensor) -> Tuple[object, ...]:
        """Track replacement and in-place mutation without a device-to-host copy."""

        return (
            id(value),
            int(value.data_ptr()),
            str(value.device),
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            int(value._version),
        )

    def prefetch_live_signature(self, image_batch: Dict) -> Tuple[object, ...]:
        image = image_batch.get("image")
        image_idx = image_batch.get("image_idx")
        if not isinstance(image, Tensor):
            raise TypeError("CPU FAS prefetch requires one homogeneous image tensor")
        if not isinstance(image_idx, Tensor):
            raise TypeError("CPU FAS prefetch requires tensor image_idx")
        return (
            int(self.num_rays_per_batch),
            self._prefetch_config_signature(),
            int(self._prefetch_data_version),
            self._prefetch_tensor_identity(image),
            self._prefetch_tensor_identity(image_idx),
        )

    def build_prefetch_snapshot(self, image_batch: Dict) -> LookCloserFASPrefetchSnapshot:
        """Freeze the exact CPU-only FAS inputs consumed by the worker."""

        if set(image_batch) != {"image", "image_idx"}:
            raise ValueError("CPU FAS prefetch supports image/image_idx batches without masks or metadata")
        image = image_batch["image"]
        image_idx = image_batch["image_idx"]
        if not isinstance(image, Tensor) or image.device.type != "cpu" or image.ndim != 4:
            raise ValueError("CPU FAS prefetch requires one homogeneous CPU image tensor")
        if not isinstance(image_idx, Tensor):
            raise TypeError("CPU FAS prefetch requires tensor image_idx")
        if not self.config.enable_fas or not self.is_initialized:
            raise RuntimeError("CPU FAS buckets must be initialized before snapshotting")
        if self.config.keep_full_image:
            raise ValueError("CPU FAS prefetch does not support keep_full_image")
        num_images, image_height, image_width, _ = image.shape
        max_metadata_index = max(self.image_shapes, default=-1)
        lut_size = max(int(num_images), max_metadata_index + 1, 1)
        image_heights = torch.full((lut_size,), int(image_height), dtype=torch.long)
        image_widths = torch.full((lut_size,), int(image_width), dtype=torch.long)
        for image_index, (height, width) in self.image_shapes.items():
            if 0 <= image_index < lut_size:
                image_heights[image_index] = int(height)
                image_widths[image_index] = int(width)
        return LookCloserFASPrefetchSnapshot(
            image=image.detach(),
            image_idx=image_idx.detach().cpu().clone(),
            buckets=tuple(self.buckets[level].detach().cpu().clone() for level in range(self.config.num_levels)),
            probs=self.probs.copy(),
            image_heights=image_heights,
            image_widths=image_widths,
            num_levels=int(self.config.num_levels),
            num_rays_per_batch=int(self.num_rays_per_batch),
            patch_size=int(self.patch_size),
            patch_stride=int(self.patch_stride),
            fas_strength=float(self.config.fas_strength),
            fas_warmup_steps=max(int(self.config.fas_warmup_steps), 0),
            fas_ramp_steps=max(int(self.config.fas_ramp_steps), 0),
            fas_decay_start_steps=int(self.config.fas_decay_start_steps),
            fas_decay_steps=max(int(self.config.fas_decay_steps), 0),
            fas_patch_group_size=max(int(self.config.fas_patch_group_size), 1),
            config_signature=self._prefetch_config_signature(),
            data_version=int(self._prefetch_data_version),
            image_order=self._prefetch_image_order(image_batch),
        )
