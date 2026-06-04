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

        # We need to initialize the buckets.
        # Since PixelSampler is initialized with the DataManager, we assume the dataset
        # is available or passed in the first sample call?
        # Standard Nerfstudio architecture doesn't pass dataset to __init__.
        # We will lazy-load on the first call to `sample`.
        self.is_initialized = False
        self.patch_size = int(self.config.patch_size)
        self.patch_stride = int(self.config.stride)
        self.image_shapes: Dict[int, Tuple[int, int]] = {}
        self.sample_count = 0
        self.current_fas_strength = 1.0

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

        # 5. Calculate Sampling Distribution (1:3 Ramp)
        ramp = np.linspace(
            self.config.sampling_ramp_start,
            self.config.sampling_ramp_end,
            self.config.num_levels
        )
        probs = ramp / ramp.sum()

        # We calculate exact counts per batch later
        self.probs = probs
        self.is_initialized = True
        CONSOLE.print("[bold green]LookCloserPixelSampler:[/bold green] Initialization complete.")

    def _active_fas_strength(self) -> float:
        target = float(np.clip(self.config.fas_strength, 0.0, 1.0))
        warmup_steps = max(int(self.config.fas_warmup_steps), 0)
        ramp_steps = max(int(self.config.fas_ramp_steps), 0)
        if self.sample_count < warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return target
        ramp_position = min(max(self.sample_count - warmup_steps, 0) / float(ramp_steps), 1.0)
        return target * ramp_position

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

        # Determine samples per level for this batch
        counts = (self.probs * fas_batch_size).astype(int)
        # Fix rounding to match batch_size exactly
        diff = fas_batch_size - counts.sum()
        if diff > 0:
            counts[-1] += diff
        elif diff < 0:
            # Should not happen with astype(int) usually under-estimating
            counts[-1] += diff

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

        for l in range(self.config.num_levels):
            n_samples = counts[l]
            if n_samples == 0:
                continue

            bucket = self.buckets[l]
            num_in_bucket = bucket.shape[0]

            if num_in_bucket > 0:
                # Random selection from bucket
                rand_idx = torch.randint(0, num_in_bucket, (n_samples,))
                selected_patches = bucket[rand_idx].to(device).long()  # (N, 3) [img, y_patch, x_patch]

                # Now convert patch top-left to random pixel within patch
                # Add random offset [0, patch_size)
                # Note: We need to ensure we don't go out of bounds if the image has uncovered tail pixels.
                # We simply clamp.

                # Offsets
                y_off = torch.randint(0, self.patch_size, (n_samples,), device=device)
                x_off = torch.randint(0, self.patch_size, (n_samples,), device=device)

                img_idx = selected_patches[:, 0]
                y_coord = selected_patches[:, 1] * self.patch_stride + y_off
                x_coord = selected_patches[:, 2] * self.patch_stride + x_off

                # Clamp to the selected image's shape when metadata is available.
                if self.image_shapes:
                    heights = torch.tensor(
                        [self.image_shapes.get(int(i.item()), (image_height, image_width))[0] for i in img_idx],
                        device=device,
                        dtype=torch.long,
                    )
                    widths = torch.tensor(
                        [self.image_shapes.get(int(i.item()), (image_height, image_width))[1] for i in img_idx],
                        device=device,
                        dtype=torch.long,
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
        shuffle_mask = torch.randperm(batch_size, device=device)
        return all_indices[shuffle_mask]

    def sample(self, image_batch: Dict):
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
        self.sample_count += 1
        return batch
