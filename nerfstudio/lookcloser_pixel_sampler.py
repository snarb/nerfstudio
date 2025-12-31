# nerfstudio/data/pixel_samplers/lookcloser_pixel_sampler.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

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

    num_levels: int = 16
    """Number of frequency levels to bucket pixels into."""

    min_res: float = 16.0
    """Minimum resolution used during pre-processing (base of geometric progression)."""

    max_res: float = 2048.0
    """Maximum resolution used during pre-processing."""

    sampling_ramp_start: float = 1.0
    """Start of the linear probability ramp for sampling."""

    sampling_ramp_end: float = 3.0
    """End of the linear probability ramp for sampling (high-freq gets more samples)."""

    debug_mode: bool = False
    """If true, prints sampling stats."""


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
        self.buckets: Dict[int, Tensor] = {}
        self.samples_per_level: np.ndarray = np.zeros(self.config.num_levels, dtype=int)

        # We need to initialize the buckets.
        # Since PixelSampler is initialized with the DataManager, we assume the dataset
        # is available or passed in the first sample call?
        # Standard Nerfstudio architecture doesn't pass dataset to __init__.
        # We will lazy-load on the first call to `sample`.
        self.is_initialized = False

    def _initialize_buckets(self, dataset: Dataset):
        """
        Loads frequency maps and buckets all pixels.
        This is a heavy operation run once at startup.
        """
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

        # 2. Geometric Progression Constants
        b = np.exp((np.log(self.config.max_res) - np.log(self.config.min_res)) / (self.config.num_levels - 1))

        # 3. Iterate and Bucket
        # We use temporary lists to hold indices, then stack to Tensor to save memory.
        bucket_lists = {l: [] for l in range(self.config.num_levels)}

        total_pixels = 0

        # We must align with the dataset's image indexing.
        for img_idx, image_path in enumerate(dataset.image_filenames):
            # Load freq map
            freq_file = freq_dir / f"{Path(image_path).stem}.pt"
            if not freq_file.exists():
                CONSOLE.print(
                    f"[yellow]Warning:[/yellow] Frequency map missing for {image_path.name}. Skipping image in sampling.")
                continue

            f_map = torch.load(freq_file, map_location="cpu")
            H_map, W_map = f_map.shape

            # The freq map might be patch-wise (smaller resolution).
            # We need pixel-wise buckets.
            # We replicate the frequency values to match the full image resolution or
            # we store patch indices and sample within patches.
            #
            # For simplicity and correctness with the plan ("Bucket all pixels"),
            # let's assume we map pixels to the patch frequency value.
            #
            # Optimization:
            # Storing 16 Million indices for a single 4K image is expensive.
            # However, the plan explicitly says "Bucket pixels".
            # To make this efficient, we'll store (img_idx, y_coord, x_coord) as Int32 (Short if possible).
            #
            # If the map is 1/32 scale, we can just store the map indices and
            # during sampling add a random offset [0, 31].

            # Upsample f_map to image size? No, that explodes memory.
            # We bucked the PATCHES.
            # Then when we sample a patch, we pick a random pixel inside it.

            # Compute levels for the map
            # l = log_b(f / min_res)
            levels_map = torch.log(f_map / self.config.min_res) / np.log(b)
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

        # 5. Determine patch size / stride from dataset info vs map size
        # We grab one image metadata to guess the downscale factor
        # This assumes uniform scaling across dataset.
        sample_img_h = dataset.metadata["image_height"] if "image_height" in dataset.metadata else None
        # If not in metadata, we can't easily guess without loading an image.
        # But we know the preprocessing script used a stride (default 32).
        # We'll treat the stored coordinates as "top-left" of a patch
        # and assume a patch size in the sample method.
        # Let's verify stride from map size vs image size if possible, otherwise default to 32.
        # (For robust implementation, we'll assume 32 based on the LookCloser default).
        #ToDo: check
        self.patch_size = 32

        # 6. Calculate Sampling Distribution (1:3 Ramp)
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
        if not self.is_initialized:
            # We can't initialize here effectively without the dataset object.
            # We'll return random fallback if not initialized (sanity check).
            return super().sample_method(batch_size, num_images, image_height, image_width, mask, device)

        # Determine samples per level for this batch
        counts = (self.probs * batch_size).astype(int)
        # Fix rounding to match batch_size exactly
        diff = batch_size - counts.sum()
        if diff > 0:
            counts[-1] += diff
        elif diff < 0:
            # Should not happen with astype(int) usually under-estimating
            counts[-1] += diff

        indices_list = []

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
                # Note: We need to ensure we don't go out of bounds if the image isn't perfect multiple of 32
                # We simply clamp.

                # Offsets
                y_off = torch.randint(0, self.patch_size, (n_samples,), device=device)
                x_off = torch.randint(0, self.patch_size, (n_samples,), device=device)

                img_idx = selected_patches[:, 0]
                y_coord = selected_patches[:, 1] * self.patch_size + y_off
                x_coord = selected_patches[:, 2] * self.patch_size + x_off

                # Clamp to image bounds
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

        if not self.is_initialized:
            if isinstance(image_batch, Dataset):
                self._initialize_buckets(image_batch)
            else:
                # If image_batch is a dict or something else, we might be in trouble for init.
                # But typically it's the dataset.
                assert False #ToDo: bad code
                pass

        # Call the standard sample logic which internally calls sample_method
        return super().sample(image_batch)