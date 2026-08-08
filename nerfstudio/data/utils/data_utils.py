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

"""Utility functions to allow easy re-use of common operations across dataloaders."""

import io
import os
from pathlib import Path
from typing import IO, List, Tuple, Union

# OpenCV ships its OpenEXR codec behind this explicit opt-in.  It must be set
# before importing cv2; callers should not need to mutate their shell environment.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage


def pil_to_numpy(im: PILImage) -> np.ndarray:
    """Converts a PIL Image object to a NumPy array.

    Args:
        im (PIL.Image.Image): The input PIL Image object.

    Returns:
        numpy.ndarray representing the image data.
    """
    # Load in image completely (PIL defaults to lazy loading)
    im.load()

    # Pillow 11 changed the private encoder setimage signature.  The public
    # array interface preserves the decoded uint8 pixels without touching
    # model, sampler, optimizer, or RNG semantics.
    return np.asarray(im)


def is_exr_path(filepath: Union[Path, IO[bytes]]) -> bool:
    """Return whether ``filepath`` names an OpenEXR image.

    Compressed-image caches pass a ``BytesIO`` without a useful name, so callers
    that already know the source path should use that path for format dispatch.
    """

    name = getattr(filepath, "name", filepath)
    return isinstance(name, (str, Path)) and Path(name).suffix.lower() == ".exr"


def load_exr_image(
    filepath: Union[Path, IO[bytes]], scale_factor: float = 1.0
) -> np.ndarray:
    """Decode an OpenEXR image as contiguous scene-linear RGB(A) float32.

    OpenCV decodes color channels as BGR(A); this function is the single channel
    order conversion used by dataset loading and preprocessing.  No transfer
    function, exposure, clipping, or normalization is applied.
    """

    if isinstance(filepath, (str, Path)):
        image = cv2.imread(str(Path(filepath).absolute()), cv2.IMREAD_UNCHANGED)
    else:
        if isinstance(filepath, io.BytesIO):
            payload = filepath.getbuffer()
        else:
            position = filepath.tell() if hasattr(filepath, "tell") else None
            if hasattr(filepath, "seek"):
                filepath.seek(0)
            payload = filepath.read()
            if position is not None and hasattr(filepath, "seek"):
                filepath.seek(position)
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Failed to decode OpenEXR image: {getattr(filepath, 'name', filepath)!s}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"OpenEXR image must have RGB or RGBA channels, got shape {image.shape}")

    if image.shape[-1] == 3:
        image = image[..., (2, 1, 0)]
    else:
        image = image[..., (2, 1, 0, 3)]
    image = np.asarray(image, dtype=np.float32)

    if scale_factor != 1.0:
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}")
        height, width = image.shape[:2]
        new_size = (max(1, int(width * scale_factor)), max(1, int(height * scale_factor)))
        interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(image, new_size, interpolation=interpolation)
        if image.ndim == 2:
            image = image[..., None]

    return np.ascontiguousarray(image, dtype=np.float32)


def write_exr_image(filepath: Path, image: Union[np.ndarray, torch.Tensor]) -> None:
    """Write scene-linear RGB(A) data to OpenEXR without clipping or transfer functions."""

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"OpenEXR output must have RGB or RGBA channels, got shape {array.shape}")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    encoded = array[..., (2, 1, 0)] if array.shape[-1] == 3 else array[..., (2, 1, 0, 3)]
    if not cv2.imwrite(str(filepath), np.ascontiguousarray(encoded)):
        raise OSError(f"Failed to write OpenEXR image: {filepath}")


def get_image_mask_tensor_from_path(filepath: Union[Path, IO[bytes]], scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.Resampling.NEAREST)
    mask_tensor = torch.from_numpy(pil_to_numpy(pil_mask)).unsqueeze(-1).bool()
    if len(mask_tensor.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return mask_tensor


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype=torch.int64).view(1, 1, -1)
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.Resampling.NEAREST)
    semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width, 1].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath).astype(np.float32) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float32) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)  # type: ignore
    return torch.from_numpy(image[:, :, np.newaxis])


def identity_collate(x):
    """This function does nothing but serves to help our dataloaders have a pickleable function, as lambdas are not pickleable"""
    return x
