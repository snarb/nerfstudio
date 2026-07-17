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
Ray generator.
"""

import time
from typing import Dict

import torch
from jaxtyping import Int
from torch import Tensor, nn

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle


class RayGenerator(nn.Module):
    """torch.nn Module for generating rays.
    This class is the interface between the scene's cameras/camera optimizer and the ray sampler.

    Args:
        cameras: Camera objects containing camera info.
        cache_rays: Precompute the pixel-dependent fields for static training cameras.
            This is deliberately opt-in and supports only uniform, one-dimensional
            batches of perspective cameras.
        cache_chunk_size: Maximum number of rays passed to ``generate_rays`` while
            constructing the cache.
    """

    image_coords: Tensor

    def __init__(self, cameras: Cameras, cache_rays: bool = False, cache_chunk_size: int = 1 << 20) -> None:
        super().__init__()
        self.cameras = cameras
        self.register_buffer("image_coords", cameras.get_image_coords(), persistent=False)

        # None buffers make the disabled/default path state-dict compatible with
        # the historical RayGenerator. All populated cache buffers are
        # non-persistent because they are derived entirely from Cameras.
        self.register_buffer("_cached_directions", None, persistent=False)
        self.register_buffer("_cached_pixel_area", None, persistent=False)
        self.register_buffer("_cached_directions_norm", None, persistent=False)
        self.register_buffer("_cached_camera_origins", None, persistent=False)
        self.register_buffer("_cached_camera_times", None, persistent=False)

        self.cache_rays = cache_rays
        self.cache_chunk_size = cache_chunk_size
        self.cache_build_seconds = 0.0
        self.cache_num_bytes = 0
        self._cached_metadata_buffer_names: Dict[str, str] = {}

        if cache_rays:
            self._build_ray_cache()

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        """Synchronize only when needed to report honest cache build time."""
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _validate_cache_inputs(self) -> None:
        """Fail closed when indexed lookup would not preserve ray semantics."""
        if self.cache_chunk_size <= 0:
            raise ValueError("cache_chunk_size must be positive")
        if self.cameras.ndim != 1:
            raise ValueError("Ray caching requires a one-dimensional camera batch")
        if len(self.cameras) == 0:
            raise ValueError("Ray caching requires at least one camera")
        if self.cameras.is_jagged:
            raise ValueError("Ray caching does not support jagged cameras")
        if not torch.all(self.cameras.camera_type == CameraType.PERSPECTIVE.value):
            raise ValueError("Ray caching currently supports perspective cameras only")

        static_tensors = {
            "camera_to_worlds": self.cameras.camera_to_worlds,
            "fx": self.cameras.fx,
            "fy": self.cameras.fy,
            "cx": self.cameras.cx,
            "cy": self.cameras.cy,
            "distortion_params": self.cameras.distortion_params,
            "times": self.cameras.times,
        }
        for name, value in static_tensors.items():
            if value is not None and value.requires_grad:
                raise ValueError(f"Ray caching requires static cameras; {name} requires gradients")

        if self.cameras.metadata is not None:
            if not isinstance(self.cameras.metadata, dict):
                raise TypeError("Ray caching requires camera metadata to be a dictionary")
            for key, value in self.cameras.metadata.items():
                if not isinstance(key, str) or not isinstance(value, Tensor):
                    raise TypeError("Ray caching supports only top-level string-to-tensor camera metadata")
                if value.requires_grad:
                    raise ValueError(f"Ray caching requires static camera metadata; {key!r} requires gradients")

    def _build_ray_cache(self) -> None:
        """Build pixel-dependent ray fields in bounded-memory chunks without RNG."""
        self._validate_cache_inputs()
        device = self.cameras.device
        self._synchronize(device)
        start_time = time.perf_counter()

        num_cameras = len(self.cameras)
        height = int(self.cameras.height.reshape(-1)[0].item())
        width = int(self.cameras.width.reshape(-1)[0].item())
        rays_per_camera = height * width
        total_rays = num_cameras * rays_per_camera

        cached_directions = None
        cached_pixel_area = None
        cached_directions_norm = None

        # image_coords is intentionally indexed on its current device, just as
        # in the uncached forward path. Cameras.generate_rays moves coords and
        # camera indices to the camera device.
        with torch.no_grad():
            for chunk_start in range(0, total_rays, self.cache_chunk_size):
                chunk_end = min(chunk_start + self.cache_chunk_size, total_rays)
                linear = torch.arange(chunk_start, chunk_end, device=self.image_coords.device)
                camera_indices = torch.div(linear, rays_per_camera, rounding_mode="floor")
                pixel_indices = linear.remainder(rays_per_camera)
                y = torch.div(pixel_indices, width, rounding_mode="floor")
                x = pixel_indices.remainder(width)
                coords = self.image_coords[y, x]

                ray_bundle = self.cameras.generate_rays(
                    camera_indices=camera_indices.unsqueeze(-1),
                    coords=coords,
                )
                directions_norm = ray_bundle.metadata["directions_norm"]

                if cached_directions is None:
                    cached_directions = torch.empty(
                        (total_rays, 3), dtype=ray_bundle.directions.dtype, device=ray_bundle.directions.device
                    )
                    cached_pixel_area = torch.empty(
                        (total_rays, 1), dtype=ray_bundle.pixel_area.dtype, device=ray_bundle.pixel_area.device
                    )
                    cached_directions_norm = torch.empty(
                        (total_rays, 1), dtype=directions_norm.dtype, device=directions_norm.device
                    )

                cached_directions[chunk_start:chunk_end].copy_(ray_bundle.directions)
                cached_pixel_area[chunk_start:chunk_end].copy_(ray_bundle.pixel_area)
                cached_directions_norm[chunk_start:chunk_end].copy_(directions_norm)

        assert cached_directions is not None
        assert cached_pixel_area is not None
        assert cached_directions_norm is not None
        self._cached_directions = cached_directions.view(num_cameras, height, width, 3)
        self._cached_pixel_area = cached_pixel_area.view(num_cameras, height, width, 1)
        self._cached_directions_norm = cached_directions_norm.view(num_cameras, height, width, 1)
        self._cached_camera_origins = self.cameras.camera_to_worlds[..., :3, 3].detach().clone()
        if self.cameras.times is not None:
            self._cached_camera_times = self.cameras.times.detach().clone()

        if self.cameras.metadata is not None:
            for index, (key, value) in enumerate(self.cameras.metadata.items()):
                buffer_name = f"_cached_camera_metadata_{index}"
                self.register_buffer(buffer_name, value.detach().clone(), persistent=False)
                self._cached_metadata_buffer_names[key] = buffer_name

        cache_buffers = [
            self._cached_directions,
            self._cached_pixel_area,
            self._cached_directions_norm,
            self._cached_camera_origins,
            self._cached_camera_times,
            *(getattr(self, name) for name in self._cached_metadata_buffer_names.values()),
        ]
        self.cache_num_bytes = sum(
            value.numel() * value.element_size() for value in cache_buffers if value is not None
        )
        self._synchronize(device)
        self.cache_build_seconds = time.perf_counter() - start_time

    def _forward_cached(self, ray_indices: Int[Tensor, "num_rays 3"]) -> RayBundle:
        """Return a byte-exact indexed subset of the precomputed static rays."""
        assert self._cached_directions is not None
        assert self._cached_pixel_area is not None
        assert self._cached_directions_norm is not None
        assert self._cached_camera_origins is not None

        # Match Cameras.generate_rays: output fields live on the cameras/cache
        # device and camera_indices are long, even if input integer width differs.
        indices = ray_indices.to(device=self._cached_directions.device)
        c = indices[:, 0]
        y = indices[:, 1]
        x = indices[:, 2]
        camera_indices = c.unsqueeze(-1).to(torch.long)

        metadata = {
            key: getattr(self, buffer_name)[c]
            for key, buffer_name in self._cached_metadata_buffer_names.items()
        }
        metadata["directions_norm"] = self._cached_directions_norm[c, y, x]

        times = self._cached_camera_times[c] if self._cached_camera_times is not None else None
        return RayBundle(
            origins=self._cached_camera_origins[c],
            directions=self._cached_directions[c, y, x],
            pixel_area=self._cached_pixel_area[c, y, x],
            camera_indices=camera_indices,
            times=times,
            metadata=metadata,
        )

    def forward(self, ray_indices: Int[Tensor, "num_rays 3"]) -> RayBundle:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indices for target rays.
        """
        if self.cache_rays:
            return self._forward_cached(ray_indices)

        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x]

        ray_bundle = self.cameras.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=coords,
        )
        return ray_bundle
