# nerfstudio/model_components/lookcloser_grid.py

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.rich_utils import CONSOLE


class FrequencyGridManager(nn.Module):
    """
    Manages the frequency voxel grid for LookCloser (FA-NeRF).

    This grid stores the maximum required frequency level (0 to L-1) for regions in 3D space.
    It is initialized from sparse SfM points and updated during training using dense depth estimates.

    Ref: https://arxiv.org/abs/2503.18513
    """

    def __init__(
        self,
        scene_box: SceneBox,
        resolution: int = 128,
        num_levels: int = 16,
        min_res: float = 16.0,
        max_res: float = 2048.0,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_levels = num_levels
        self.min_res = min_res
        self.max_res = max_res

        # Store AABB bounds for world-to-grid conversion
        self.aabb_min = scene_box.aabb[0]
        self.aabb_max = scene_box.aabb[1]
        self.register_buffer("aabb_min_buf", self.aabb_min)
        self.register_buffer("aabb_max_buf", self.aabb_max)
        self.register_buffer("aabb_size_buf", self.aabb_max - self.aabb_min)

        # The dense voxel grid.
        # We store as float to allow potential interpolation, though Logic uses discrete levels.
        # Initialized to -1.0 so we can distinguish empty voxels if needed,
        # but standard logic initializes to 0.0 (lowest freq).
        self.register_buffer(
            "grid",
            torch.zeros((resolution, resolution, resolution), dtype=torch.float32)
        )

        # Geometric growth factor 'b' for level conversion
        # N_l = N_min * b^l  =>  b = (N_max / N_min) ^ (1 / (L-1))
        self.b = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))

    def world_to_grid(self, positions: Tensor) -> Tensor:
        """
        Maps world coordinates to normalized grid coordinates [0, resolution-1].
        """
        # Normalize to [0, 1] within AABB
        norm_pos = (positions - self.aabb_min_buf) / self.aabb_size_buf

        # Scale to grid resolution and clamp
        grid_coords = torch.clamp(norm_pos * (self.resolution - 1), 0, self.resolution - 1.001)
        return grid_coords

    def grid_to_indices(self, positions: Tensor) -> Tensor:
        """
        Maps world coordinates to discrete grid indices (x, y, z).
        """
        coords = self.world_to_grid(positions).long()
        return coords

    def freq_to_level(self, f_scalar: Union[float, Tensor]) -> Union[int, Tensor]:
        """
        Maps scalar frequency resolution (N_l) to discrete level index (0..num_levels-1).
        Formula: l = log_b( N_l / N_min )
        """
        if torch.is_tensor(f_scalar):
            val = torch.log(f_scalar / self.min_res) / np.log(self.b)
            return torch.clamp(torch.round(val), 0, self.num_levels - 1).float()
        else:
            val = np.log(f_scalar / self.min_res) / np.log(self.b)
            return int(np.clip(np.round(val), 0, self.num_levels - 1))

    def level_to_freq(self, level: Union[int, Tensor]) -> Union[float, Tensor]:
        """Maps level index to scalar frequency resolution."""
        return self.min_res * (self.b ** level)

    def query(self, positions: Tensor) -> Tensor:
        """
        Queries the grid at specific 3D positions using Nearest Neighbor.

        Args:
            positions: (N, 3) World coordinates.
        Returns:
            levels: (N, 1) Frequency levels.
        """
        indices = self.grid_to_indices(positions) # (N, 3)
        levels = self.grid[indices[:, 0], indices[:, 1], indices[:, 2]]
        return levels.unsqueeze(-1)

    def update_max(self, positions: Tensor, new_levels: Tensor):
        """
        Updates the grid using an atomic MAX operation: grid[pos] = max(grid[pos], new_levels).
        Uses scatter_reduce_ for efficient GPU execution.
        """
        if positions.numel() == 0:
            return

        indices = self.grid_to_indices(positions) # (N, 3)
        x, y, z = indices[:, 0], indices[:, 1], indices[:, 2]

        # Flatten indices for 1D scatter
        flat_indices = x * (self.resolution**2) + y * self.resolution + z
        flat_grid = self.grid.view(-1)

        # Ensure new_levels matches shape
        vals = new_levels.reshape(-1).to(flat_grid.dtype)

        # Atomic Max Update
        # Requires PyTorch >= 1.12
        flat_grid.scatter_reduce_(0, flat_indices, vals, reduce="amax", include_self=True)

    def initialize_from_sparse(
        self,
        sparse_points: Tensor,
        observations: Dict[str, Tensor],
        image_freq_maps: Dict[int, Tensor],
        cameras: Cameras,
    ):
        """
        Part 2 Initialization: Populates the grid using sparse SfM points.

        For each point, we calculate the 3D frequency requirement based on its projections
        in visible cameras, take the median across views, and update the grid.

        Args:
            sparse_points: (P, 3) XYZ coordinates of sparse 3D points.
            observations: Dictionary containing visibility graph:
                - 'point_indices': (K,) Indices into sparse_points.
                - 'img_indices': (K,) Indices into cameras.
                - 'uv': (K, 2) Normalized UV coordinates of the projection [0,1].
            image_freq_maps: Dict mapping image_idx -> (H, W) frequency map tensor.
            cameras: Cameras object.
        """
        device = self.grid.device
        sparse_points = sparse_points.to(device)

        # Unpack observations
        pt_indices = observations['point_indices'].to(device)
        img_indices = observations['img_indices'].to(device)
        uv_coords = observations['uv'].to(device)

        num_obs = pt_indices.shape[0]
        if num_obs == 0:
            CONSOLE.print("[yellow]Warning: No observations provided for LookCloser initialization. Grid remains empty.[/yellow]")
            return

        CONSOLE.print(f"[bold green]LookCloser:[/bold green] Initializing grid from {sparse_points.shape[0]} points and {num_obs} observations...")

        # --- Step 1: Compute Candidate f_3D for all observations ---
        # Formula: f_3d = f_2d * (focal / dist)

        # A. Gather f_2D from maps
        # Since maps are separate tensors (potentially different shapes), we iterate unique images.
        f_2d_candidates = torch.zeros(num_obs, device=device)

        unique_imgs = torch.unique(img_indices)
        for img_idx in unique_imgs:
            if img_idx.item() not in image_freq_maps:
                continue

            # Get mask for this image
            mask = (img_indices == img_idx)
            batch_uvs = uv_coords[mask] # (N_i, 2)

            # Sample from frequency map
            # Grid sample expects (1, C, H, W) and (1, N, 1, 2) in [-1, 1]
            freq_map = image_freq_maps[img_idx.item()].to(device) # (H, W)
            if freq_map.ndim == 2:
                freq_map = freq_map.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

            # Remap UV [0, 1] -> [-1, 1]
            grid_uvs = (batch_uvs * 2.0 - 1.0).unsqueeze(0).unsqueeze(2) # (1, N_i, 1, 2)

            sampled = torch.nn.functional.grid_sample(
                freq_map, grid_uvs, mode='nearest', align_corners=True
            ) # (1, 1, N_i, 1)

            f_2d_candidates[mask] = sampled.view(-1)

        # B. Compute Geometry (Focals and Depths)
        # Gather point positions
        obs_points = sparse_points[pt_indices] # (K, 3)

        # Gather camera centers and focals
        # Note: Cameras object indexing returns a slice, we need explicit indexing
        cam_indices_cpu = img_indices.cpu()
        cam_centers = cameras.camera_to_worlds[cam_indices_cpu, :3, 3].to(device)

        # Average focal length (fx + fy) / 2
        fx = cameras.fx[cam_indices_cpu].to(device).flatten()
        fy = cameras.fy[cam_indices_cpu].to(device).flatten()
        focals = (fx + fy) / 2.0

        # Distance
        dists = torch.norm(obs_points - cam_centers, dim=-1) + 1e-6

        # C. Compute f_3D
        f_3d_candidates = f_2d_candidates * (focals / dists)

        # --- Step 2: Aggregate Median per Point ---
        # We need to group by `pt_indices` and take the median.
        # GPU implementation of Group-Median with variable group sizes is non-trivial.
        # Strategy: Sort by point_idx, then perform CPU-based segmentation for correctness.

        # Sort observations by point index
        sorted_pt_indices, sort_order = torch.sort(pt_indices)
        sorted_f3d = f_3d_candidates[sort_order]

        # Move to CPU for safe groupby-median loop
        sorted_pt_cpu = sorted_pt_indices.cpu().numpy()
        sorted_f3d_cpu = sorted_f3d.cpu().numpy()

        # Find boundaries where point index changes
        # unique_consecutive gives us the counts
        unique_pts, counts = np.unique(sorted_pt_cpu, return_counts=True)

        # Compute medians efficiently
        # We split the sorted values array based on cumsum of counts
        split_indices = np.cumsum(counts)[:-1]
        grouped_values = np.split(sorted_f3d_cpu, split_indices)

        # List comprehension is faster than python loop for simple stats
        medians_cpu = np.array([np.median(g) for g in grouped_values], dtype=np.float32)

        # Result: unique_pts (indices into sparse_points) and their median f_3d
        active_point_indices = torch.from_numpy(unique_pts).to(device).long()
        active_f3d_medians = torch.from_numpy(medians_cpu).to(device)

        # --- Step 3: Update Grid ---
        # Map to levels
        target_levels = self.freq_to_level(active_f3d_medians)

        # Get positions of these active points
        target_positions = sparse_points[active_point_indices]

        # Update
        self.update_max(target_positions, target_levels)

        CONSOLE.print(f"[bold green]LookCloser:[/bold green] Grid initialized. {len(active_point_indices)} voxels touched.")

    def update_step(
        self,
        step: int,
        positions: Tensor,
        rendered_depth: Tensor,
        focals: Tensor,
        patch_f2d: Tensor
    ):
        """
        Runtime update: re-evaluates frequency requirements based on dense neural rendering.
        Should be called periodically (e.g. every 1024 steps).

        Args:
            step: Current training step.
            positions: (N, 3) World positions (e.g. ray surface intersections).
            rendered_depth: (N, 1) Predicted depth.
            focals: (N, 1) Focal lengths.
            patch_f2d: (N, 1) Associated 2D frequency target from input metadata.
        """
        # Calculate f_3D
        # f_3D = f_2D * (focal / depth)

        safe_depth = rendered_depth + 1e-6
        f_3d = patch_f2d * (focals / safe_depth)

        # Map to level
        l_new = self.freq_to_level(f_3d)

        # Update Grid
        self.update_max(positions, l_new)