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
Temporal bounded Instant-NGP: ``F(x, y, z, t) -> density, color``.

Hypothesis H1: time is a fourth coordinate of the hash grid. The 3D nerfacc occupancy
grid is kept; occupancy density at a cell is approximated as the maximum density over a
set of training times (union over time). This is intentionally minimal - no deformation
field, canonical space, 4D occupancy, or temporal regularization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Type, cast

import nerfacc
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.temporal_ngp_field import TemporalNGPField
from nerfstudio.model_components.losses import scale_gradients_by_distance_squared
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.temporal_arm_sampler import TemporalARMSampler
from nerfstudio.model_components.lookcloser_grid import FrequencyGridManager
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class TemporalInstantNGPModelConfig(InstantNGPModelConfig):
    """Temporal Instant-NGP (4D hash grid) model config."""

    _target: Type = field(default_factory=lambda: TemporalInstantNGPModel)
    hypothesis: Literal["H1", "H2", "H3"] = "H1"
    """Which temporal architecture to use. H1: pure 4D hash grid. H2: concat(3D hash, 4D hash).
    H3: 3D hash features + scalar time (cheap control)."""
    static_num_levels: int = 16
    """(H2/H3) Number of hashgrid levels for the static 3D branch."""
    static_max_res: int = 512
    """(H2/H3) Max resolution for the static 3D branch."""
    static_log2_hashmap_size: int = 19
    """(H2/H3) Hashmap size for the static 3D branch."""
    appearance_embedding_dim: int = 0
    """Per-image appearance embedding dim (0 disables). instant-ngp-bounded uses 32."""
    use_average_appearance_embedding: bool = True
    """At eval, use the average appearance embedding (vs zeros) when appearance embedding is enabled."""
    # H1 defaults: 4D hash grids have higher collision pressure than 3D, so start larger.
    num_levels: int = 16
    """Number of hashgrid levels for the 4D base mlp."""
    features_per_level: int = 2
    """Number of hashgrid feature channels per level."""
    log2_hashmap_size: int = 20
    """Size of the hashmap for the 4D base mlp."""
    base_res: int = 16
    """Minimum resolution of the hashmap (applied to all 4 dims)."""
    max_res: int = 512
    """Maximum resolution of the hashmap (applied to all 4 dims)."""
    # Occupancy "union over time" controls.
    occ_time_chunk: int = 16
    """Number of times evaluated together per occupancy chunk (bounds VRAM)."""
    occ_points_chunk: int = 262144
    """Number of grid points evaluated together per occupancy chunk (bounds VRAM)."""
    occ_update_times_after_warmup: int = 32
    """Number of random training times used per occupancy update once there are many times."""
    occ_full_update_every: int = 16
    """Every Nth occupancy update evaluates all training times instead of a random subset."""
    occ_all_times_threshold: int = 64
    """If the number of training times is <= this, always use all times for occupancy."""
    occ_warmup_steps: int = 4096
    """nerfacc occupancy-grid warmup steps (full-grid updates). Higher = fewer occupancy holes /
    structural artifacts for dynamic scenes (nerfacc default 256 is too low here; LookCloser uses 4096).
    During warmup every cell is evaluated, unioned over ALL training times (frames)."""
    occ_update_n: int = 16
    """After warmup, run the occupancy update every N steps (nerfacc default 16)."""
    occ_binary_warmup_steps: int = 4096
    """Keep the occupancy binary grid fully occupied for the first N steps (prevents early pruning of
    thin/moving detail). Part of the LookCloser artifact-reduction recipe (4096)."""
    # --- Budget-aware Adaptive Ray Marching (ARM) — LookCloser's decisive artifact lever. ---
    enable_adaptive_ray_marching: bool = False
    """Use budget-aware ARM instead of fixed-step VolumetricSampler (off by default for back-compat)."""
    adaptive_coarse_step_size: Optional[float] = 0.00625
    """nerfacc occupancy coarse-traversal step for ARM (LookCloser's decisive value)."""
    adaptive_min_step_size: float = 1e-4
    """ARM fine step lower clamp."""
    adaptive_max_step_size: float = 0.1
    """ARM fine step upper clamp."""
    adaptive_min_frequency_level: float = 0.0
    adaptive_max_frequency_level: Optional[float] = None
    adaptive_interval_level_mode: Literal["midpoint", "max3"] = "midpoint"
    max_steps_per_ray: int = 1024
    """Per-ray sample budget for ARM (budget-aware dt scaling target)."""
    transmittance_threshold: float = 0.0
    """Passed to nerfacc as early_stop_eps during ARM coarse traversal."""
    arm_fallback_frequency_level: float = 8.0
    """Constant frequency level used by ARM when no frequency grid is present (ARM without freq-maps)."""
    arm_num_frequency_levels: int = 16
    arm_min_res: float = 16.0
    arm_max_res: float = 2048.0
    # --- LookCloser quality recipe: Charbonnier RGB + Mip-NeRF360 distortion loss ---
    reconstruction_loss_type: Literal["mse", "charbonnier"] = "mse"
    """RGB reconstruction loss. 'charbonnier' = sqrt((pred-gt)^2 + eps) (LookCloser leader; robust to
    outliers, better LPIPS than plain MSE). 'mse' keeps the original behaviour."""
    charbonnier_eps: float = 1e-4
    """Epsilon inside the Charbonnier sqrt (LookCloser uses 1e-4)."""
    distortion_loss_mult: float = 0.0
    """Weight of the Mip-NeRF360 distortion loss (LookCloser leader uses 0.01). 0 disables it. This is
    the critical missing piece vs the static leader — it regularizes sample spacing and removes floaters
    (LPIPS ~0.48 -> ~0.40 on the static scene)."""
    # --- Real frequency grid for ARM (baked offline; see LookCloser/scripts/bake_frequency_grid.py) ---
    frequency_grid_path: Optional[str] = None
    """Path to a baked FrequencyGridManager grid (.pt from bake_frequency_grid.py). When set, ARM queries
    this grid for a REAL per-scene frequency level instead of the constant ``arm_fallback_frequency_level``.
    None => byte-identical to the constant-fallback behaviour."""
    frequency_grid_resolution: int = 128
    """Voxel resolution of the frequency grid (must match the baked grid)."""
    frequency_grid_min_res: float = 16.0
    frequency_grid_max_res: float = 8192.0
    frequency_grid_num_levels: int = 16
    # --- LookCloser feature-reweighting (FR); H2 only, requires a frequency grid ---
    enable_feature_reweighting: bool = False
    """Dampen high hash-grid levels above the per-point grid frequency level (LookCloser Eq. 6). Needs
    frequency_grid_path set and hypothesis H2. Default False => byte-identical behaviour."""
    feature_reweighting_strength: float = 1.0
    """Blend toward identity: 1.0 = full FR, 0.0 = no reweighting."""


class TemporalInstantNGPModel(NGPModel):
    """Temporal bounded Instant-NGP model with a 4D (xyzt) hash field."""

    config: TemporalInstantNGPModelConfig
    field: TemporalNGPField

    def __init__(self, config: TemporalInstantNGPModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules (mirrors NGPModel but with a 4D temporal field)."""
        # Bounded scene only; scene contraction is not used for the temporal field.
        from nerfstudio.models.base_model import Model  # local import to avoid cycle confusion

        Model.populate_modules(self)

        self.field = TemporalNGPField(
            aabb=self.scene_box.aabb,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            base_res=self.config.base_res,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            features_per_level=self.config.features_per_level,
            num_layers_color=self.config.num_layers_color,
            hidden_dim_color=self.config.hidden_dim_color,
            rgb_output_activation=self.config.rgb_output_activation,
            hypothesis=self.config.hypothesis,
            static_num_levels=self.config.static_num_levels,
            static_max_res=self.config.static_max_res,
            static_log2_hashmap_size=self.config.static_log2_hashmap_size,
            num_images=self.num_train_data,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        if self.config.render_step_size is None:
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 1000

        # 3D occupancy grid kept unchanged; only the occ_eval_fn unions density over time.
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # Budget-aware ARM sampler (optional; LookCloser artifact lever). Uses a constant frequency
        # level so it runs without the frequency-map pipeline.
        self.arm_sampler = TemporalARMSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
            aabb=self.scene_box.aabb,
            min_res=self.config.arm_min_res,
            max_res=self.config.arm_max_res,
            num_levels=self.config.arm_num_frequency_levels,
            freq_level_fn=None,
            fallback_frequency_level=self.config.arm_fallback_frequency_level,
        )

        # Optional REAL frequency grid for ARM (baked offline). When a path is given, ARM queries this
        # grid for a per-scene frequency level; otherwise it keeps the constant-fallback behaviour.
        self.freq_grid: Optional[FrequencyGridManager] = None
        if self.config.frequency_grid_path is not None:
            from pathlib import Path as _Path

            grid_path = _Path(self.config.frequency_grid_path)
            if not grid_path.exists():
                CONSOLE.print(
                    f"[yellow]TemporalInstantNGP: frequency_grid_path {grid_path} not found; "
                    "ARM falls back to the constant frequency level."
                )
            else:
                self.freq_grid = FrequencyGridManager(
                    scene_box=self.scene_box,
                    resolution=self.config.frequency_grid_resolution,
                    num_levels=self.config.frequency_grid_num_levels,
                    min_res=self.config.frequency_grid_min_res,
                    max_res=self.config.frequency_grid_max_res,
                    enabled=True,
                )
                saved = torch.load(grid_path, map_location="cpu")
                grid_tensor = saved["grid"] if isinstance(saved, dict) and "grid" in saved else saved
                assert tuple(grid_tensor.shape) == tuple(self.freq_grid.grid.shape), (
                    f"Baked grid shape {tuple(grid_tensor.shape)} != model grid "
                    f"{tuple(self.freq_grid.grid.shape)} (check --resolution)."
                )
                self.freq_grid.grid.copy_(grid_tensor.to(self.freq_grid.grid.dtype))
                if isinstance(saved, dict):
                    for k in ("aabb_min_buf", "aabb_max_buf", "aabb_size_buf"):
                        if k in saved:
                            getattr(self.freq_grid, k).copy_(saved[k].to(getattr(self.freq_grid, k).dtype))
                self.arm_sampler.freq_level_fn = self.freq_grid.query
                nonzero = int((self.freq_grid.grid > 0).sum().item())
                CONSOLE.print(
                    f"TemporalInstantNGP: loaded baked frequency grid {tuple(grid_tensor.shape)} "
                    f"({nonzero} non-empty voxels, level range "
                    f"[{float(self.freq_grid.grid.min()):.1f},{float(self.freq_grid.grid.max()):.1f}]) "
                    "-> ARM uses real frequency levels."
                )

        # LookCloser feature-reweighting (FR): dampen high hash levels above the grid's frequency
        # level. Requires the frequency grid; H2 only (separate enc3/enc4 expose raw hash features).
        if self.config.enable_feature_reweighting:
            if self.freq_grid is None:
                CONSOLE.print(
                    "[yellow]TemporalInstantNGP: enable_feature_reweighting=True but no frequency grid "
                    "loaded; FR disabled."
                )
            elif self.config.hypothesis != "H2":
                CONSOLE.print(
                    f"[yellow]TemporalInstantNGP: FR only supported for H2 (got {self.config.hypothesis}); "
                    "FR disabled."
                )
            else:
                self.field.enable_feature_reweighting = True
                self.field.feature_reweighting_strength = self.config.feature_reweighting_strength
                self.field.freq_level_fn = self.freq_grid.query
                CONSOLE.print(
                    f"TemporalInstantNGP: feature-reweighting ENABLED (strength "
                    f"{self.config.feature_reweighting_strength}) on enc3/enc4."
                )

        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        self.rgb_loss = MSELoss()

        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Unique training times, populated lazily from the dataparser in get_training_callbacks.
        self._train_times: Optional[torch.Tensor] = None
        self._occ_update_count: int = 0

    def get_outputs(self, ray_bundle: RayBundle):  # type: ignore
        """Fixed-step (parent) or budget-aware ARM sampling; both expose packed spacing/weights so the
        distortion loss can be computed downstream."""
        assert self.field is not None
        num_rays = len(ray_bundle)

        if not self.config.enable_adaptive_ray_marching:
            with torch.no_grad():
                ray_samples, ray_indices = self.sampler(
                    ray_bundle=ray_bundle,
                    near_plane=self.config.near_plane,
                    far_plane=self.config.far_plane,
                    render_step_size=self.config.render_step_size,
                    alpha_thre=self.config.alpha_thre,
                    cone_angle=self.config.cone_angle,
                )
        else:
            coarse_step = float(self.config.adaptive_coarse_step_size or cast(float, self.config.render_step_size))
            with torch.no_grad():
                ray_samples, ray_indices, _stats = self.arm_sampler(
                    ray_bundle=ray_bundle,
                    render_step_size=coarse_step,
                    near_plane=self.config.near_plane,
                    far_plane=self.config.far_plane,
                    alpha_thre=self.config.alpha_thre,
                    early_stop_eps=self.config.transmittance_threshold,
                    cone_angle=self.config.cone_angle,
                    adaptive_min_step_size=self.config.adaptive_min_step_size,
                    adaptive_max_step_size=self.config.adaptive_max_step_size,
                    adaptive_min_frequency_level=self.config.adaptive_min_frequency_level,
                    adaptive_max_frequency_level=self.config.adaptive_max_frequency_level,
                    adaptive_interval_level_mode=self.config.adaptive_interval_level_mode,
                    max_steps_per_ray=self.config.max_steps_per_ray,
                )

        field_outputs = self.field(ray_samples)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB], weights=weights, ray_indices=ray_indices, num_rays=num_rays
        )
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays)
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
        }
        if self.training and self.config.distortion_loss_mult > 0:
            # Normalized [0,1] spacing for the Mip-NeRF360 distortion loss (matches LookCloser).
            near, far = self.config.near_plane, self.config.far_plane
            span = max(far - near, 1e-6)
            starts = ray_samples.frustums.starts[..., 0].detach()
            ends = ray_samples.frustums.ends[..., 0].detach()
            outputs["packed_spacing_starts"] = ((starts - near) / span).clamp(0.0, 1.0)
            outputs["packed_spacing_ends"] = ((ends - near) / span).clamp(0.0, 1.0)
            outputs["packed_weights"] = weights[..., 0]
            outputs["packed_ray_indices"] = ray_indices
            outputs["num_rays"] = num_rays
        return outputs

    @staticmethod
    def _packed_distortion_loss(
        spacing_starts: torch.Tensor,
        spacing_ends: torch.Tensor,
        weights: torch.Tensor,
        ray_indices: torch.Tensor,
        num_rays: int,
    ) -> torch.Tensor:
        """Linear-time Mip-NeRF 360 distortion loss for packed, ray-sorted samples (ported from
        the static LookCloser leader)."""
        if weights.numel() == 0:
            return weights.new_zeros((num_rays, 1))
        starts = spacing_starts.reshape(-1)
        ends = spacing_ends.reshape(-1)
        w = weights.reshape(-1)
        mid = 0.5 * (starts + ends)
        interval = (ends - starts).clamp_min(0.0)

        packed = nerfacc.pack_info(ray_indices, num_rays)
        first = packed[ray_indices, 0]
        global_prefix_w = torch.cumsum(w, dim=0) - w
        global_prefix_wm = torch.cumsum(w * mid, dim=0) - w * mid

        first_prefix_w = torch.zeros_like(w)
        first_prefix_wm = torch.zeros_like(w)
        nonempty = packed[:, 1] > 0
        first_indices = packed[nonempty, 0]
        first_prefix_w[first_indices] = global_prefix_w[first_indices]
        first_prefix_wm[first_indices] = global_prefix_wm[first_indices]
        base_prefix_w = first_prefix_w[first]
        base_prefix_wm = first_prefix_wm[first]

        prefix_w = global_prefix_w - base_prefix_w
        prefix_wm = global_prefix_wm - base_prefix_wm
        inter = 2.0 * w * (mid * prefix_w - prefix_wm)
        intra = (w**2) * interval / 3.0

        per_sample = inter + intra
        per_ray = torch.zeros((num_rays,), dtype=weights.dtype, device=weights.device)
        per_ray.scatter_add_(0, ray_indices, per_sample)
        return per_ray[:, None]

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        if self.config.reconstruction_loss_type == "charbonnier":
            rgb_loss = torch.sqrt((pred_rgb - image) ** 2 + self.config.charbonnier_eps).mean()
        elif self.config.loss_type == "instant_ngp_huber":
            rgb_loss = torch.nn.functional.huber_loss(pred_rgb, image, delta=0.1, reduction="mean") / 5.0
        else:
            rgb_loss = self.rgb_loss(image, pred_rgb)
        loss_dict = {"rgb_loss": rgb_loss}
        if self.config.distortion_loss_mult > 0 and "packed_weights" in outputs:
            distortion = self._packed_distortion_loss(
                spacing_starts=outputs["packed_spacing_starts"],
                spacing_ends=outputs["packed_spacing_ends"],
                weights=outputs["packed_weights"],
                ray_indices=outputs["packed_ray_indices"],
                num_rays=outputs["num_rays"],
            ).mean()
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * distortion
        return loss_dict

    def _select_occ_times(self, device: torch.device) -> torch.Tensor:
        """Choose which training times to union over for this occupancy update."""
        assert self._train_times is not None
        times = self._train_times.to(device)
        n = times.numel()
        if n <= self.config.occ_all_times_threshold:
            chosen = times
        elif self._occ_update_count % self.config.occ_full_update_every == 0:
            # Periodic (and warmup, since count starts at 0) full union over all times.
            chosen = times
        else:
            k = min(self.config.occ_update_times_after_warmup, n)
            idx = torch.randperm(n, device=device)[:k]
            chosen = times[idx]
        self._occ_update_count += 1
        return chosen

    @torch.no_grad()
    def _occ_eval_fn(self, x: torch.Tensor) -> torch.Tensor:
        """occ_density(x) = max_t sigma(x, t), evaluated in point/time chunks to bound VRAM."""
        times = self._select_occ_times(x.device)  # [T]
        render_step_size = cast(float, self.config.render_step_size)
        num_points = x.shape[0]
        max_sigma = torch.zeros((num_points, 1), device=x.device)

        p_chunk = self.config.occ_points_chunk
        t_chunk = self.config.occ_time_chunk
        for p0 in range(0, num_points, p_chunk):
            x_block = x[p0 : p0 + p_chunk]  # [P, 3]
            p = x_block.shape[0]
            block_max = torch.zeros((p, 1), device=x.device)
            for t0 in range(0, times.numel(), t_chunk):
                t_block = times[t0 : t0 + t_chunk]  # [c]
                c = t_block.numel()
                x_rep = x_block.repeat_interleave(c, dim=0)  # [P*c, 3]
                t_rep = t_block.repeat(p).unsqueeze(-1)  # [P*c, 1]
                sigma = self.field.density_fn(x_rep, times=t_rep).view(p, c)
                block_max = torch.maximum(block_max, sigma.max(dim=1).values[..., None])
            max_sigma[p0 : p0 + p_chunk] = block_max
        return max_sigma * render_step_size

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        # Source the unique training times from the dataparser cameras (per-frame `time`).
        train_times: Optional[torch.Tensor] = None
        pipeline = training_callback_attributes.pipeline
        if pipeline is not None:
            dpo = getattr(pipeline.datamanager, "train_dataparser_outputs", None)
            if dpo is not None and dpo.cameras.times is not None:
                train_times = torch.unique(dpo.cameras.times.reshape(-1).float())
        if train_times is None or train_times.numel() == 0:
            # Single-frame / static fallback: t = 0.0
            train_times = torch.zeros(1)
        self._train_times = train_times
        self._occ_update_count = 0

        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=self._occ_eval_fn,
                warmup_steps=self.config.occ_warmup_steps,
                n=self.config.occ_update_n,
            )
            # Binary warmup: keep the grid fully occupied early so thin/moving detail is not pruned.
            if step < self.config.occ_binary_warmup_steps:
                self.occupancy_grid.binaries.fill_(True)

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]
