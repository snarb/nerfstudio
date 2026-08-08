"""
LookCloser (FA-NeRF) Model Implementation.
Integrates Frequency-Aware Neural Radiance Fields with Adaptive Ray Marching.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type

import nerfacc
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.lookcloser_field import LookCloserField, TCNNNetworkJITScope
from nerfstudio.model_components.lookcloser_grid import FrequencyGridManager
from nerfstudio.model_components.lookcloser_occupancy import stable_ema_max_update_
from nerfstudio.model_components.lookcloser_samplers import FrequencyAwareSamplerStats, FrequencyAwareVolumetricSampler
from nerfstudio.model_components.losses import nerfstudio_distortion_loss, scale_gradients_by_distance_squared
from nerfstudio.model_components import renderers as renderer_module
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.lookcloser_rng import fork_seeded_rng
from nerfstudio.utils.hdr import hdr_display_preview, scene_linear_to_pq


@dataclass
class LookCloserModelConfig(ModelConfig):
    """Configuration for LookCloser Model."""

    _target: Type = field(default_factory=lambda: LookCloserModel)

    training_seed: int = 42
    """Recorded campaign seed used to derive the occupancy RNG stream."""

    independent_rng_streams: bool = False
    """Isolate occupancy candidate sampling from pixel/FAS and frequency-grid RNG."""

    # Grid parameters
    enable_frequency_grid: bool = True
    """Whether to use the 3D frequency grid; disabled runs use fallback_frequency_level."""

    grid_resolution: int = 128
    """Resolution of the frequency voxel grid."""

    occupancy_grid_levels: int = 1
    """Number of nerfacc occupancy-grid AABB cascade levels."""

    num_frequency_levels: int = 16
    """Number of discrete frequency levels."""

    min_res: float = 16.0
    """Minimum resolution (N_min)."""

    max_res: Optional[float] = 8192.0
    """Maximum resolution (N_max). If unset, uses max_res_base * scene_size. 8192 matches the HD frequency maps generated for this scene."""

    max_res_base: float = 2048.0
    """Per-scene-size maximum hash-grid resolution multiplier."""

    fallback_frequency_level: float = 0.0
    """Frequency level returned when the frequency grid is disabled."""

    # Field / feature re-weighting settings
    enable_feature_reweighting: bool = True
    """Whether to apply LookCloser Eq. 6 frequency-aware feature re-weighting."""

    feature_reweighting_strength: float = 1.0
    """Blend strength for feature re-weighting; 1.0 preserves Eq. 6, 0.0 is identity."""

    geo_feat_dim: int = 15
    """Geometry feature dimension emitted by the density MLP."""

    hash_features_per_level: int = 2
    """Feature channels per hash-grid level."""

    log2_hashmap_size: int = 23
    """Hash table size exponent for the 3D field hash grid."""

    field_hidden_dim: int = 64
    """Hidden width for LookCloser field MLPs."""

    geo_num_layers: int = 1
    """Hidden layer count for the density/geometry MLP."""

    color_num_layers: int = 2
    """Hidden layer count for the color MLP."""

    appearance_embedding_dim: int = 0
    """Optional per-training-image appearance embedding dimension."""

    sh_degree: int = 4
    """Spherical harmonics degree passed to tinycudann for view direction encoding."""

    tcnn_network_jit: bool = False
    """Opt in to tiny-cuda-nn runtime JIT fusion for the network(s) selected by the JIT scope."""

    tcnn_network_jit_scope: TCNNNetworkJITScope = "both"
    """TCNN field network(s) affected by initial and live JIT enablement."""

    # Loss weights
    distortion_loss_mult: float = 0.01
    """Multiplier for Mip-NeRF 360 distortion loss."""

    depth_loss_mult: float = 0.001
    """Multiplier for sparse depth supervision."""

    depth_loss_steps: int = 5000
    """Number of steps to apply depth loss."""

    # Marching settings
    ray_sampling_mode: Literal["auto", "adaptive", "occupancy", "fixed"] = "adaptive"
    """Ray sampling mode; auto preserves enable_adaptive_ray_marching backward compatibility."""

    enable_adaptive_ray_marching: bool = True
    """Whether to use frequency-guided adaptive step sizes."""

    max_steps_per_ray: int = 1024
    """Maximum number of steps per ray for adaptive marching."""

    adaptive_min_step_size: float = 1e-4
    """Minimum adaptive ray marching step size."""

    adaptive_max_step_size: float = 0.1
    """Maximum adaptive ray marching step size."""

    adaptive_coarse_step_size: Optional[float] = 0.00625
    """Coarse nerfacc traversal step for adaptive marching; unset uses adaptive_max_step_size."""

    adaptive_min_frequency_level: float = 0.0
    """Minimum frequency-grid level used only for adaptive interval sizing."""

    adaptive_max_frequency_level: Optional[float] = None
    """Maximum frequency-grid level used only for adaptive interval sizing."""

    adaptive_warmup_steps: int = 4096
    """Use the leader's fixed 256-sample warmup before adaptive marching."""

    adaptive_fixed_fallback_samples_per_ray: int = 0
    """Uniform fallback samples per ray appended to adaptive ARM samples; 0 preserves pure ARM."""

    adaptive_interval_level_mode: Literal["midpoint", "max3"] = "midpoint"
    """Frequency level query mode for ARM interval subdivision."""

    corrected_arm_allocator: bool = False
    """Use minimum-one/largest-remainder capping with deterministic interval merging."""

    transmittance_threshold: float = 0.0
    """Ray termination threshold for remaining transmittance."""

    render_step_size: Optional[float] = None
    """Step size used when updating the nerfacc occupancy grid."""

    render_step_size_mult: float = 1.0
    """Multiplier for the default scene-diagonal/1000 coarse traversal step size."""

    occupancy_occ_thre: float = 1e-2
    """Nerfacc occupancy binarization threshold cap."""

    occupancy_ema_decay: float = 0.95
    """Nerfacc max-with-decay occupancy update factor."""

    occupancy_warmup_steps: int = 4096
    """Nerfacc dense all-cells occupancy update warmup steps."""

    occupancy_update_interval: int = 16
    """Run nerfacc occupancy updates every N training steps."""

    occupancy_update_step_size: Optional[float] = None
    """Scale density values for occupancy updates; unset uses render_step_size."""

    occupancy_thre_clamp_mult: float = 1.0
    """Multiplier on mean(occs) in custom threshold clamp; 1.0 preserves nerfacc default."""

    occupancy_dilation_radius: int = 0
    """Voxel dilation radius applied to binary occupancy after every grid update."""

    occupancy_binary_warmup_steps: int = 4096
    """Keep occupancy binaries fully occupied for this many initial steps to avoid cold-start empty grids."""

    occupancy_fixed_fallback_samples_per_ray: int = 0
    """Uniform safety samples per ray appended to occupancy traversal; 0 preserves pure occupancy traversal."""

    stable_occupancy_reduction: bool = True
    """Reduce duplicate occupancy candidates by max; disable only for legacy forensic controls."""

    occupancy_diagnostics: bool = True
    """Collect occupancy-grid reduction metrics after updates; disable to remove their hot-path overhead."""

    near_plane: float = 0.01
    """Near plane passed to nerfacc adaptive traversal."""

    far_plane: float = 1000.0
    """Far plane passed to nerfacc adaptive traversal."""

    alpha_thre: float = 0.0
    """Opacity threshold for nerfacc adaptive traversal visibility pruning."""

    cone_angle: float = 0.0
    """Cone angle for nerfacc adaptive traversal."""

    use_gradient_scaling: bool = False
    """Whether to scale field-output gradients by squared distance."""

    fixed_num_samples_per_ray: int = 256
    """Number of uniform samples per ray when adaptive ray marching is disabled."""

    background_color: Literal["random", "last_sample", "black", "white"] = "black"
    """Background color strategy."""

    reconstruction_loss_type: Literal[
        "charbonnier",
        "mse",
        "huber",
        "linear_l1",
        "rawnerf_weighted_l2",
        "linear_pq",
        "pq_l1",
        "eag_pq_dssim",
    ] = "charbonnier"
    """RGB reconstruction loss for LookCloser training."""

    huber_delta: float = 0.1
    """Delta used by Huber RGB reconstruction loss."""

    rgb_output_parameterization: Literal["sigmoid", "linear_softplus", "pq_code"] = "sigmoid"
    """Color-head representation; HDR campaigns use an explicitly selected non-sigmoid mode."""

    hdr_linear_scale: Optional[float] = None
    """Scene-linear normalization scale; unset reads deterministic train-split calibration metadata."""

    hdr_initial_radiance: Optional[float] = None
    """Initial scene-linear radiance represented by zero color logits."""

    hdr_softplus_beta: float = 1.0
    """Softplus beta for unbounded non-negative scene-linear radiance."""

    pq_nits_per_scene_unit: Optional[float] = None
    """Dataset-wide scene-linear to display-nits scale; unset uses train-split calibration."""

    pq_black_nits: float = 0.005
    """Black floor used inside PQ transforms."""

    pq_peak_nits: float = 10_000.0
    """Peak represented by the PQ-code head."""

    pq_code_temperature: float = 1.0
    """Sigmoid temperature for the PQ-code output ablation."""

    rawnerf_epsilon: float = 1e-3
    """Stop-gradient denominator floor in normalized scene-linear units."""

    rawnerf_grad_clip: float = 0.1
    """Recorded gradient clipping value for RawNeRF campaign runners."""

    pq_linear_anchor_weight: float = 0.0
    """Optional normalized linear-L1 anchor added to PQ reconstruction losses."""

    eag_dssim_weight: float = 0.2
    """DSSIM weight for the EAG-PT-inspired patch loss."""

    eag_patch_size: int = 11
    """Contiguous patch width required by EAG-PT-inspired DSSIM."""

    eag_edge_weight: float = 0.0
    """PQ finite-difference edge consistency added to the EAG patch loss."""


class LookCloserModel(Model):
    """
    LookCloser: Frequency-Aware NeRF with Adaptive Ray Marching.

    This model maintains a 3D frequency grid that guides both the feature
    encoding capacity (via the Field) and the rendering step size (via Adaptive Ray Marching).
    """

    config: LookCloserModelConfig

    @staticmethod
    def _render_packed_black(
        rgb_samples: torch.Tensor,
        weights: torch.Tensor,
        ray_samples: RaySamples,
        ray_indices: torch.Tensor,
        num_rays: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render packed samples while sharing the black-background accumulation."""
        scalar_weights = weights[..., 0]
        rgb = nerfacc.accumulate_along_rays(
            scalar_weights, values=rgb_samples, ray_indices=ray_indices, n_rays=num_rays
        )
        accumulation = nerfacc.accumulate_along_rays(
            scalar_weights, values=None, ray_indices=ray_indices, n_rays=num_rays
        )
        steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) * 0.5
        depth = nerfacc.accumulate_along_rays(
            scalar_weights, values=steps, ray_indices=ray_indices, n_rays=num_rays
        )
        depth = depth / (accumulation + 1e-10)
        depth = torch.clip(depth, steps.min(), steps.max())
        return rgb, accumulation, depth
    field: LookCloserField
    freq_grid: FrequencyGridManager

    def populate_modules(self):
        """Set up fields and modules."""
        super().populate_modules()
        if self.config.num_frequency_levels < 2:
            raise ValueError("num_frequency_levels must be >= 2.")
        scene_size = float(torch.max(self.scene_box.aabb[1] - self.scene_box.aabb[0]).item())
        max_res = (
            float(self.config.max_res)
            if self.config.max_res is not None
            else float(round(self.config.max_res_base * scene_size))
        )
        if self.config.min_res <= 0 or max_res <= self.config.min_res:
            raise ValueError("Expected 0 < min_res < max_res.")
        if self.config.grid_resolution <= 0:
            raise ValueError("grid_resolution must be > 0.")
        if self.config.occupancy_grid_levels <= 0:
            raise ValueError("occupancy_grid_levels must be > 0.")
        if self.config.appearance_embedding_dim < 0:
            raise ValueError("appearance_embedding_dim must be >= 0.")
        if self.config.feature_reweighting_strength < 0:
            raise ValueError("feature_reweighting_strength must be >= 0.")
        if self.config.huber_delta <= 0:
            raise ValueError("huber_delta must be > 0.")
        if self.config.hdr_softplus_beta <= 0 or self.config.pq_code_temperature <= 0:
            raise ValueError("HDR activation beta/temperature must be positive.")
        if self.config.rawnerf_epsilon <= 0 or self.config.rawnerf_grad_clip <= 0:
            raise ValueError("RawNeRF epsilon and grad clip must be positive.")
        if not 0 <= self.config.eag_dssim_weight < 1:
            raise ValueError("eag_dssim_weight must be in [0, 1).")
        if self.config.eag_edge_weight < 0:
            raise ValueError("eag_edge_weight must be non-negative.")
        if self.config.eag_patch_size <= 1:
            raise ValueError("eag_patch_size must be > 1.")

        calibration = dict(self.kwargs.get("metadata", {}).get("hdr_calibration", {}))
        hdr_enabled = self.config.rgb_output_parameterization != "sigmoid"
        if hdr_enabled and not calibration and (
            self.config.hdr_linear_scale is None
            or self.config.hdr_initial_radiance is None
            or self.config.pq_nits_per_scene_unit is None
        ):
            raise ValueError(
                "HDR output requires train-split calibration metadata or explicit hdr_linear_scale, "
                "hdr_initial_radiance, and pq_nits_per_scene_unit."
            )
        self.hdr_linear_scale = float(
            self.config.hdr_linear_scale if self.config.hdr_linear_scale is not None else calibration.get("linear_scale", 1.0)
        )
        self.hdr_initial_radiance = float(
            self.config.hdr_initial_radiance
            if self.config.hdr_initial_radiance is not None
            else calibration.get("initial_radiance", 0.5)
        )
        self.pq_nits_per_scene_unit = float(
            self.config.pq_nits_per_scene_unit
            if self.config.pq_nits_per_scene_unit is not None
            else calibration.get("nits_per_scene_unit", 100.0)
        )
        if self.config.reconstruction_loss_type == "pq_l1" and self.config.rgb_output_parameterization != "pq_code":
            raise ValueError("pq_l1 requires rgb_output_parameterization='pq_code'.")
        if self.config.reconstruction_loss_type in {"linear_pq", "eag_pq_dssim"} and self.config.rgb_output_parameterization == "pq_code":
            raise ValueError("linear_pq/eag_pq_dssim require a linear RGB output parameterization.")
        if self.config.fixed_num_samples_per_ray <= 0:
            raise ValueError("fixed_num_samples_per_ray must be > 0.")
        if self.config.adaptive_min_step_size <= 0 or self.config.adaptive_max_step_size <= 0:
            raise ValueError("adaptive step sizes must be > 0.")
        if self.config.adaptive_max_step_size < self.config.adaptive_min_step_size:
            raise ValueError("adaptive_max_step_size must be >= adaptive_min_step_size.")
        if self.config.adaptive_coarse_step_size is not None and self.config.adaptive_coarse_step_size <= 0:
            raise ValueError("adaptive_coarse_step_size must be > 0.")
        if self.config.adaptive_min_frequency_level < 0:
            raise ValueError("adaptive_min_frequency_level must be >= 0.")
        if self.config.adaptive_max_frequency_level is not None:
            if self.config.adaptive_max_frequency_level < 0:
                raise ValueError("adaptive_max_frequency_level must be >= 0.")
            if self.config.adaptive_max_frequency_level < self.config.adaptive_min_frequency_level:
                raise ValueError("adaptive_max_frequency_level must be >= adaptive_min_frequency_level.")
        if self.config.adaptive_warmup_steps < 0:
            raise ValueError("adaptive_warmup_steps must be >= 0.")
        if self.config.adaptive_fixed_fallback_samples_per_ray < 0:
            raise ValueError("adaptive_fixed_fallback_samples_per_ray must be >= 0.")
        if self.config.occupancy_fixed_fallback_samples_per_ray < 0:
            raise ValueError("occupancy_fixed_fallback_samples_per_ray must be >= 0.")
        if self.config.render_step_size_mult <= 0:
            raise ValueError("render_step_size_mult must be > 0.")
        if self.config.occupancy_occ_thre <= 0:
            raise ValueError("occupancy_occ_thre must be > 0.")
        if not 0 < self.config.occupancy_ema_decay <= 1:
            raise ValueError("Expected 0 < occupancy_ema_decay <= 1.")
        if self.config.occupancy_warmup_steps < 0:
            raise ValueError("occupancy_warmup_steps must be >= 0.")
        if self.config.occupancy_update_interval <= 0:
            raise ValueError("occupancy_update_interval must be > 0.")
        if self.config.occupancy_update_step_size is not None and self.config.occupancy_update_step_size <= 0:
            raise ValueError("occupancy_update_step_size must be > 0.")
        if self.config.occupancy_thre_clamp_mult <= 0:
            raise ValueError("occupancy_thre_clamp_mult must be > 0.")
        if self.config.occupancy_dilation_radius < 0:
            raise ValueError("occupancy_dilation_radius must be >= 0.")
        if self.config.near_plane < 0 or self.config.far_plane <= self.config.near_plane:
            raise ValueError("Expected 0 <= near_plane < far_plane.")

        # 1. Frequency Grid Manager (Persistent State)
        self.freq_grid = FrequencyGridManager(
            scene_box=self.scene_box,
            resolution=self.config.grid_resolution,
            num_levels=self.config.num_frequency_levels,
            min_res=self.config.min_res,
            max_res=max_res,
            enabled=self.config.enable_frequency_grid,
            fallback_level=self.config.fallback_frequency_level,
        )

        # 2. LookCloser Field (Frequency-Aware)
        self.field = LookCloserField(
            aabb=self.scene_box.aabb,
            freq_grid=self.freq_grid,
            num_levels=self.config.num_frequency_levels,
            min_res=self.config.min_res,
            max_res=max_res,
            geo_feat_dim=self.config.geo_feat_dim,
            log2_hashmap_size=self.config.log2_hashmap_size,
            features_per_level=self.config.hash_features_per_level,
            hidden_dim=self.config.field_hidden_dim,
            geo_num_layers=self.config.geo_num_layers,
            color_num_layers=self.config.color_num_layers,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            num_images=self.num_train_data,
            sh_degree=self.config.sh_degree,
            tcnn_network_jit=self.config.tcnn_network_jit,
            tcnn_network_jit_scope=self.config.tcnn_network_jit_scope,
            enable_feature_reweighting=self.config.enable_feature_reweighting,
            feature_reweighting_strength=self.config.feature_reweighting_strength,
            rgb_output_parameterization=self.config.rgb_output_parameterization,
            hdr_linear_scale=self.hdr_linear_scale,
            hdr_initial_radiance=self.hdr_initial_radiance,
            pq_nits_per_scene_unit=self.pq_nits_per_scene_unit,
            pq_black_nits=self.config.pq_black_nits,
            pq_peak_nits=self.config.pq_peak_nits,
            hdr_softplus_beta=self.config.hdr_softplus_beta,
            pq_code_temperature=self.config.pq_code_temperature,
        )

        # 3. Renderers
        self.renderer_rgb = RGBRenderer(
            background_color=self.config.background_color,
            clamp_output=self.config.rgb_output_parameterization == "sigmoid",
        )
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box, near_plane=self.config.near_plane)

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)
        if self.config.render_step_size is None:
            scene_diag = torch.linalg.norm(self.scene_box.aabb[1] - self.scene_box.aabb[0]).item()
            self.config.render_step_size = scene_diag / 1000.0 * self.config.render_step_size_mult
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.occupancy_grid_levels,
        )
        # A cross-frame full resume may preserve optimizer/frequency state while
        # restarting occupancy. This runtime offset makes warmup local to that
        # resume without changing ordinary absolute-step behavior.
        self._occupancy_warmup_start_step = 0
        self.adaptive_sampler = FrequencyAwareVolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            freq_grid=self.freq_grid,
            density_fn=self.field.density_fn,
        )

        # Metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.current_train_step = 0
        self._last_occupancy_binaries: Optional[torch.Tensor] = None
        self._last_occupancy_stats: Dict[str, float] = {}

    def _resolved_ray_sampling_mode(self) -> str:
        if self.config.ray_sampling_mode != "auto":
            return self.config.ray_sampling_mode
        return "adaptive" if self.config.enable_adaptive_ray_marching else "fixed"

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)
        if self._resolved_ray_sampling_mode() == "fixed":
            return callbacks

        def update_occupancy_grid(step: int):
            assert self.config.render_step_size is not None
            occupancy_step = max(
                step - int(getattr(self, "_occupancy_warmup_start_step", 0)), 0
            )
            update_step_size = (
                float(self.config.occupancy_update_step_size)
                if self.config.occupancy_update_step_size is not None
                else float(self.config.render_step_size)
            )
            update_interval = int(self.config.occupancy_update_interval)
            if self.config.stable_occupancy_reduction and occupancy_step % update_interval != 0:
                return

            def apply_update() -> None:
                if self.config.stable_occupancy_reduction:
                    self._stable_update_occupancy_grid(
                        step=occupancy_step,
                        occ_eval_fn=lambda x: self.field.density_fn(x) * update_step_size,
                    )
                else:
                    self.occupancy_grid.update_every_n_steps(
                        step=occupancy_step,
                        occ_eval_fn=lambda x: self.field.density_fn(x) * update_step_size,
                        occ_thre=float(self.config.occupancy_occ_thre),
                        ema_decay=float(self.config.occupancy_ema_decay),
                        warmup_steps=int(self.config.occupancy_warmup_steps),
                        n=update_interval,
                    )

            if self.config.independent_rng_streams:
                with fork_seeded_rng(self.config.training_seed, "occupancy", step, self.device):
                    apply_update()
            else:
                apply_update()
            if occupancy_step % update_interval != 0:
                return
            self._postprocess_occupancy_grid(occupancy_step)

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            )
        )
        return callbacks

    @torch.no_grad()
    def _stable_update_occupancy_grid(
        self,
        step: int,
        occ_eval_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Nerfacc 0.5.2 occupancy update with stable duplicate-ID reduction."""

        grid = self.occupancy_grid
        if step < int(self.config.occupancy_warmup_steps):
            level_indices = grid._get_all_cells()
        else:
            level_indices = grid._sample_uniform_and_occupied_cells(grid.cells_per_lvl // 4)

        for level, indices in enumerate(level_indices):
            grid_coords = grid.grid_coords[indices]
            positions = (
                grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)
            ) / grid.resolution
            positions = grid.aabbs[level, :3] + positions * (
                grid.aabbs[level, 3:] - grid.aabbs[level, :3]
            )
            candidates = occ_eval_fn(positions).squeeze(-1)
            cell_ids = level * grid.cells_per_lvl + indices
            stable_ema_max_update_(
                grid.occs,
                cell_ids,
                candidates,
                float(self.config.occupancy_ema_decay),
            )

        grid.binaries = (
            grid.occs > torch.clamp(grid.occs.mean(), max=float(self.config.occupancy_occ_thre))
        ).view(grid.binaries.shape)

    def _postprocess_occupancy_grid(self, step: Optional[int] = None) -> None:
        grid = self.occupancy_grid
        diagnostics = bool(self.config.occupancy_diagnostics)
        clamp_mult = float(self.config.occupancy_thre_clamp_mult)
        occ_mean: Optional[float] = None
        if diagnostics or clamp_mult != 1.0:
            occ_mean = float(grid.occs.mean().item())

        effective_thre: Optional[float] = None
        if clamp_mult != 1.0:
            assert occ_mean is not None
            effective_thre = min(
                occ_mean * clamp_mult,
                float(self.config.occupancy_occ_thre),
            )
            grid.binaries = (grid.occs > effective_thre).view(grid.binaries.shape)
        if self.config.occupancy_dilation_radius > 0:
            self._dilate_occ_binaries(int(self.config.occupancy_dilation_radius))
        if step is not None and step < int(self.config.occupancy_binary_warmup_steps):
            grid.binaries.fill_(True)

        if not diagnostics:
            # These fields are diagnostic-only, non-persistent state. Clear any
            # values left by a runtime policy switch without cloning the grid.
            self._last_occupancy_binaries = None
            self._last_occupancy_stats = {}
            return

        assert occ_mean is not None
        occ_max = float(grid.occs.max().item())
        default_thre = min(occ_mean, float(self.config.occupancy_occ_thre))
        if effective_thre is None:
            effective_thre = default_thre
        binaries = grid.binaries.detach()
        previous = self._last_occupancy_binaries
        flipped_on = 0.0
        flipped_off = 0.0
        if previous is not None and previous.shape == binaries.shape:
            flipped_on = float((binaries & ~previous).sum().item())
            flipped_off = float((~binaries & previous).sum().item())
        self._last_occupancy_binaries = binaries.clone()
        level_dims = tuple(range(1, binaries.ndim))
        ratios = binaries.float().mean(dim=level_dims)
        self._last_occupancy_stats = {
            "occupancy_ratio": float(binaries.float().mean().item()),
            "occupancy_ratio_level0": float(ratios[0].item()) if ratios.numel() > 0 else 0.0,
            "occupancy_occs_mean": occ_mean,
            "occupancy_occs_max": occ_max,
            "occupancy_effective_threshold": effective_thre,
            "occupancy_default_threshold": default_thre,
            "occupancy_effective_alpha_thre": min(float(self.config.alpha_thre), occ_mean),
            "occupancy_flipped_on": flipped_on,
            "occupancy_flipped_off": flipped_off,
        }

    def _dilate_occ_binaries(self, radius: int) -> None:
        if radius <= 0:
            return
        binaries = self.occupancy_grid.binaries.float()[:, None]
        kernel_size = 2 * radius + 1
        self.occupancy_grid.binaries = (
            F.max_pool3d(binaries, kernel_size=kernel_size, stride=1, padding=radius)[:, 0] > 0
        )

    def _fallback_ray_samples(self, ray_bundle: RayBundle, samples_per_ray: int) -> Tuple[RaySamples, torch.Tensor]:
        device = ray_bundle.origins.device
        num_rays = len(ray_bundle)
        if samples_per_ray <= 0 or num_rays == 0:
            empty = torch.zeros((0, 1), dtype=ray_bundle.origins.dtype, device=device)
            return (
                RaySamples(
                    frustums=Frustums(
                        origins=torch.zeros((0, 3), dtype=ray_bundle.origins.dtype, device=device),
                        directions=torch.zeros((0, 3), dtype=ray_bundle.directions.dtype, device=device),
                        starts=empty,
                        ends=empty,
                        pixel_area=empty,
                    ),
                    camera_indices=torch.zeros((0, 1), dtype=torch.long, device=device),
                    deltas=empty,
                    spacing_starts=empty,
                    spacing_ends=empty,
                ),
                torch.zeros((0,), dtype=torch.long, device=device),
            )

        nears = ray_bundle.nears.reshape(-1) if ray_bundle.nears is not None else torch.full((num_rays,), self.config.near_plane, device=device)
        fars = ray_bundle.fars.reshape(-1) if ray_bundle.fars is not None else torch.full((num_rays,), self.config.far_plane, device=device)
        valid_rays = torch.isfinite(nears) & torch.isfinite(fars) & (fars > nears)
        if not valid_rays.any():
            return self._fallback_ray_samples(ray_bundle, 0)

        edges = torch.linspace(0.0, 1.0, samples_per_ray + 1, device=device, dtype=ray_bundle.origins.dtype)
        valid_indices = torch.nonzero(valid_rays, as_tuple=False).flatten()
        starts = nears[valid_indices, None] + (fars[valid_indices] - nears[valid_indices])[:, None] * edges[:-1][None, :]
        ends = nears[valid_indices, None] + (fars[valid_indices] - nears[valid_indices])[:, None] * edges[1:][None, :]
        ray_indices = valid_indices[:, None].expand(-1, samples_per_ray).reshape(-1)
        starts_flat = starts.reshape(-1)
        ends_flat = ends.reshape(-1)
        origins = ray_bundle.origins[ray_indices]
        directions = ray_bundle.directions[ray_indices]
        camera_indices = ray_bundle.camera_indices
        if camera_indices is not None:
            camera_indices = camera_indices.contiguous()[ray_indices]

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=directions,
                starts=starts_flat[..., None],
                ends=ends_flat[..., None],
                pixel_area=ray_bundle[ray_indices].pixel_area,
            ),
            camera_indices=camera_indices,
            deltas=(ends_flat - starts_flat)[..., None],
            spacing_starts=((starts_flat - nears[ray_indices]) / (fars[ray_indices] - nears[ray_indices]).clamp_min(1e-6))[..., None],
            spacing_ends=((ends_flat - nears[ray_indices]) / (fars[ray_indices] - nears[ray_indices]).clamp_min(1e-6))[..., None],
        )
        if ray_bundle.times is not None:
            ray_samples.times = ray_bundle.times[ray_indices]
        return ray_samples, ray_indices

    @staticmethod
    def _concat_packed_ray_samples(
        first_samples: RaySamples,
        first_indices: torch.Tensor,
        second_samples: RaySamples,
        second_indices: torch.Tensor,
    ) -> Tuple[RaySamples, torch.Tensor]:
        if second_indices.numel() == 0:
            return first_samples, first_indices
        if first_indices.numel() == 0:
            return second_samples, second_indices

        ray_indices = torch.cat([first_indices, second_indices], dim=0)
        starts = torch.cat([first_samples.frustums.starts[..., 0], second_samples.frustums.starts[..., 0]], dim=0)
        ends = torch.cat([first_samples.frustums.ends[..., 0], second_samples.frustums.ends[..., 0]], dim=0)
        max_t = torch.cat([starts, ends]).max().detach().clamp_min(1.0) + 1.0
        order = torch.argsort(ray_indices.to(starts.dtype) * max_t + starts)
        ray_indices = ray_indices[order]

        def cat_attr(name: str) -> Optional[torch.Tensor]:
            a = getattr(first_samples, name)
            b = getattr(second_samples, name)
            if a is None or b is None:
                return None
            return torch.cat([a, b], dim=0)[order]

        origins = torch.cat([first_samples.frustums.origins, second_samples.frustums.origins], dim=0)[order]
        directions = torch.cat([first_samples.frustums.directions, second_samples.frustums.directions], dim=0)[order]
        pixel_area = torch.cat([first_samples.frustums.pixel_area, second_samples.frustums.pixel_area], dim=0)[order]
        starts_sorted = torch.cat([first_samples.frustums.starts, second_samples.frustums.starts], dim=0)[order]
        ends_sorted = torch.cat([first_samples.frustums.ends, second_samples.frustums.ends], dim=0)[order]
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=directions,
                starts=starts_sorted,
                ends=ends_sorted,
                pixel_area=pixel_area,
            ),
            camera_indices=cat_attr("camera_indices"),
            deltas=cat_attr("deltas"),
            spacing_starts=cat_attr("spacing_starts"),
            spacing_ends=cat_attr("spacing_ends"),
        )
        if first_samples.times is not None and second_samples.times is not None:
            ray_samples.times = torch.cat([first_samples.times, second_samples.times], dim=0)[order]
        return ray_samples, ray_indices

    def _append_fallback_samples(
        self,
        ray_bundle: RayBundle,
        ray_samples: RaySamples,
        ray_indices: torch.Tensor,
        stats,
        fallback_count: Optional[int] = None,
    ) -> Tuple[RaySamples, torch.Tensor, object, int]:
        if fallback_count is None:
            fallback_count = int(self.config.adaptive_fixed_fallback_samples_per_ray)
        if fallback_count <= 0:
            return ray_samples, ray_indices, stats, 0
        fallback_samples, fallback_indices = self._fallback_ray_samples(ray_bundle, fallback_count)
        if fallback_indices.numel() == 0:
            return ray_samples, ray_indices, stats, 0
        merged_samples, merged_indices = self._concat_packed_ray_samples(
            ray_samples,
            ray_indices,
            fallback_samples,
            fallback_indices,
        )
        num_rays = len(ray_bundle)
        packed = nerfacc.pack_info(merged_indices, num_rays)
        sample_counts = packed[:, 1]
        stats.num_samples = torch.tensor(merged_indices.numel(), device=merged_indices.device)
        stats.mean_samples_per_ray = sample_counts.float().mean()
        stats.max_samples_per_ray = sample_counts.max().float()
        stats.packed_info = packed
        if int(self.config.max_steps_per_ray) > 0:
            stats.saturation_rate = (sample_counts >= int(self.config.max_steps_per_ray)).float().mean()
        return merged_samples, merged_indices, stats, int(fallback_indices.numel())

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        # Frequency grid is a buffer, not a parameter
        return param_groups

    @staticmethod
    def _packed_distortion_loss(
        spacing_starts: torch.Tensor,
        spacing_ends: torch.Tensor,
        weights: torch.Tensor,
        ray_indices: torch.Tensor,
        num_rays: int,
    ) -> torch.Tensor:
        """Linear-time Mip-NeRF 360 distortion loss for packed, ray-sorted samples."""
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

    @staticmethod
    def _dense_distortion_loss(
        spacing_starts: torch.Tensor,
        spacing_ends: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Linear-time Mip-NeRF 360 distortion loss for dense, sorted ray samples."""
        if weights.numel() == 0:
            return weights.sum(dim=-2)

        starts = spacing_starts.reshape(*spacing_starts.shape[:-1])
        ends = spacing_ends.reshape(*spacing_ends.shape[:-1])
        w = weights.reshape(*weights.shape[:-1])
        mid = 0.5 * (starts + ends)
        interval = (ends - starts).clamp_min(0.0)

        prefix_w = torch.cumsum(w, dim=-1) - w
        prefix_wm = torch.cumsum(w * mid, dim=-1) - w * mid
        inter = 2.0 * w * (mid * prefix_w - prefix_wm)
        intra = (w**2) * interval / 3.0
        return (inter + intra).sum(dim=-1, keepdim=True)

    def adaptive_ray_marching(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Packed frequency-aware ray marching using nerfacc occupancy traversal."""
        assert self.config.render_step_size is not None
        coarse_step_size = (
            float(self.config.adaptive_coarse_step_size)
            if self.config.adaptive_coarse_step_size is not None
            else float(self.config.adaptive_max_step_size)
        )
        num_rays = len(ray_bundle)
        ray_samples, ray_indices, stats = self.adaptive_sampler(
            ray_bundle=ray_bundle,
            render_step_size=coarse_step_size,
            near_plane=float(self.config.near_plane),
            far_plane=float(self.config.far_plane),
            alpha_thre=float(self.config.alpha_thre),
            early_stop_eps=float(self.config.transmittance_threshold),
            cone_angle=float(self.config.cone_angle),
            adaptive_min_step_size=float(self.config.adaptive_min_step_size),
            adaptive_max_step_size=float(self.config.adaptive_max_step_size),
            adaptive_min_frequency_level=float(self.config.adaptive_min_frequency_level),
            adaptive_max_frequency_level=(
                None if self.config.adaptive_max_frequency_level is None else float(self.config.adaptive_max_frequency_level)
            ),
            adaptive_interval_level_mode=self.config.adaptive_interval_level_mode,
            max_steps_per_ray=int(self.config.max_steps_per_ray),
            corrected_allocator=bool(self.config.corrected_arm_allocator),
        )
        ray_samples, ray_indices, stats, fallback_samples = self._append_fallback_samples(
            ray_bundle=ray_bundle,
            ray_samples=ray_samples,
            ray_indices=ray_indices,
            stats=stats,
        )

        if ray_indices.numel() == 0:
            rgb = torch.zeros((num_rays, 3), device=ray_bundle.origins.device)
            accumulation = torch.zeros((num_rays, 1), device=ray_bundle.origins.device)
            depth = torch.zeros((num_rays, 1), device=ray_bundle.origins.device)
            return {
                "rgb": self.renderer_rgb.combine_rgb(
                    rgb=torch.zeros((0, 3), device=ray_bundle.origins.device),
                    weights=torch.zeros((0, 1), device=ray_bundle.origins.device),
                    background_color=self.config.background_color,
                    ray_indices=ray_indices,
                    num_rays=num_rays,
                ),
                "depth": depth,
                "accumulation": accumulation,
                "num_samples_per_ray": torch.zeros((num_rays,), device=ray_bundle.origins.device),
                "adaptive_num_samples": stats.num_samples.float(),
                "adaptive_samples_mean": stats.mean_samples_per_ray,
                "adaptive_samples_max": stats.max_samples_per_ray,
                "adaptive_saturation_rate": stats.saturation_rate,
                "adaptive_fallback_samples": torch.tensor(float(fallback_samples), device=ray_bundle.origins.device),
                "packed_spacing_starts": ray_samples.spacing_starts,
                "packed_spacing_ends": ray_samples.spacing_ends,
                "packed_ray_indices": ray_indices,
                "packed_weights": torch.zeros((0, 1), device=ray_bundle.origins.device),
            }

        field_outputs = self.field(ray_samples)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        packed_info = stats.packed_info
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0][..., None]

        if (
            self.training
            and self.config.background_color == "black"
            and renderer_module.BACKGROUND_COLOR_OVERRIDE is None
        ):
            rgb, accumulation, depth = self._render_packed_black(
                rgb_samples=field_outputs[FieldHeadNames.RGB],
                weights=weights,
                ray_samples=ray_samples,
                ray_indices=ray_indices,
                num_rays=num_rays,
            )
        else:
            rgb = self.renderer_rgb(
                rgb=field_outputs[FieldHeadNames.RGB],
                weights=weights,
                ray_indices=ray_indices,
                num_rays=num_rays,
            )
            accumulation = self.renderer_accumulation(
                weights=weights, ray_indices=ray_indices, num_rays=num_rays
            )
            depth = self.renderer_depth(
                weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
            )

        return {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
            "num_samples_per_ray": packed_info[:, 1],
            "adaptive_num_samples": stats.num_samples.float(),
            "adaptive_samples_mean": stats.mean_samples_per_ray,
            "adaptive_samples_max": stats.max_samples_per_ray,
            "adaptive_saturation_rate": stats.saturation_rate,
            "adaptive_fallback_samples": torch.tensor(float(fallback_samples), device=ray_bundle.origins.device),
            "packed_spacing_starts": ray_samples.spacing_starts,
            "packed_spacing_ends": ray_samples.spacing_ends,
            "packed_ray_indices": ray_indices,
            "packed_weights": weights,
        }

    def occupancy_ray_marching(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Packed constant-step nerfacc occupancy traversal without frequency-aware subdivision."""
        assert self.config.render_step_size is not None
        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        num_rays = len(ray_bundle)
        t_min = ray_bundle.nears.contiguous().reshape(-1) if ray_bundle.nears is not None else None
        t_max = ray_bundle.fars.contiguous().reshape(-1) if ray_bundle.fars is not None else None

        def sigma_fn(t_starts: torch.Tensor, t_ends: torch.Tensor, ray_indices: torch.Tensor) -> torch.Tensor:
            positions = rays_o[ray_indices] + rays_d[ray_indices] * ((t_starts + t_ends)[:, None] * 0.5)
            return self.field.density_fn(positions).squeeze(-1)

        with torch.no_grad():
            ray_indices, starts, ends = self.occupancy_grid.sampling(
                rays_o=rays_o,
                rays_d=rays_d,
                t_min=t_min,
                t_max=t_max,
                sigma_fn=sigma_fn if self.training else None,
                render_step_size=float(self.config.render_step_size),
                near_plane=float(self.config.near_plane),
                far_plane=float(self.config.far_plane),
                early_stop_eps=float(self.config.transmittance_threshold),
                stratified=self.training,
                cone_angle=float(self.config.cone_angle),
                alpha_thre=float(self.config.alpha_thre),
            )

        device = rays_o.device
        if starts.numel() == 0:
            empty = torch.zeros((0, 1), dtype=rays_o.dtype, device=device)
            return {
                "rgb": self.renderer_rgb.combine_rgb(
                    rgb=torch.zeros((0, 3), dtype=rays_o.dtype, device=device),
                    weights=empty,
                    background_color=self.config.background_color,
                    ray_indices=ray_indices,
                    num_rays=num_rays,
                ),
                "depth": torch.zeros((num_rays, 1), dtype=rays_o.dtype, device=device),
                "accumulation": torch.zeros((num_rays, 1), dtype=rays_o.dtype, device=device),
                "num_samples_per_ray": torch.zeros((num_rays,), dtype=torch.long, device=device),
                "occupancy_traversal_num_samples": torch.tensor(0.0, device=device),
                "occupancy_traversal_samples_mean": torch.tensor(0.0, device=device),
                "occupancy_traversal_samples_max": torch.tensor(0.0, device=device),
                "packed_spacing_starts": empty,
                "packed_spacing_ends": empty,
                "packed_ray_indices": ray_indices,
                "packed_weights": empty,
            }

        stats = FrequencyAwareSamplerStats(
            num_samples=torch.tensor(starts.numel(), device=device),
            mean_samples_per_ray=torch.zeros((), device=device),
            max_samples_per_ray=torch.zeros((), device=device),
            saturation_rate=torch.zeros((), device=device),
            packed_info=nerfacc.pack_info(ray_indices, num_rays),
        )
        origins = rays_o[ray_indices]
        directions = rays_d[ray_indices]
        camera_indices = ray_bundle.camera_indices
        if camera_indices is not None:
            camera_indices = camera_indices.contiguous()[ray_indices]
        if t_min is None:
            ray_nears = torch.full_like(starts, float(self.config.near_plane))
        else:
            ray_nears = t_min[ray_indices].to(dtype=starts.dtype)
        if t_max is None:
            ray_fars = torch.full_like(ends, float(self.config.far_plane))
        else:
            ray_fars = t_max[ray_indices].to(dtype=ends.dtype)
        ray_spans = (ray_fars - ray_nears).clamp_min(1e-6)
        spacing_starts = ((starts - ray_nears) / ray_spans).clamp(0.0, 1.0)
        spacing_ends = ((ends - ray_nears) / ray_spans).clamp(0.0, 1.0)

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=directions,
                starts=starts[..., None],
                ends=ends[..., None],
                pixel_area=ray_bundle[ray_indices].pixel_area,
            ),
            camera_indices=camera_indices,
            deltas=(ends - starts)[..., None],
            spacing_starts=spacing_starts[..., None],
            spacing_ends=spacing_ends[..., None],
        )
        if ray_bundle.times is not None:
            ray_samples.times = ray_bundle.times[ray_indices]
        ray_samples, ray_indices, stats, fallback_samples = self._append_fallback_samples(
            ray_bundle=ray_bundle,
            ray_samples=ray_samples,
            ray_indices=ray_indices,
            stats=stats,
            fallback_count=int(self.config.occupancy_fixed_fallback_samples_per_ray),
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
        )[0][..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays)
        sample_counts = packed_info[:, 1]

        return {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
            "num_samples_per_ray": sample_counts,
            "occupancy_traversal_num_samples": torch.tensor(float(starts.numel()), device=device),
            "occupancy_traversal_samples_mean": sample_counts.float().mean(),
            "occupancy_traversal_samples_max": sample_counts.max().float(),
            "occupancy_fallback_samples": torch.tensor(float(fallback_samples), device=device),
            "packed_spacing_starts": ray_samples.spacing_starts,
            "packed_spacing_ends": ray_samples.spacing_ends,
            "packed_ray_indices": ray_indices,
            "packed_weights": weights,
        }

    def adaptive_ray_marching_python(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """
        Legacy Python-loop adaptive ray marcher retained for reference.
        Uses pre-allocated rectangular buffers to support efficient creation of
        padded RaySamples for the standard distortion loss.
        """
        rays_o = ray_bundle.origins
        rays_d = ray_bundle.directions
        n_rays = rays_o.shape[0]
        device = rays_o.device

        # --- 1. Initialization ---
        t_vals = ray_bundle.nears.clone()

        # Accumulators for final image
        acc_rgb = torch.zeros((n_rays, 3), device=device)
        acc_depth = torch.zeros((n_rays, 1), device=device)
        acc_weights = torch.zeros((n_rays, 1), device=device)
        transmittance = torch.ones((n_rays, 1), device=device)

        # --- Pre-allocation for Padded History ---
        # We need (N_rays, Max_Steps) to satisfy RaySamples structure for distortion loss.
        max_steps = self.config.max_steps_per_ray

        # Buffers initialized to 0.
        # Note: Padding 0s here is handled safely in the Finalization step by clamping.
        history_weights = torch.zeros((n_rays, max_steps, 1), device=device)
        history_starts = torch.zeros((n_rays, max_steps, 1), device=device)
        history_ends = torch.zeros((n_rays, max_steps, 1), device=device)

        # Track insertion index per ray
        step_indices = torch.zeros(n_rays, dtype=torch.long, device=device)

        # Active mask
        active_mask = torch.ones(n_rays, dtype=torch.bool, device=device)

        # Constants
        N_min = self.config.min_res
        b_val = self.freq_grid.b
        min_step_size = self.config.adaptive_min_step_size
        max_step_size = self.config.adaptive_max_step_size

        step_iter = 0

        # --- 2. Ray Marching Loop ---
        while active_mask.any() and step_iter < max_steps:
            # A. Current Positions
            # We work only on active rays to save compute on the Field query
            curr_t = t_vals[active_mask]
            curr_pos = rays_o[active_mask] + rays_d[active_mask] * curr_t

            # B. Adaptive Step Size Calculation
            l_indices = self.freq_grid.query(curr_pos).float()
            N_l = N_min * (b_val ** l_indices)

            # Nyquist step: delta = 1 / (2 * N_l)
            dt = 1.0 / (2.0 * N_l)
            dt = torch.clamp(dt, min=min_step_size, max=max_step_size)

            # C. Model Query (Density & Color)
            # Pass l_indices for Eq. 6 feature re-weighting
            view_dirs = rays_d[active_mask]
            density, rgb = self.field.query_points(curr_pos, view_dirs, l_grid=l_indices)

            # D. Volumetric Integration
            sigma = F.relu(density)
            alpha = 1.0 - torch.exp(-sigma * dt)

            curr_transmittance = transmittance[active_mask]
            weight = curr_transmittance * alpha

            # Accumulate Render
            acc_rgb[active_mask] += weight * rgb
            acc_depth[active_mask] += weight * curr_t
            acc_weights[active_mask] += weight

            # Update Transmittance
            transmittance[active_mask] *= (1.0 - alpha + 1e-10)

            # E. Store History (Scatter into Padded Buffers)
            # We use the 'step_indices' to place samples in the correct column per ray
            curr_step_idx = step_indices[active_mask]

            # Advanced indexing: [rows, cols]
            active_ray_idx = torch.nonzero(active_mask).squeeze(-1)

            history_weights[active_ray_idx, curr_step_idx] = weight
            history_starts[active_ray_idx, curr_step_idx] = curr_t
            history_ends[active_ray_idx, curr_step_idx] = curr_t + dt

            # Increment step indices for active rays
            step_indices[active_mask] += 1

            # F. Advance Rays
            t_vals[active_mask] += dt

            # G. Pruning
            opaque = transmittance < self.config.transmittance_threshold
            out_of_bounds = t_vals > ray_bundle.fars
            newly_finished = (opaque | out_of_bounds).flatten() & active_mask
            active_mask = active_mask & (~newly_finished)

            step_iter += 1

        # --- 3. Finalization ---
        depth_final = acc_depth / (acc_weights + 1e-6)

        # Background composition
        if self.renderer_rgb.background_color == "white":
            acc_rgb = acc_rgb + transmittance
        elif self.renderer_rgb.background_color == "random":
            bg = torch.rand_like(acc_rgb) if self.training else torch.zeros_like(acc_rgb)
            acc_rgb = acc_rgb + transmittance * bg

        # --- 4. Construct RaySamples for Loss ---
        # We need to normalize t -> s [0, 1] for MipNeRF 360 distortion loss
        # s = (t - near) / (far - near)

        # Expand near/far for broadcasting: (N, 1, 1)
        nears = ray_bundle.nears.unsqueeze(-1)
        fars = ray_bundle.fars.unsqueeze(-1)
        span = (fars - nears).clamp(min=1e-6)

        norm_starts = (history_starts - nears) / span
        norm_ends = (history_ends - nears) / span

        # Correctness Fix: Clamp to [0, 1] to handle padding logic cleanly.
        # Padded zeros (t=0) might become s < 0 if near > 0.
        # Clamping forces them to valid range [0, 1].
        # Since their weight is 0, they won't contribute to loss, but coordinates remain valid.
        norm_starts = norm_starts.clamp(min=0.0, max=1.0)
        norm_ends = norm_ends.clamp(min=0.0, max=1.0)

        # Create Frustums and RaySamples
        # We use dummy directions/origins for the samples as distortion loss doesn't use them.
        dummy_dirs = torch.zeros_like(history_starts).expand(-1, -1, 3)
        dummy_origins = torch.zeros_like(history_starts).expand(-1, -1, 3)

        frustums = Frustums(
            origins=dummy_origins,
            directions=dummy_dirs,
            starts=norm_starts,
            ends=norm_ends,
            pixel_area=torch.zeros_like(norm_starts) # Dummy
        )

        # camera_indices must match the dimensions (N_rays, Max_Steps, 1) or (N_rays, 1)
        # RaySamples expects one index per sample in the flattened structure,
        # or we can broadcast. Since RaySamples is a TensorDataclass,
        # we construct it with full shape (N, M, 1).
        loss_ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=torch.zeros_like(history_starts, dtype=torch.long),
            deltas=norm_ends - norm_starts,
            spacing_starts=norm_starts, # Explicitly used by nerfstudio_distortion_loss
            spacing_ends=norm_ends
        )

        return {
            "rgb": acc_rgb,
            "depth": depth_final,
            "accumulation": acc_weights,
            # Pass data needed for loss
            "loss_ray_samples": loss_ray_samples,
            "loss_weights": history_weights
        }

    def fixed_ray_marching(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Baseline fixed-step ray marcher used when Adaptive RM is disabled."""
        rays_o = ray_bundle.origins
        rays_d = ray_bundle.directions
        n_rays = rays_o.shape[0]
        device = rays_o.device

        num_samples = int(self.config.fixed_num_samples_per_ray)
        if num_samples <= 0:
            raise ValueError("fixed_num_samples_per_ray must be > 0.")

        nears = ray_bundle.nears[:, None, :]
        fars = ray_bundle.fars[:, None, :]
        span = (fars - nears).clamp(min=1e-6)

        edges = torch.linspace(0.0, 1.0, num_samples + 1, device=device)
        starts = nears + span * edges[:-1].view(1, num_samples, 1)
        ends = nears + span * edges[1:].view(1, num_samples, 1)
        mids = 0.5 * (starts + ends)
        deltas = ends - starts

        positions = rays_o[:, None, :] + rays_d[:, None, :] * mids
        directions = rays_d[:, None, :].expand(-1, num_samples, -1)

        density, rgb = self.field.query_points(
            positions.reshape(-1, 3),
            directions.reshape(-1, 3),
            camera_indices=(
                None
                if ray_bundle.camera_indices is None
                else ray_bundle.camera_indices[:, None, :].expand(-1, num_samples, -1).reshape(-1, 1)
            ),
        )
        density = F.relu(density).view(n_rays, num_samples, 1)
        rgb = rgb.view(n_rays, num_samples, 3)

        alpha = 1.0 - torch.exp(-density * deltas)
        trans = torch.cumprod(
            torch.cat([torch.ones((n_rays, 1, 1), device=device), 1.0 - alpha + 1e-10], dim=1),
            dim=1,
        )[:, :-1]
        weights = alpha * trans

        acc_rgb = torch.sum(weights * rgb, dim=1)
        acc_depth = torch.sum(weights * mids, dim=1)
        acc_weights = torch.sum(weights, dim=1)
        transmittance = 1.0 - acc_weights

        if self.renderer_rgb.background_color == "white":
            acc_rgb = acc_rgb + transmittance
        elif self.renderer_rgb.background_color == "random":
            bg = torch.rand_like(acc_rgb) if self.training else torch.zeros_like(acc_rgb)
            acc_rgb = acc_rgb + transmittance * bg

        norm_starts = edges[:-1].view(1, num_samples, 1).expand(n_rays, -1, -1)
        norm_ends = edges[1:].view(1, num_samples, 1).expand(n_rays, -1, -1)
        dummy_dirs = torch.zeros((n_rays, num_samples, 3), device=device)
        dummy_origins = torch.zeros((n_rays, num_samples, 3), device=device)

        loss_ray_samples = RaySamples(
            frustums=Frustums(
                origins=dummy_origins,
                directions=dummy_dirs,
                starts=norm_starts,
                ends=norm_ends,
                pixel_area=torch.zeros_like(norm_starts),
            ),
            camera_indices=torch.zeros_like(norm_starts, dtype=torch.long),
            deltas=norm_ends - norm_starts,
            spacing_starts=norm_starts,
            spacing_ends=norm_ends,
        )

        return {
            "rgb": acc_rgb,
            "depth": acc_depth / (acc_weights + 1e-6),
            "accumulation": acc_weights,
            "loss_ray_samples": loss_ray_samples,
            "loss_weights": weights,
        }

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        mode = self._resolved_ray_sampling_mode()
        if mode == "adaptive":
            if self.training and self.current_train_step < self.config.adaptive_warmup_steps:
                return self.fixed_ray_marching(ray_bundle)
            return self.adaptive_ray_marching(ray_bundle)
        if mode == "occupancy":
            return self.occupancy_ray_marching(ray_bundle)
        if mode != "fixed":
            raise ValueError(f"Unknown ray_sampling_mode={self.config.ray_sampling_mode!r}.")
        return self.fixed_ray_marching(ray_bundle)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Render camera rays while dropping packed training-only tensors."""
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        image_output_names = {"rgb", "depth", "accumulation", "num_samples_per_ray"}
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(i, i + num_rays_per_chunk)
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name in image_output_names:
                output = outputs.get(output_name)
                if isinstance(output, torch.Tensor) and output.ndim > 0:
                    outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        if self.config.rgb_output_parameterization == "sigmoid":
            metrics_dict["psnr"] = self.psnr(rgb, image)
        else:
            if not bool(torch.isfinite(rgb).all()):
                raise FloatingPointError("Non-finite HDR RGB prediction")
            valid = torch.isfinite(image)
            target = torch.where(valid, image, torch.zeros_like(image))
            prediction = torch.where(valid, rgb, torch.zeros_like(rgb))
            normalized_prediction = prediction / self.hdr_linear_scale
            normalized_target = target / self.hdr_linear_scale
            linear_mse = ((normalized_prediction - normalized_target).square()[valid]).mean()
            linear_psnr = -10.0 * torch.log10(linear_mse.clamp_min(1e-12))
            pq_prediction = scene_linear_to_pq(
                prediction,
                nits_per_scene_unit=self.pq_nits_per_scene_unit,
                black_nits=self.config.pq_black_nits,
            )
            pq_target = scene_linear_to_pq(
                target,
                nits_per_scene_unit=self.pq_nits_per_scene_unit,
                black_nits=self.config.pq_black_nits,
            )
            pq_mse = ((pq_prediction - pq_target).square()[valid]).mean()
            pq_psnr = -10.0 * torch.log10(pq_mse.clamp_min(1e-12))
            metrics_dict["psnr"] = pq_psnr
            metrics_dict["linear_psnr"] = linear_psnr
            metrics_dict["pq_psnr"] = pq_psnr
            metrics_dict["pq_upper_clip_rate"] = (
                self.config.pq_black_nits + self.pq_nits_per_scene_unit * prediction.clamp_min(0.0)
                > self.config.pq_peak_nits
            ).float().mean()
        if "num_samples_per_ray" in outputs:
            metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
            metrics_dict["samples_per_ray_mean"] = outputs["num_samples_per_ray"].float().mean()
            metrics_dict["zero_sample_ray_rate"] = (outputs["num_samples_per_ray"] == 0).float().mean()
        if "adaptive_samples_mean" in outputs:
            metrics_dict["adaptive_samples_mean"] = outputs["adaptive_samples_mean"]
            metrics_dict["adaptive_samples_max"] = outputs["adaptive_samples_max"]
            metrics_dict["adaptive_saturation_rate"] = outputs["adaptive_saturation_rate"]
        if "adaptive_fallback_samples" in outputs:
            metrics_dict["adaptive_fallback_samples"] = outputs["adaptive_fallback_samples"]
        if "occupancy_traversal_samples_mean" in outputs:
            metrics_dict["occupancy_traversal_num_samples"] = outputs["occupancy_traversal_num_samples"]
            metrics_dict["occupancy_traversal_samples_mean"] = outputs["occupancy_traversal_samples_mean"]
            metrics_dict["occupancy_traversal_samples_max"] = outputs["occupancy_traversal_samples_max"]
        if "occupancy_fallback_samples" in outputs:
            metrics_dict["occupancy_fallback_samples"] = outputs["occupancy_fallback_samples"]
        for name, value in self._last_occupancy_stats.items():
            metrics_dict[name] = value
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)

        # 1. Charbonnier Reconstruction Loss
        if self.config.reconstruction_loss_type == "charbonnier":
            epsilon = 1e-4
            loss_dict["rgb_loss"] = torch.sqrt((outputs["rgb"] - image) ** 2 + epsilon).mean()
        elif self.config.reconstruction_loss_type == "mse":
            loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        elif self.config.reconstruction_loss_type == "huber":
            loss_dict["rgb_loss"] = (
                F.huber_loss(outputs["rgb"], image, delta=float(self.config.huber_delta), reduction="mean") / 5.0
            )
        elif self.config.reconstruction_loss_type in {
            "linear_l1",
            "rawnerf_weighted_l2",
            "linear_pq",
            "pq_l1",
            "eag_pq_dssim",
        }:
            prediction = outputs["rgb"]
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("Non-finite HDR RGB prediction")
            valid = torch.isfinite(image)
            if not bool(valid.any()):
                raise ValueError("HDR batch contains no finite target channels")
            target = torch.where(valid, image, torch.zeros_like(image))
            prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
            normalized_prediction = prediction / self.hdr_linear_scale
            normalized_target = target / self.hdr_linear_scale
            linear_l1 = (normalized_prediction - normalized_target).abs()[valid].mean()

            if self.config.reconstruction_loss_type == "linear_l1":
                reconstruction = linear_l1
            elif self.config.reconstruction_loss_type == "rawnerf_weighted_l2":
                denominator = normalized_prediction.detach().clamp_min(0.0) + float(self.config.rawnerf_epsilon)
                reconstruction = (((normalized_prediction - normalized_target) / denominator).square())[valid].mean()
            else:
                pq_prediction = scene_linear_to_pq(
                    prediction,
                    nits_per_scene_unit=self.pq_nits_per_scene_unit,
                    black_nits=self.config.pq_black_nits,
                )
                pq_target = scene_linear_to_pq(
                    target,
                    nits_per_scene_unit=self.pq_nits_per_scene_unit,
                    black_nits=self.config.pq_black_nits,
                )
                pq_l1 = (pq_prediction - pq_target).abs()[valid].mean()
                reconstruction = pq_l1 + float(self.config.pq_linear_anchor_weight) * linear_l1
                if self.config.reconstruction_loss_type == "eag_pq_dssim":
                    patch_size = int(self.config.eag_patch_size)
                    rays_per_patch = patch_size * patch_size
                    if prediction.ndim != 2 or prediction.shape[0] % rays_per_patch != 0:
                        if getattr(self, "training", True):
                            raise ValueError(
                                "eag_pq_dssim requires contiguous patch batches with ray count divisible by "
                                f"eag_patch_size**2 ({rays_per_patch})."
                            )
                        # Nerfstudio's lightweight eval-loss batch is an unstructured ray batch.
                        # Full-image PQ SSIM is still measured by get_image_metrics_and_images.
                    else:
                        predicted_patches = pq_prediction.reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)
                        target_patches = pq_target.reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)
                        dssim = 1.0 - self.ssim(predicted_patches, target_patches, data_range=1.0)
                        weight = float(self.config.eag_dssim_weight)
                        reconstruction = (1.0 - weight) * pq_l1 + weight * dssim
                        if self.config.eag_edge_weight > 0:
                            pred_dx = predicted_patches[..., 1:] - predicted_patches[..., :-1]
                            target_dx = target_patches[..., 1:] - target_patches[..., :-1]
                            pred_dy = predicted_patches[..., 1:, :] - predicted_patches[..., :-1, :]
                            target_dy = target_patches[..., 1:, :] - target_patches[..., :-1, :]
                            edge_loss = 0.5 * (
                                (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()
                            )
                            reconstruction = reconstruction + float(self.config.eag_edge_weight) * edge_loss
                        reconstruction = reconstruction + float(self.config.pq_linear_anchor_weight) * linear_l1
            loss_dict["rgb_loss"] = reconstruction
        else:
            raise ValueError(f"Unknown reconstruction_loss_type={self.config.reconstruction_loss_type!r}.")

        # 2. Distortion Loss (Mip-NeRF 360)
        # Uses the standard Nerfstudio implementation which expects (RaySamples, weights)
        if self.config.distortion_loss_mult > 0:
            if "packed_weights" in outputs:
                distortion = self._packed_distortion_loss(
                    spacing_starts=outputs["packed_spacing_starts"],
                    spacing_ends=outputs["packed_spacing_ends"],
                    weights=outputs["packed_weights"],
                    ray_indices=outputs["packed_ray_indices"],
                    num_rays=outputs["rgb"].shape[0],
                ).mean()
            else:
                distortion = self._dense_distortion_loss(
                    spacing_starts=outputs["loss_ray_samples"].spacing_starts,
                    spacing_ends=outputs["loss_ray_samples"].spacing_ends,
                    weights=outputs["loss_weights"],
                ).mean()
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * distortion

        # 3. Depth Loss (Sparse Supervision)
        if (
            self.config.depth_loss_mult > 0
            and "depth_image" in batch
        ):
            gt_depth = batch["depth_image"].to(self.device)
            mask = gt_depth > 0
            if mask.any():
                pred_depth = outputs["depth"]
                depth_loss = F.mse_loss(pred_depth[mask], gt_depth[mask])
                loss_dict["depth_loss"] = self.config.depth_loss_mult * depth_loss

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        hdr_enabled = self.config.rgb_output_parameterization != "sigmoid"
        if hdr_enabled:
            preview_ev = math.log2(0.18 / max(self.hdr_initial_radiance, 1e-8))
            image_for_display = hdr_display_preview(image, exposure_ev=preview_ev)
            rgb_for_display = hdr_display_preview(rgb, exposure_ev=preview_ev)
        else:
            image_for_display = image
            rgb_for_display = rgb
        combined_rgb = torch.cat([image_for_display, rgb_for_display], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics.
        image_chw = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_chw = torch.moveaxis(rgb, -1, 0)[None, ...]

        if hdr_enabled:
            if not bool(torch.isfinite(rgb_chw).all()):
                raise FloatingPointError("Non-finite HDR RGB prediction")
            image_chw = torch.nan_to_num(image_chw)
            normalized_error = (rgb_chw - image_chw) / self.hdr_linear_scale
            linear_psnr = -10.0 * torch.log10(normalized_error.square().mean().clamp_min(1e-12))
            pq_image = scene_linear_to_pq(
                image_chw,
                nits_per_scene_unit=self.pq_nits_per_scene_unit,
                black_nits=self.config.pq_black_nits,
            )
            pq_rgb = scene_linear_to_pq(
                rgb_chw,
                nits_per_scene_unit=self.pq_nits_per_scene_unit,
                black_nits=self.config.pq_black_nits,
            )
            pq_psnr = self.psnr(pq_image, pq_rgb)
            pq_ssim = self.ssim(pq_image, pq_rgb, data_range=1.0)
            pq_lpips = self.lpips(pq_image, pq_rgb)
            metrics_dict = {
                "psnr": float(pq_psnr.item()),
                "ssim": float(pq_ssim),
                "lpips": float(pq_lpips),
                "linear_psnr": float(linear_psnr),
                "pq_psnr": float(pq_psnr),
                "pq_ssim": float(pq_ssim),
                "pq_lpips": float(pq_lpips),
            }
        else:
            psnr = self.psnr(image_chw, rgb_chw)
            ssim = self.ssim(image_chw, rgb_chw)
            lpips = self.lpips(image_chw, rgb_chw)
            metrics_dict = {
                "psnr": float(psnr.item()),
                "ssim": float(ssim),
                "lpips": float(lpips),
            }
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }
        return metrics_dict, images_dict
