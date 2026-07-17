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
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.fields.lookcloser_field import TCNNNetworkJITScope
from nerfstudio.models.lookcloser import LookCloserModel
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.lookcloser_rng import fork_seeded_rng, stream_seed
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class LookCloserPipelineConfig(VanillaPipelineConfig):
    """Configuration for LookCloser Pipeline."""

    _target: Type = field(default_factory=lambda: LookCloserPipeline)

    training_seed: int = 42
    """Recorded campaign seed used to derive independent per-step RNG streams."""

    independent_rng_streams: bool = False
    """Isolate pixel/FAS and frequency-grid RNG from other stochastic subsystems."""

    frequency_map_dir: str = "lookcloser_frequencies"
    """Name of the directory inside the data folder containing pre-computed frequency maps."""

    enable_frequency_grid: bool = True
    """Whether to load 2D maps and run periodic 3D frequency-grid updates."""

    grid_update_interval: int = 1024
    """Step interval for updating the 3D frequency grid using dense depth rendering."""

    grid_update_batch_size: int = 2048
    """Number of rays to sample for the grid update step."""

    frequency_patch_size: int = 8
    """Fallback patch size for legacy frequency maps without sidecar metadata."""

    frequency_stride: int = 8
    """Fallback frequency-map stride for legacy maps without sidecar metadata."""

    train_rays_switch_step: Optional[int] = None
    """Optional trainer step at which to change the live pixel-sampler ray batch."""

    train_rays_after_switch: Optional[int] = None
    """Ray batch used at and after ``train_rays_switch_step`` without restarting RNG streams."""

    feature_reweighting_switch_step: Optional[int] = None
    """Optional trainer step for an in-process feature-reweighting strength change."""

    feature_reweighting_after_switch: Optional[float] = None
    """Feature-reweighting strength used at and after the live switch."""

    tcnn_network_jit_switch_step: Optional[int] = None
    """Optional trainer step that enables the model-selected TCNN JIT scope in-process."""

    tcnn_network_jit_second_switch_step: Optional[int] = None
    """Optional later trainer step that enables an additional TCNN JIT scope in-process."""

    tcnn_network_jit_second_switch_scope: Optional[TCNNNetworkJITScope] = None
    """TCNN network scope enabled by the optional second live JIT switch."""

    target_num_samples_per_batch: int = 0
    """Target field points per update; non-positive preserves the historical fixed ray batch."""

    target_num_samples_switch_step: Optional[int] = None
    """Optional trainer step whose completed update starts a new dynamic point target."""

    target_num_samples_after_switch: Optional[int] = None
    """Dynamic field-point target used to choose the ray batch after the switch update."""

    dynamic_rays_start_step: int = 0
    """First trainer step allowed to change rays; zero preserves the original controller behavior."""

    dynamic_rays_ema: float = 0.9
    """EMA decay for observed samples per ray under point-budget control."""

    dynamic_rays_min: int = 256
    """Minimum live ray batch under point-budget control."""

    dynamic_rays_max: int = 32768
    """Maximum live ray batch under point-budget control."""

    dynamic_rays_change_limit: float = 1.25
    """Maximum multiplicative ray-batch change after one optimizer update."""


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
        self._validate_independent_rng_config(config)
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        switch_values = (self.config.train_rays_switch_step, self.config.train_rays_after_switch)
        if self.config.datamanager.cpu_fas_prefetch and (
            switch_values[0] is not None
            or self.config.target_num_samples_per_batch > 0
            or self.config.target_num_samples_switch_step is not None
        ):
            raise ValueError("CPU FAS prefetch v1 requires a fixed ray batch without dynamic point targets")
        if (switch_values[0] is None) != (switch_values[1] is None):
            raise ValueError("train_rays_switch_step and train_rays_after_switch must be set together")
        if switch_values[0] is not None and switch_values[0] < 0:
            raise ValueError("train_rays_switch_step must be non-negative")
        if switch_values[1] is not None and switch_values[1] <= 0:
            raise ValueError("train_rays_after_switch must be positive")
        if self.config.target_num_samples_per_batch > 0 and switch_values[0] is not None:
            raise ValueError("Dynamic point-budget control cannot be combined with a fixed train-ray switch")
        self._train_rays_switch_applied = False
        fr_switch_values = (
            self.config.feature_reweighting_switch_step,
            self.config.feature_reweighting_after_switch,
        )
        if (fr_switch_values[0] is None) != (fr_switch_values[1] is None):
            raise ValueError(
                "feature_reweighting_switch_step and feature_reweighting_after_switch must be set together"
            )
        if fr_switch_values[0] is not None and fr_switch_values[0] < 0:
            raise ValueError("feature_reweighting_switch_step must be non-negative")
        if fr_switch_values[1] is not None and fr_switch_values[1] < 0:
            raise ValueError("feature_reweighting_after_switch must be non-negative")
        self._feature_reweighting_switch_applied = False
        self._validate_tcnn_network_jit_switch_config()
        jit_scope = self.config.model.tcnn_network_jit_scope
        self._tcnn_network_jit_switch_applied = self.model.field.get_tcnn_network_jit(scope=jit_scope)
        if self._tcnn_network_jit_switch_applied != bool(self.config.model.tcnn_network_jit):
            raise RuntimeError(f"Initial TCNN JIT state does not match config for scope {jit_scope!r}")
        self._tcnn_network_jit_second_switch_applied = False
        self._validate_dynamic_batch_config()
        self.register_buffer(
            "cumulative_point_samples",
            torch.zeros((), dtype=torch.int64, device=self.device),
            persistent=True,
        )
        self.register_buffer(
            "dynamic_samples_per_ray_ema",
            torch.zeros((), dtype=torch.float64, device=self.device),
            persistent=True,
        )
        self._dynamic_samples_per_ray_ema_value = 0.0
        self.register_buffer(
            "dynamic_rays_per_batch_state",
            torch.zeros((), dtype=torch.int64, device=self.device),
            persistent=True,
        )
        self.register_buffer(
            "fas_sample_count_state",
            torch.zeros((), dtype=torch.int64, device=self.device),
            persistent=True,
        )
        self.dynamic_num_rays_per_batch = int(self.datamanager.get_train_rays_per_batch())
        initial_target = self._target_num_samples_for_step(0)
        if initial_target > 0 and self.config.dynamic_rays_start_step == 0:
            initial = initial_target // max(
                int(getattr(self.config.model, "max_steps_per_ray", 1024)), 1
            )
            self.dynamic_num_rays_per_batch = self._clamp_dynamic_rays(initial)
            self._set_train_rays_per_batch(self.dynamic_num_rays_per_batch)
        self.dynamic_rays_per_batch_state.fill_(self.dynamic_num_rays_per_batch)
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

    @staticmethod
    def _validate_independent_rng_config(config: LookCloserPipelineConfig) -> None:
        """Reject partial or mismatched per-subsystem RNG stream configuration."""

        pipeline_enabled = bool(config.independent_rng_streams)
        model_enabled = bool(getattr(config.model, "independent_rng_streams", False))
        if pipeline_enabled != model_enabled:
            raise ValueError("Pipeline and model independent_rng_streams must be enabled together")
        if pipeline_enabled and int(config.training_seed) != int(getattr(config.model, "training_seed", -1)):
            raise ValueError("Pipeline and model training_seed must match for independent RNG streams")

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

    def _validate_dynamic_batch_config(self) -> None:
        target_switch = (
            self.config.target_num_samples_switch_step,
            self.config.target_num_samples_after_switch,
        )
        if (target_switch[0] is None) != (target_switch[1] is None):
            raise ValueError(
                "target_num_samples_switch_step and target_num_samples_after_switch must be set together."
            )
        if target_switch[0] is not None and target_switch[0] < 0:
            raise ValueError("target_num_samples_switch_step must be non-negative.")
        if target_switch[1] is not None and target_switch[1] <= 0:
            raise ValueError("target_num_samples_after_switch must be positive.")
        if target_switch[0] is not None and self.config.target_num_samples_per_batch <= 0:
            raise ValueError("A dynamic point-target switch requires a positive initial target.")
        if not 0.0 <= float(self.config.dynamic_rays_ema) < 1.0:
            raise ValueError("dynamic_rays_ema must be in [0, 1).")
        if self.config.dynamic_rays_min <= 0 or self.config.dynamic_rays_max < self.config.dynamic_rays_min:
            raise ValueError("Expected 0 < dynamic_rays_min <= dynamic_rays_max.")
        if self.config.dynamic_rays_change_limit < 1.0:
            raise ValueError("dynamic_rays_change_limit must be >= 1.")
        if self.config.dynamic_rays_start_step < 0:
            raise ValueError("dynamic_rays_start_step must be non-negative.")

    def _clamp_dynamic_rays(self, rays: int) -> int:
        return max(int(self.config.dynamic_rays_min), min(int(self.config.dynamic_rays_max), int(rays)))

    def _target_num_samples_for_step(self, step: int) -> int:
        switch_step = self.config.target_num_samples_switch_step
        target_after_switch = self.config.target_num_samples_after_switch
        if switch_step is not None and target_after_switch is not None and step >= switch_step:
            return int(target_after_switch)
        return int(self.config.target_num_samples_per_batch)

    def _set_train_rays_per_batch(self, rays: int) -> None:
        sampler = self.datamanager.train_pixel_sampler
        if sampler is None:
            raise RuntimeError("The train pixel sampler must be initialized before changing its ray batch")
        sampler.set_num_rays_per_batch(int(rays))

    def _update_dynamic_rays(self, step: int, num_samples: int, actual_rays: int) -> None:
        target_points = self._target_num_samples_for_step(step)
        if (
            target_points <= 0
            or step < self.config.dynamic_rays_start_step
            or num_samples <= 0
            or actual_rays <= 0
        ):
            return
        observed = float(num_samples) / float(actual_rays)
        prior = self._dynamic_samples_per_ray_ema_value
        if prior <= 0.0:
            ema = observed
        else:
            decay = float(self.config.dynamic_rays_ema)
            ema = decay * prior + (1.0 - decay) * observed
        self.dynamic_samples_per_ray_ema.fill_(ema)
        self._dynamic_samples_per_ray_ema_value = ema

        target = int(round(float(target_points) / max(ema, 1e-8)))
        current = max(int(actual_rays), 1)
        limit = float(self.config.dynamic_rays_change_limit)
        lower = max(1, int(current / limit))
        upper = max(lower, int(current * limit))
        target = max(lower, min(upper, target))
        self.dynamic_num_rays_per_batch = self._clamp_dynamic_rays(target)
        self.dynamic_rays_per_batch_state.fill_(self.dynamic_num_rays_per_batch)
        self._set_train_rays_per_batch(self.dynamic_num_rays_per_batch)

    def load_pipeline(self, loaded_state: Dict[str, Tensor], step: int) -> None:
        """Load legacy checkpoints and restore point-budget controller state when present."""
        required_buffers = {
            "cumulative_point_samples": torch.zeros((), dtype=torch.int64, device=self.device),
            "dynamic_samples_per_ray_ema": torch.zeros((), dtype=torch.float64, device=self.device),
            "dynamic_rays_per_batch_state": torch.tensor(
                self.dynamic_num_rays_per_batch, dtype=torch.int64, device=self.device
            ),
            "fas_sample_count_state": torch.zeros((), dtype=torch.int64, device=self.device),
        }
        normalized_keys = {key.removeprefix("module.") for key in loaded_state}
        if any(name not in normalized_keys for name in required_buffers):
            loaded_state = dict(loaded_state)
            for name, value in required_buffers.items():
                if name not in normalized_keys:
                    loaded_state[name] = value
        super().load_pipeline(loaded_state, step)
        self._dynamic_samples_per_ray_ema_value = float(self.dynamic_samples_per_ray_ema.item())
        pixel_sampler = self.datamanager.train_pixel_sampler
        if pixel_sampler is not None and hasattr(pixel_sampler, "sample_count"):
            pixel_sampler.sample_count = int(self.fas_sample_count_state.item())
        if self.config.target_num_samples_per_batch > 0:
            restored_rays = int(self.dynamic_rays_per_batch_state.item())
            if restored_rays <= 0:
                restored_rays = int(self.datamanager.get_train_rays_per_batch())
            self.dynamic_num_rays_per_batch = self._clamp_dynamic_rays(restored_rays)
            self.dynamic_rays_per_batch_state.fill_(self.dynamic_num_rays_per_batch)
            self._set_train_rays_per_batch(self.dynamic_num_rays_per_batch)
        self._sync_tcnn_network_jit_to_step(step)

    def _sync_tcnn_network_jit_to_step(self, step: int) -> None:
        """Restore both non-checkpointed TCNN JIT flags for a loaded step."""

        expected = self._expected_tcnn_network_jit_states(step)
        for network_scope in ("geometry", "color"):
            self.model.field.set_tcnn_network_jit(expected[network_scope], scope=network_scope)
        self._assert_tcnn_network_jit_states(expected, context="checkpoint resync")

        first_step = self.config.tcnn_network_jit_switch_step
        second_step = self.config.tcnn_network_jit_second_switch_step
        self._tcnn_network_jit_switch_applied = (
            bool(self.config.model.tcnn_network_jit)
            if first_step is None
            else step >= first_step
        )
        self._tcnn_network_jit_second_switch_applied = second_step is not None and step >= second_step

    def _validate_tcnn_network_jit_switch_config(self) -> None:
        """Fail closed on incomplete or ambiguous two-stage JIT schedules."""

        first_step = self.config.tcnn_network_jit_switch_step
        second_step = self.config.tcnn_network_jit_second_switch_step
        second_scope = self.config.tcnn_network_jit_second_switch_scope
        if first_step is not None and first_step < 0:
            raise ValueError("tcnn_network_jit_switch_step must be non-negative")
        if first_step is not None and self.config.model.tcnn_network_jit:
            raise ValueError("TCNN network JIT cannot be enabled both initially and by a live switch")
        if (second_step is None) != (second_scope is None):
            raise ValueError(
                "tcnn_network_jit_second_switch_step and tcnn_network_jit_second_switch_scope must be set together"
            )
        if second_step is None:
            return
        assert second_scope is not None
        self._tcnn_network_names_for_scope(second_scope)
        if first_step is None:
            raise ValueError("A second TCNN network JIT switch requires the first live switch")
        if second_step <= first_step:
            raise ValueError("tcnn_network_jit_second_switch_step must be strictly greater than the first switch")

    @staticmethod
    def _tcnn_network_names_for_scope(scope: TCNNNetworkJITScope) -> Tuple[str, ...]:
        if scope == "both":
            return "geometry", "color"
        if scope in ("geometry", "color"):
            return (scope,)
        raise ValueError(f"Unsupported TCNN network JIT scope: {scope!r}")

    def _expected_tcnn_network_jit_states(self, step: int) -> Dict[str, bool]:
        """Derive exact per-network JIT flags from config and a loaded/training step."""

        states = {"geometry": False, "color": False}
        if self.config.model.tcnn_network_jit:
            for name in self._tcnn_network_names_for_scope(self.config.model.tcnn_network_jit_scope):
                states[name] = True
        first_step = self.config.tcnn_network_jit_switch_step
        if first_step is not None and step >= first_step:
            for name in self._tcnn_network_names_for_scope(self.config.model.tcnn_network_jit_scope):
                states[name] = True
        second_step = self.config.tcnn_network_jit_second_switch_step
        second_scope = self.config.tcnn_network_jit_second_switch_scope
        if second_step is not None and second_scope is not None and step >= second_step:
            for name in self._tcnn_network_names_for_scope(second_scope):
                states[name] = True
        return states

    def _assert_tcnn_network_jit_states(self, expected: Dict[str, bool], context: str) -> None:
        for network_scope in ("geometry", "color"):
            actual = self.model.field.get_tcnn_network_jit(scope=network_scope)
            if actual != expected[network_scope]:
                raise RuntimeError(
                    f"TCNN JIT {context} failed for {network_scope}: "
                    f"expected {expected[network_scope]}, got {actual}"
                )

    def _apply_live_tcnn_jit_switch(self, step: int) -> None:
        """Apply the configured TCNN JIT switch for training or checkpoint evaluation."""

        switch_step = self.config.tcnn_network_jit_switch_step
        if switch_step is None:
            return
        first_scope = self.config.model.tcnn_network_jit_scope
        if step >= switch_step and not self._tcnn_network_jit_switch_applied:
            self.model.field.set_tcnn_network_jit(True, scope=first_scope)
            self._tcnn_network_jit_switch_applied = True
            CONSOLE.print(f"LookCloserPipeline: TCNN network JIT enabled for {first_scope} at step {step}.")

        second_step = self.config.tcnn_network_jit_second_switch_step
        second_scope = self.config.tcnn_network_jit_second_switch_scope
        if (
            second_step is not None
            and second_scope is not None
            and step >= second_step
            and not self._tcnn_network_jit_second_switch_applied
        ):
            self.model.field.set_tcnn_network_jit(True, scope=second_scope)
            self._tcnn_network_jit_second_switch_applied = True
            CONSOLE.print(
                f"LookCloserPipeline: second TCNN network JIT enabled for {second_scope} at step {step}."
            )

        self._assert_tcnn_network_jit_states(
            self._expected_tcnn_network_jit_states(step), context=f"live switch at step {step}"
        )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Run a live JIT switch before model callbacks such as occupancy updates."""

        callbacks = super().get_training_callbacks(training_callback_attributes)
        if self.config.tcnn_network_jit_switch_step is None:
            return callbacks
        switch_callback = TrainingCallback(
            where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
            update_every_num_iters=1,
            func=self._apply_live_tcnn_jit_switch,
        )
        return [switch_callback, *callbacks]

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
        self._apply_live_tcnn_jit_switch(step)
        # 1. Standard Training Step
        switch_step = self.config.train_rays_switch_step
        switch_rays = self.config.train_rays_after_switch
        if (
            not self._train_rays_switch_applied
            and switch_step is not None
            and switch_rays is not None
            and step >= switch_step
        ):
            sampler = self.datamanager.train_pixel_sampler
            if sampler is None:
                raise RuntimeError("Cannot apply train-ray schedule before the train pixel sampler is initialized")
            sampler.set_num_rays_per_batch(switch_rays)
            self._train_rays_switch_applied = True
            CONSOLE.print(
                f"LookCloserPipeline: train ray batch switched to {sampler.num_rays_per_batch} at step {step}."
            )
        fr_switch_step = self.config.feature_reweighting_switch_step
        fr_switch_strength = self.config.feature_reweighting_after_switch
        if (
            not self._feature_reweighting_switch_applied
            and fr_switch_step is not None
            and fr_switch_strength is not None
            and step >= fr_switch_step
        ):
            self.model.config.feature_reweighting_strength = fr_switch_strength
            self.model.field.feature_reweighting_strength = fr_switch_strength
            self._feature_reweighting_switch_applied = True
            CONSOLE.print(
                "LookCloserPipeline: feature reweighting strength switched to "
                f"{fr_switch_strength} at step {step}."
            )
        if hasattr(self.model, "current_train_step"):
            self.model.current_train_step = step
        actual_train_rays = int(self.datamanager.get_train_rays_per_batch())
        if self.config.independent_rng_streams:
            if self.config.datamanager.cpu_fas_prefetch:
                next_train_seeded = getattr(self.datamanager, "next_train_seeded_prefetch", None)
                if not callable(next_train_seeded):
                    raise TypeError("Independent RNG streams require seeded CPU FAS prefetch support")
                pixel_seed = stream_seed(self.config.training_seed, "pixel", step)
                next_pixel_seed = stream_seed(self.config.training_seed, "pixel", step + 1)
                ray_bundle, batch = next_train_seeded(step, pixel_seed, next_pixel_seed)
            else:
                with fork_seeded_rng(self.config.training_seed, "pixel", step, self.device):
                    ray_bundle, batch = self.datamanager.next_train(step)
            model_outputs = self._model(ray_bundle)
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        else:
            model_outputs, loss_dict, metrics_dict = super().get_train_loss_dict(step)
        pixel_sampler = self.datamanager.train_pixel_sampler
        if pixel_sampler is not None and hasattr(pixel_sampler, "sample_count"):
            self.fas_sample_count_state.fill_(int(pixel_sampler.sample_count))

        metrics_dict["train_rays_per_batch"] = torch.tensor(
            actual_train_rays, dtype=torch.int64, device=self.device
        )
        metrics_dict["feature_reweighting_strength"] = torch.tensor(
            self.model.field.feature_reweighting_strength, dtype=torch.float32, device=self.device
        )

        packed_ray_indices = model_outputs.get("packed_ray_indices")
        if packed_ray_indices is not None:
            point_samples = packed_ray_indices.numel()
        else:
            point_samples_metric = metrics_dict.get("num_samples_per_batch")
            if point_samples_metric is not None:
                point_samples = int(point_samples_metric.item())
            else:
                point_samples = model_outputs["rgb"].shape[0] * int(self.model.config.fixed_num_samples_per_ray)
        self.cumulative_point_samples.add_(point_samples)
        metrics_dict["cumulative_point_samples"] = self.cumulative_point_samples.clone()
        self._update_dynamic_rays(step, point_samples, actual_train_rays)
        metrics_dict["target_num_samples_per_batch"] = torch.tensor(
            self._target_num_samples_for_step(step), dtype=torch.int64, device=self.device
        )
        metrics_dict["next_train_rays_per_batch"] = torch.tensor(
            self.datamanager.get_train_rays_per_batch(), dtype=torch.int64, device=self.device
        )
        metrics_dict["dynamic_samples_per_ray_ema"] = self.dynamic_samples_per_ray_ema.float()

        # 2. Side-Channel Grid Update
        # "Every 1024 training steps... render depth... update voxel"
        if (
                self.config.grid_update_interval > 0
                and self.config.enable_frequency_grid
                and step % self.config.grid_update_interval == 0
                and step > 0
        ):
            if self.config.independent_rng_streams:
                with fork_seeded_rng(self.config.training_seed, "frequency_grid", step, self.device):
                    self._update_frequency_grid(step)
            else:
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

        if not self.config.independent_rng_streams:
            # The historical prefetch path derives its next batch from global
            # CPU RNG. Discard it before the grid update advances that state.
            barrier = getattr(self.datamanager, "train_batch_prefetch_barrier", None)
            if callable(barrier):
                barrier()

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

        rand_img_indices = torch.tensor(available_indices, dtype=torch.long)[
            torch.randint(0, len(available_indices), (num_samples,))
        ]
        camera_indices_cpu = rand_img_indices.long()

        # Randomly choose pixels (y, x)
        # We need image dimensions. Cameras object holds this.
        cameras = dataset.cameras
        H = cameras.height[camera_indices_cpu].squeeze(-1)  # (N,)
        W = cameras.width[camera_indices_cpu].squeeze(-1)  # (N,)

        H = H.long()
        W = W.long()
        rand_y = (torch.rand(num_samples) * H.float()).long()
        rand_x = (torch.rand(num_samples) * W.float()).long()

        # Clamp to be safe
        rand_y = torch.minimum(torch.clamp_min(rand_y, 0), H - 1)
        rand_x = torch.minimum(torch.clamp_min(rand_x, 0), W - 1)

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
        camera_indices = camera_indices_cpu.to(self.device).unsqueeze(-1)  # (N, 1)

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
        fx = cameras.fx[camera_indices_cpu].to(self.device).squeeze()
        fy = cameras.fy[camera_indices_cpu].to(self.device).squeeze()
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
