from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_lookcloser_temporal_finetune as temporal


def test_selected_recipe_is_frozen() -> None:
    recipe = temporal.recipe_manifest()

    assert temporal.INITIAL_LR == 0.015
    assert temporal.FINAL_LR == 0.0001
    assert temporal.SCHEDULER_MAX_STEPS == 300_000
    assert temporal.TARGET_STEP == 151_880
    assert temporal.MAX_NUM_ITERATIONS == 151_881
    assert temporal.TARGET_STEP == 10 * temporal.v2.INTERVAL
    assert recipe["checkpoint_load_mode"] == "model_parameters_only"
    assert recipe["log2_hashmap_size"] == 23
    assert recipe["frequency_maps"] == str(temporal.v2.TARGET_MAPS)
    assert recipe["fas_strength"] == 1.0
    assert recipe["feature_reweighting_strength"] == 0.3
    assert recipe["fixed_traversal_and_fresh_occupancy_warmup_updates"] == 4096
    assert recipe["fused_adam"] is False
    assert recipe["tcnn_network_jit"] is False
    assert recipe["cached_train_rays"] is False
    assert recipe["cpu_fas_prefetch"] is False
    assert recipe["independent_rng_streams"] is False


def test_default_output_is_new_timestamped_v2_directory() -> None:
    args = temporal.parse_args([])

    assert args.output_dir.parent == temporal.DEFAULT_OUTPUT_ROOT
    assert args.output_dir.name.startswith(temporal.RECIPE_ID + "_")
    assert not args.resume


def test_resume_requires_explicit_output_directory() -> None:
    with pytest.raises(SystemExit):
        temporal.parse_args(["--resume"])


def test_removed_sweep_arguments_fail_closed() -> None:
    with pytest.raises(SystemExit):
        temporal.parse_args(["--lr-candidates", "0.01,0.015"])
    with pytest.raises(SystemExit):
        temporal.parse_args(["--start-frame", "007754"])


def test_fixed_segments_reproduce_authoritative_process_boundaries(
    tmp_path: Path,
) -> None:
    args = temporal.parse_args(["--output-dir", str(tmp_path / "run")])
    segments = temporal.fixed_segments(args)

    assert [segment.target_step for segment in segments] == [
        60_752,
        75_940,
        91_128,
        106_316,
        121_504,
        136_692,
        151_880,
    ]
    assert segments[0].segment_id == f"{temporal.ARM_ID}-to-60752"
    assert segments[0].load_mode == "model_parameters_only"
    assert segments[0].parent_checkpoint == temporal.v2.LEADER_CHECKPOINT
    assert all(
        segment.load_mode == "resume" for segment in segments[1:]
    )
    assert all(segment.arm == temporal.fixed_arm() for segment in segments)
    for previous, current in zip(segments, segments[1:]):
        assert current.parent_checkpoint == temporal.v2.checkpoint_path(
            previous.run_dir, previous.target_step
        )


def test_effective_config_matches_selected_recipe(tmp_path: Path) -> None:
    args = temporal.parse_args(["--output-dir", str(tmp_path / "run")])
    config, differences = temporal.v2.configured_segment(
        args, temporal.initial_segment(args)
    )
    sampler = config.pipeline.datamanager.pixel_sampler
    model = config.pipeline.model
    optimizer = config.optimizers["fields"]["optimizer"]
    scheduler = config.optimizers["fields"]["scheduler"]

    assert config.max_num_iterations == temporal.v2.INITIAL_FINAL_STEP + 1
    assert config.checkpoint_load_mode == "model_parameters_only"
    assert config.load_checkpoint == temporal.v2.LEADER_CHECKPOINT
    assert config.load_optimizers is False
    assert config.load_scheduler is False
    assert config.checkpoint_load_parameter_hash_audit is True
    assert optimizer.lr == pytest.approx(temporal.INITIAL_LR)
    assert optimizer.eps == pytest.approx(1e-15)
    assert optimizer.weight_decay == 0
    assert optimizer.fused is False
    assert scheduler.lr_final == pytest.approx(temporal.FINAL_LR)
    assert scheduler.max_steps == temporal.SCHEDULER_MAX_STEPS
    assert scheduler.warmup_steps == 0
    assert config.steps_per_save == temporal.v2.INTERVAL
    assert config.steps_per_eval_all_images == temporal.v2.INTERVAL
    assert config.pipeline.datamanager.train_num_rays_per_batch == 4096
    assert sampler.enable_fas is True
    assert sampler.fas_strength == pytest.approx(1.0)
    assert sampler.frequency_map_dir == "lookcloser_frequencies"
    assert model.log2_hashmap_size == 23
    assert model.num_frequency_levels == 16
    assert model.hash_features_per_level == 2
    assert model.feature_reweighting_strength == pytest.approx(0.3)
    assert model.adaptive_warmup_steps == 4096
    assert model.occupancy_warmup_steps == 4096
    assert model.occupancy_binary_warmup_steps == 4096
    assert model.tcnn_network_jit is False
    assert config.pipeline.datamanager.cache_train_rays is False
    assert config.pipeline.datamanager.cpu_fas_prefetch is False
    assert config.pipeline.independent_rng_streams is False
    assert set(differences) <= temporal.v2.ALLOWED_CONFIG_DIFFS


def test_dry_run_is_deterministic_and_has_no_screen(tmp_path: Path) -> None:
    args = temporal.parse_args(
        ["--dry-run", "--output-dir", str(tmp_path / "run")]
    )

    first = temporal.deterministic_dry_run(args)
    second = temporal.deterministic_dry_run(args)

    assert first == second
    assert first["segments"][-1]["target_step"] == temporal.TARGET_STEP
    assert first["segments"][0]["load_mode"] == "model_parameters_only"
    assert all(
        segment["load_mode"] == "resume"
        for segment in first["segments"][1:]
    )
    assert (
        first["segments"][-1]["effective"]["max_num_iterations"]
        == temporal.MAX_NUM_ITERATIONS
    )
    assert first["segments"][0]["effective"]["optimizer_lr"] == pytest.approx(
        0.015
    )
    assert (
        first["segments"][0]["effective"]["scheduler_max_steps"] == 300_000
    )
    assert "wave_a" not in first
    assert "wave_b" not in first


def test_reference_selected_result_passes_leader_thresholds() -> None:
    metrics = temporal.REFERENCE_METRICS

    assert metrics["psnr"] >= temporal.v2.PSNR_THRESHOLD
    assert metrics["ssim"] >= temporal.v2.SSIM_THRESHOLD
    assert metrics["lpips"] <= temporal.v2.LPIPS_THRESHOLD


def test_static_preflight_comparison_ignores_only_storage() -> None:
    previous = {"git": {"commit": "x"}, "storage": {"free": 10}}
    current = {"git": {"commit": "x"}, "storage": {"free": 20}}
    changed = {"git": {"commit": "y"}, "storage": {"free": 20}}

    assert temporal._same_static_preflight(previous, current)
    assert not temporal._same_static_preflight(previous, changed)
