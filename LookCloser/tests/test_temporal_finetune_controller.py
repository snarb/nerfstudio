from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_lookcloser_temporal_campaign as campaign
from scripts import run_lookcloser_temporal_finetune as temporal
from scripts import temporal_finetune_common as common


def boundary(
    *,
    seed: int = 42,
    step: int,
    psnr: float,
    ssim: float = 0.676,
    lpips: float = 0.215,
    root: Path = Path("/tmp"),
) -> common.Boundary:
    return common.Boundary(
        seed=seed,
        local_step=step,
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
        checkpoint=root / f"step-{step:09d}.ckpt",
        checkpoint_sha256=f"sha-{seed}-{step}",
        eval_json=root / "eval.json",
        render_dir=root / "renders",
        completed_wall_time_ns=step,
    )


def test_fixed_recipe_and_chain_are_frozen() -> None:
    assert len(common.TARGET_FRAMES) == 43
    assert common.TARGET_FRAMES[0] == "007754"
    assert common.TARGET_FRAMES[-1] == "008048"
    assert common.SEEDS == (42, 43, 44)
    assert common.INITIAL_LR == 0.015
    assert common.FINAL_LR == 0.0001
    assert common.SCHEDULER_MAX_STEPS == 300_000
    assert common.INITIAL_TARGET_STEP == 151_880
    assert common.INITIAL_PROCESS_TARGETS == (
        60_752,
        75_940,
        91_128,
        106_316,
        121_504,
        136_692,
        151_880,
    )
    assert common.PSNR_MIN == 29.7
    assert common.SSIM_MIN == 0.668
    assert common.LPIPS_MAX == 0.22
    assert common.PREFERRED_PSNR == 29.88
    assert common.PREFERRED_SSIM == 0.676
    assert common.PREFERRED_LPIPS == 0.215


def test_runner_requires_target_parent_and_seed() -> None:
    with pytest.raises(SystemExit):
        temporal.parse_args([])
    args = temporal.parse_args(
        [
            "--target-frame",
            "007754",
            "--parent-snapshot",
            str(common.DATA_ROOT / "007747" / "snapshot"),
            "--seed",
            "43",
        ]
    )
    assert args.target_frame == "007754"
    assert args.seed == 43
    assert args.output_dir.name == "seed-43"


def test_extension_requires_resume() -> None:
    with pytest.raises(SystemExit):
        temporal.parse_args(
            [
                "--target-frame",
                "007754",
                "--parent-snapshot",
                str(common.DATA_ROOT / "007747" / "snapshot"),
                "--seed",
                "42",
                "--extend-one-interval",
            ]
        )


def test_generalized_config_uses_seed_and_model_only_parent(tmp_path: Path) -> None:
    args = temporal.parse_args(
        [
            "--target-frame",
            "007754",
            "--parent-snapshot",
            str(common.DATA_ROOT / "007747" / "snapshot"),
            "--seed",
            "44",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    parent = temporal.configure_v2(args)
    segments = temporal.initial_segments(args)
    config, differences = temporal.v2.configured_segment(args, segments[0])
    sampler = config.pipeline.datamanager.pixel_sampler
    model = config.pipeline.model
    optimizer = config.optimizers["fields"]["optimizer"]
    scheduler = config.optimizers["fields"]["scheduler"]

    assert parent["frame"] == "007747"
    assert config.machine.seed == 44
    assert config.checkpoint_load_mode == "model_parameters_only"
    assert config.load_optimizers is False
    assert config.load_scheduler is False
    assert config.load_checkpoint == common.snapshot_checkpoint(args.parent_snapshot)
    assert config.pipeline.datamanager.dataparser.data == common.DATA_ROOT / "007754"
    assert optimizer.lr == pytest.approx(0.015)
    assert optimizer.eps == pytest.approx(1e-15)
    assert optimizer.weight_decay == 0
    assert optimizer.fused is False
    assert scheduler.lr_final == pytest.approx(0.0001)
    assert scheduler.max_steps == 300_000
    assert config.steps_per_eval_all_images == common.INTERVAL
    assert config.steps_per_save == common.INTERVAL
    assert config.pipeline.datamanager.train_num_rays_per_batch == 4096
    assert sampler.fas_strength == pytest.approx(1.0)
    assert sampler.frequency_map_dir == "lookcloser_frequencies"
    assert model.log2_hashmap_size == 23
    assert model.feature_reweighting_strength == pytest.approx(0.3)
    assert model.adaptive_warmup_steps == 4096
    assert model.occupancy_warmup_steps == 4096
    assert model.occupancy_binary_warmup_steps == 4096
    assert model.tcnn_network_jit is False
    assert config.pipeline.datamanager.cache_train_rays is False
    assert config.pipeline.datamanager.cpu_fas_prefetch is False
    assert config.pipeline.independent_rng_streams is False
    assert "machine.seed" in differences


def test_process_boundaries_and_same_frame_resume(tmp_path: Path) -> None:
    args = temporal.parse_args(
        [
            "--target-frame",
            "007754",
            "--parent-snapshot",
            str(common.DATA_ROOT / "007747" / "snapshot"),
            "--seed",
            "42",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    temporal.configure_v2(args)
    segments = temporal.initial_segments(args)
    assert [row.target_step for row in segments] == list(common.INITIAL_PROCESS_TARGETS)
    assert segments[0].load_mode == "model_parameters_only"
    assert all(row.load_mode == "resume" for row in segments[1:])
    for previous, current in zip(segments, segments[1:]):
        assert current.parent_checkpoint == temporal.v2.checkpoint_path(
            previous.run_dir, previous.target_step
        )


def test_dataset_manifest_freezes_exact_standard_sets(tmp_path: Path) -> None:
    manifest = common.compute_dataset_manifest(
        "007754", common.DATA_ROOT / "007754"
    )
    assert len(manifest["jpeg"]["files"]) == 69
    assert len(manifest["frequency_maps"]["files"]) == 132
    assert manifest["frequency_maps"]["directory"] == "lookcloser_frequencies"
    path = tmp_path / "input_manifest.json"
    common.freeze_dataset_manifest("007754", common.DATA_ROOT / "007754", path)
    assert json.loads(path.read_text()) == manifest


def test_selector_uses_inclusive_psnr_window_then_lpips() -> None:
    rows = [
        boundary(step=15_188, psnr=30.0, lpips=0.218),
        boundary(seed=43, step=30_376, psnr=29.93, lpips=0.210),
        boundary(seed=44, step=45_564, psnr=29.929999, lpips=0.200),
    ]
    assert common.select_boundary(rows) == rows[1]
    assert common.contender_seeds(rows) == (42, 43)


def test_hard_gate_bootstrap_uses_latest_psnr_ssim_visual_pass() -> None:
    rows = {
        42: [
            boundary(seed=42, step=151_880, psnr=29.69, ssim=0.680, lpips=0.224)
        ],
        43: [
            boundary(seed=43, step=151_880, psnr=29.74, ssim=0.684, lpips=0.227)
        ],
        44: [
            boundary(seed=44, step=151_880, psnr=29.71, ssim=0.682, lpips=0.229)
        ],
    }
    decisions = {
        common.visual_key("007761", seed, 151_880): {
            "verdict": "pass",
            "change_from_previous": "no_improvement",
            "note": "reviewed",
        }
        for seed in rows
    }
    assert common.hard_gate_bootstrap_seeds("007761", rows, decisions) == (43, 44)

    decisions[common.visual_key("007761", 43, 151_880)]["verdict"] = "fail"
    assert common.hard_gate_bootstrap_seeds("007761", rows, decisions) == (44,)


def test_hard_gate_bootstrap_tolerates_psnr_oscillation_while_lpips_converges() -> None:
    rows = {
        43: [
            boundary(seed=43, step=151_880, psnr=29.74, ssim=0.684, lpips=0.227),
            boundary(seed=43, step=167_068, psnr=29.49, ssim=0.685, lpips=0.225),
        ]
    }
    decisions = {
        common.visual_key("007761", 43, step): {
            "verdict": "pass",
            "change_from_previous": "no_improvement",
            "note": "reviewed",
        }
        for step in (151_880, 167_068)
    }
    assert common.hard_gate_bootstrap_seeds("007761", rows, decisions) == (43,)


def test_visual_and_hard_gates_filter_before_selection() -> None:
    passing = boundary(step=15_188, psnr=29.7, ssim=0.668, lpips=0.22)
    bad_psnr = boundary(seed=43, step=15_188, psnr=29.699999)
    decisions = {
        common.visual_key("007754", 42, 15_188): {
            "verdict": "pass",
            "change_from_previous": "not_applicable",
            "note": "",
        },
        common.visual_key("007754", 43, 15_188): {
            "verdict": "pass",
            "change_from_previous": "not_applicable",
            "note": "",
        },
    }
    assert common.boundary_is_valid("007754", passing, decisions)
    assert not common.boundary_is_valid("007754", bad_psnr, decisions)
    decisions[common.visual_key("007754", 42, 15_188)]["verdict"] = "fail"
    assert not common.boundary_is_valid("007754", passing, decisions)


def test_preferred_quality_is_stricter_than_hard_minimum() -> None:
    fallback = boundary(
        step=151_880, psnr=29.87, ssim=0.6759, lpips=0.2151
    )
    preferred = boundary(
        step=167_068, psnr=29.88, ssim=0.676, lpips=0.215
    )
    assert fallback.numeric_pass
    assert not fallback.preferred_pass
    assert preferred.preferred_pass


def test_plateau_requires_two_complete_valid_visual_intervals() -> None:
    rows = [
        boundary(step=121_504, psnr=29.85, ssim=0.6750, lpips=0.2160),
        boundary(step=136_692, psnr=29.86, ssim=0.6755, lpips=0.2150),
        boundary(step=151_880, psnr=29.87, ssim=0.6760, lpips=0.2145),
    ]
    decisions = {
        common.visual_key("007754", 42, row.local_step): {
            "verdict": "pass",
            "change_from_previous": (
                "not_applicable" if index == 0 else "no_improvement"
            ),
            "note": "",
        }
        for index, row in enumerate(rows)
    }
    assert common.plateau_confirmed("007754", rows, decisions)
    decisions[common.visual_key("007754", 42, 151_880)][
        "change_from_previous"
    ] = "improved"
    assert not common.plateau_confirmed("007754", rows, decisions)


def test_native_comparison_preserves_crop_resolution(tmp_path: Path) -> None:
    def pair(path: Path, color: tuple[int, int, int]) -> None:
        image = Image.new("RGB", (2560, 720), color)
        image.save(path)

    leader = tmp_path / "leader.png"
    previous = tmp_path / "previous.png"
    target = tmp_path / "target.png"
    pair(leader, (10, 20, 30))
    pair(previous, (20, 30, 40))
    pair(target, (30, 40, 50))
    payload = common.build_native_comparison(
        frame="007754",
        seed=42,
        step=15_188,
        target_render=target,
        previous_accepted_render=previous,
        leader_render=leader,
        output_dir=tmp_path / "comparison",
    )
    assert payload["native_crop_size"] == [420, 380]
    crop = Image.open(payload["sources"]["target 007754 seed 42 step 15188"]["render"])
    assert crop.size == (420, 380)


def test_metrics_csv_requires_exact_header_and_unique_frames(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=common.METRICS_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "frame": "007740",
                "seed": "42",
                "parent_frame": "",
                "selected_step": "91128",
                "psnr": "29.84",
                "ssim": "0.669",
                "lpips": "0.219",
                "visual_gate": "pass",
                "checkpoint": "/checkpoint",
                "checkpoint_sha256": "sha",
            }
        )
    assert common.read_metrics_rows(path)[0]["frame"] == "007740"


def test_campaign_dry_run_uses_user_authorized_fast_seed_default() -> None:
    args = campaign.parse_args(["--dry-run"])
    payload = campaign.deterministic_dry_run(args)
    first = payload["commands"]["007754"]
    assert set(first) == {"43"}
    assert all("--parent-snapshot" in command for command in first.values())
    assert payload["initial_seed_policy"]["reason"] == "user-authorized wall-clock optimization"
    assert payload["tail_policy"].startswith("PSNR-window")


def test_campaign_can_pin_frozen_trajectory_source(tmp_path: Path) -> None:
    frozen = tmp_path / "run_lookcloser_temporal_finetune.py"
    args = campaign.parse_args(
        ["--dry-run", "--trajectory-script", str(frozen)]
    )
    payload = campaign.deterministic_dry_run(args)
    command = payload["commands"]["007754"]["43"]
    assert command[1] == str(frozen)


def test_pruning_removes_only_nonselected_checkpoints_and_is_resumable(
    tmp_path: Path,
) -> None:
    args = campaign.parse_args(["--campaign-root", str(tmp_path), "--resume"])
    root = (
        tmp_path
        / "007754"
        / "runs"
        / "seed-42-attempt-01"
        / "authoritative"
        / temporal.ARM_ID
        / "lookcloser"
        / "run"
        / "nerfstudio_models"
    )
    root.mkdir(parents=True)
    selected_path = root / "step-000151880.ckpt"
    removed_path = root / "step-000136692.ckpt"
    selected_path.write_bytes(b"selected")
    removed_path.write_bytes(b"removed")
    selected = boundary(
        step=151_880,
        psnr=29.9,
        root=root,
    )
    selected = common.Boundary(
        **{
            **selected.__dict__,
            "checkpoint": selected_path,
            "checkpoint_sha256": common.sha256_file(selected_path),
        }
    )
    store = campaign.CampaignStore(tmp_path / "campaign.json", resume=False)
    store.data["frames"]["007754"] = {}
    store.flush()

    campaign.prune_nonselected_checkpoints(
        args, store, frame="007754", selected=selected
    )
    campaign.prune_nonselected_checkpoints(
        args, store, frame="007754", selected=selected
    )

    assert selected_path.is_file()
    assert not removed_path.exists()
    manifest = json.loads((tmp_path / "007754" / "pruning.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["entries"][0]["sha256"]
