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
    assert common.LPIPS_MAX == 0.217
    assert common.BUDGET_NUMERATOR == 13
    assert common.BUDGET_DENOMINATOR == 10
    assert common.SCHEDULER_MAX_STEPS == 300_000
    assert common.TAIL_MAX_STEPS == 600_000
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


def test_short_budget_stops_initial_trajectory_at_complete_boundary(
    tmp_path: Path,
) -> None:
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
            "--initial-target-step",
            "45564",
        ]
    )
    temporal.configure_v2(args)
    segments = temporal.initial_segments(args)
    assert [row.target_step for row in segments] == [45_564]
    assert segments[0].load_mode == "model_parameters_only"


def test_initial_target_must_be_complete_boundary() -> None:
    with pytest.raises(SystemExit):
        temporal.parse_args(
            [
                "--target-frame",
                "007754",
                "--parent-snapshot",
                str(common.DATA_ROOT / "007747" / "snapshot"),
                "--seed",
                "42",
                "--initial-target-step",
                "45565",
            ]
        )


def test_visual_recovery_requires_frame_and_complete_boundary() -> None:
    with pytest.raises(SystemExit):
        campaign.parse_args(
            [
                "--visual-recovery-frame",
                "007817",
            ]
        )
    with pytest.raises(SystemExit):
        campaign.parse_args(
            [
                "--visual-recovery-frame",
                "007817",
                "--visual-recovery-through-step",
                "45565",
            ]
        )
    args = campaign.parse_args(
        [
            "--start-frame",
            "007817",
            "--end-frame",
            "007817",
            "--visual-recovery-frame",
            "007817",
            "--visual-recovery-through-step",
            "151880",
        ]
    )
    assert args.visual_recovery_frame == "007817"
    assert args.visual_recovery_through_step == 151_880


def test_visual_recovery_creates_fresh_attempt_without_mutating_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = campaign.parse_args(
        [
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--start-frame",
            "007817",
            "--end-frame",
            "007817",
            "--visual-recovery-frame",
            "007817",
            "--visual-recovery-through-step",
            "151880",
            "--resume",
        ]
    )
    store = campaign.CampaignStore(args.campaign_root / "campaign.json", resume=True)
    old_run = tmp_path / "immutable-attempt-01"
    store.data["frames"]["007817"] = {
        "status": "quality_failed",
        "parent_frame": "007810",
        "parent_snapshot": str(common.DATA_ROOT / "007810" / "snapshot"),
        "parent_checkpoint_sha256": "parent-sha",
        "quality_failure": {"at": "then", "reason": "visual failure"},
        "attempt": 1,
        "active_runs": {"43": str(old_run)},
        "training_budget": common.training_budget(45_564),
        "initial_target_step": 45_564,
    }
    store.flush()
    monkeypatch.setattr(
        campaign.common,
        "freeze_dataset_manifest",
        lambda *args, **kwargs: {"frozen": True},
    )
    monkeypatch.setattr(
        campaign,
        "gpu_preflight",
        lambda *args, **kwargs: {"selected": 1},
    )
    monkeypatch.setattr(
        campaign,
        "storage_preflight",
        lambda *args, **kwargs: {"projected": True},
    )

    captured = {}

    class RecoveryStarted(RuntimeError):
        pass

    def stop_before_training(*args, **kwargs):
        captured.update(kwargs)
        raise RecoveryStarted

    monkeypatch.setattr(campaign, "run_wave", stop_before_training)
    parent = {
        "frame": "007810",
        "snapshot": str(common.DATA_ROOT / "007810" / "snapshot"),
        "checkpoint": str(common.snapshot_checkpoint(common.DATA_ROOT / "007810" / "snapshot")),
        "checkpoint_sha256": "parent-sha",
        "checkpoint_step": 45_564,
    }
    with pytest.raises(RecoveryStarted):
        campaign.process_frame(
            args, store, frame="007817", parent_info=parent
        )

    record = store.data["frames"]["007817"]
    assert record["attempt"] == 2
    assert record["active_runs"]["43"].endswith("seed-43-attempt-02")
    assert record["superseded_active_runs"][0]["active_runs"] == {
        "43": str(old_run)
    }
    assert record["visual_recovery_override"]["initial_target_step"] == 151_880
    assert captured["extend"] is False
    assert captured["initial_target_step"] == 151_880


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


def test_budget_tail_keeps_one_visual_pass_seed_before_psnr_gate() -> None:
    rows = {
        42: [
            boundary(seed=42, step=151_880, psnr=29.31, ssim=0.680, lpips=0.234)
        ],
        43: [
            boundary(seed=43, step=151_880, psnr=29.37, ssim=0.685, lpips=0.232)
        ],
    }
    decisions = {
        common.visual_key("007768", seed, 151_880): {
            "verdict": "pass",
            "change_from_previous": "no_improvement",
            "note": "reviewed",
        }
        for seed in rows
    }
    assert common.hard_gate_bootstrap_seeds("007768", rows, decisions) == ()
    assert common.budget_tail_seed(
        "007768", rows, decisions, maximum_eval_step=273_384
    ) == (43,)


def test_budget_tail_stops_at_cap_or_visual_failure() -> None:
    row = boundary(seed=43, step=273_384, psnr=29.5, ssim=0.685, lpips=0.225)
    rows = {43: [row]}
    decisions = {
        common.visual_key("007768", 43, 273_384): {
            "verdict": "pass",
            "change_from_previous": "no_improvement",
            "note": "reviewed",
        }
    }
    assert common.budget_tail_seed(
        "007768", rows, decisions, maximum_eval_step=273_384
    ) == ()
    decisions[common.visual_key("007768", 43, 273_384)]["verdict"] = "fail"
    assert common.budget_tail_seed(
        "007768", rows, decisions, maximum_eval_step=288_572
    ) == ()


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


def test_hard_gate_bootstrap_tolerates_one_lpips_regression_with_net_progress() -> None:
    rows = {
        42: [
            boundary(seed=42, step=167_068, psnr=29.75, ssim=0.683, lpips=0.2253),
            boundary(seed=42, step=197_444, psnr=29.59, ssim=0.682, lpips=0.2238),
            boundary(seed=42, step=212_632, psnr=29.47, ssim=0.683, lpips=0.2222),
            boundary(seed=42, step=227_820, psnr=29.43, ssim=0.682, lpips=0.2226),
        ]
    }
    decisions = {
        common.visual_key("007761", 42, row.local_step): {
            "verdict": "pass",
            "change_from_previous": "no_improvement",
            "note": "reviewed",
        }
        for row in rows[42]
    }
    assert common.hard_gate_bootstrap_seeds("007761", rows, decisions) == (42,)


def test_hard_gate_bootstrap_finishes_last_scheduled_boundary() -> None:
    rows = {
        44: [
            boundary(seed=44, step=258_196, psnr=29.71, ssim=0.682, lpips=0.2238),
            boundary(seed=44, step=273_384, psnr=29.49, ssim=0.684, lpips=0.2242),
        ]
    }
    decisions = {
        common.visual_key("007761", 44, row.local_step): {
            "verdict": "pass",
            "change_from_previous": "no_improvement",
            "note": "reviewed",
        }
        for row in rows[44]
    }
    assert common.hard_gate_bootstrap_seeds("007761", rows, decisions) == (44,)


def test_hard_gate_bootstrap_can_tail_at_minimum_lr_after_scheduler_horizon() -> None:
    rows = {
        43: [
            boundary(seed=43, step=258_196, psnr=29.72, ssim=0.681, lpips=0.2212),
            boundary(seed=43, step=273_384, psnr=29.59, ssim=0.681, lpips=0.2205),
            boundary(seed=43, step=288_572, psnr=29.48, ssim=0.684, lpips=0.2213),
        ]
    }
    decisions = {
        common.visual_key("007761", 43, row.local_step): {
            "verdict": "pass",
            "change_from_previous": "no_improvement",
            "note": "reviewed",
        }
        for row in rows[43]
    }
    assert common.hard_gate_bootstrap_seeds("007761", rows, decisions) == (43,)


def test_visual_and_hard_gates_filter_before_selection() -> None:
    passing = boundary(step=15_188, psnr=29.7, ssim=0.668, lpips=0.217)
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


def test_budget_uses_previous_selected_step_and_selects_best_visual_fallback() -> None:
    budget = common.training_budget(212_632)
    assert budget["maximum_step"] == 276_421
    assert budget["maximum_eval_step"] == 273_384
    rows = [
        boundary(seed=42, step=167_068, psnr=29.751, ssim=0.683, lpips=0.2253),
        boundary(seed=43, step=212_632, psnr=29.730, ssim=0.682, lpips=0.2222),
        boundary(seed=43, step=273_384, psnr=29.589, ssim=0.681, lpips=0.2205),
        boundary(seed=43, step=288_572, psnr=29.702, ssim=0.681, lpips=0.2202),
    ]
    decisions = {
        common.visual_key("007761", row.seed, row.local_step): {
            "verdict": "pass",
            "change_from_previous": "no_improvement",
            "note": "reviewed",
        }
        for row in rows
    }
    selected = common.select_budget_fallback(
        "007761",
        rows,
        decisions,
        maximum_eval_step=budget["maximum_eval_step"],
    )
    assert (selected.seed, selected.local_step) == (43, 212_632)


def test_budget_fallback_preserves_psnr_window_lpips_rule_without_numeric_pass() -> None:
    rows = [
        boundary(step=30_376, psnr=29.002, ssim=0.692, lpips=0.302),
        boundary(step=136_692, psnr=28.974, ssim=0.686, lpips=0.245),
        boundary(step=197_444, psnr=28.916, ssim=0.686, lpips=0.241),
    ]
    decisions = {
        common.visual_key("007775", row.seed, row.local_step): {
            "verdict": "pass",
            "change_from_previous": "no_improvement",
            "note": "reviewed",
        }
        for row in rows
    }
    selected = common.select_budget_fallback(
        "007775", rows, decisions, maximum_eval_step=197_444
    )
    assert selected.local_step == 136_692


def test_existing_comparison_uses_incremental_source_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    render = tmp_path / "eval_img_0000.png"
    render.write_bytes(b"render-v1")
    comparison_json = tmp_path / "comparison.json"
    common.atomic_json(
        comparison_json,
        {
            "sources": {
                "target 007761 seed 43 step 303760": {
                    "source": str(render),
                    "source_sha256": common.sha256_file(render),
                }
            }
        },
    )
    original = common.sha256_file
    calls = 0

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(campaign.common, "sha256_file", counted)
    campaign._load_or_validate_comparison(comparison_json, render)
    assert calls == 1
    campaign._load_or_validate_comparison(comparison_json, render)
    assert calls == 1

    render.write_bytes(b"render-v2-changed")
    with pytest.raises(common.InfrastructureError):
        campaign._load_or_validate_comparison(comparison_json, render)
    assert calls == 2


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


def test_tail_candidates_are_limited_to_authorized_fast_seeds() -> None:
    args = campaign.parse_args(["--initial-seeds", "43"])
    assert campaign.authorized_tail_seeds(args, (42, 43, 44)) == (43,)


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
