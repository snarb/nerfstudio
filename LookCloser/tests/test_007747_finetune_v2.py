from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


LOOKCLOSER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOOKCLOSER))
sys.path.insert(0, str(LOOKCLOSER / "scripts"))

from scripts import run_lookcloser_007747_finetune_v2 as v2  # noqa: E402
import static_target_roi_protocol as roi_protocol  # noqa: E402


def args(tmp_path: Path):
    return v2.parse_args(["--output-dir", str(tmp_path / "campaign")])


def protocol(tmp_path: Path, name: str, *, verdict: str = "pass", change: str = "no_improvement") -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "visual_gate": {
                    "verdict": verdict,
                    "change_from_previous": change,
                },
                "full_view_serious_count": 0,
                "roi": {"artifact": {"serious": False}},
            }
        ),
        encoding="utf-8",
    )
    return path


def boundary(
    tmp_path: Path,
    step: int,
    psnr: float,
    ssim: float,
    lpips: float,
    *,
    arm: str = "arm",
    verdict: str = "pass",
    change: str = "no_improvement",
) -> v2.Boundary:
    return v2.Boundary(
        arm_id=arm,
        local_step=step,
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
        checkpoint=tmp_path / f"step-{step:09d}.ckpt",
        eval_json=tmp_path / f"eval-{step}.json",
        protocol_json=protocol(
            tmp_path, f"protocol-{arm}-{step}", verdict=verdict, change=change
        ),
        eval_completed_wall_time_ns=step,
    )


def test_dry_run_is_deterministic_and_uses_original_hash23_leader(tmp_path: Path) -> None:
    parsed = args(tmp_path)
    first = v2.deterministic_dry_run(parsed)
    second = v2.deterministic_dry_run(parsed)

    assert first == second
    assert [
        (row["arm"]["lr_init"], row["arm"]["scheduler_max_steps"])
        for row in first["wave_a"]
    ] == [(0.0075, 200_000), (0.01, 200_000), (0.015, 200_000)]
    assert all(
        row["segment"]["parent_checkpoint"] == str(v2.LEADER_CHECKPOINT)
        for row in first["wave_a"]
    )
    assert all(row["segment"]["load_mode"] == "model_parameters_only" for row in first["wave_a"])


def test_initial_config_changes_only_whitelisted_fields(tmp_path: Path) -> None:
    parsed = args(tmp_path)
    arm = v2.wave_a_arms()[0]
    segment = v2.initial_segment(parsed, arm)
    config, differences = v2.configured_segment(parsed, segment)

    assert set(differences) <= v2.ALLOWED_CONFIG_DIFFS
    assert config.pipeline.datamanager.dataparser.data == v2.TARGET_DATASET
    assert config.load_checkpoint == v2.LEADER_CHECKPOINT
    assert config.checkpoint_load_mode == "model_parameters_only"
    assert config.load_optimizers is False
    assert config.load_scheduler is False
    assert config.resume_fields_lr_override is None
    assert config.pipeline.model.log2_hashmap_size == 23
    assert config.pipeline.model.feature_reweighting_strength == pytest.approx(0.3)
    assert config.pipeline.datamanager.pixel_sampler.fas_strength == pytest.approx(1.0)
    assert config.pipeline.datamanager.pixel_sampler.frequency_map_dir == "lookcloser_frequencies"
    assert config.get_base_dir() == segment.run_dir


def test_same_frame_resume_loads_target_state_without_lr_override(tmp_path: Path) -> None:
    parsed = args(tmp_path)
    arm = v2.Arm("resume", 0.01, 150_000, "authoritative")
    parent = tmp_path / "step-000060752.ckpt"
    segment = v2.authoritative_segment(
        parsed, arm, target_step=75_940, parent=parent
    )
    config, differences = v2.configured_segment(parsed, segment)

    assert set(differences) <= v2.ALLOWED_CONFIG_DIFFS
    assert config.checkpoint_load_mode == "resume"
    assert config.load_optimizers is True
    assert config.load_scheduler is True
    assert config.resume_fields_lr_override is None
    assert config.resume_reset_occupancy_grid is False
    assert config.resume_reset_frequency_grid is False
    assert config.checkpoint_load_parameter_hash_audit is False


def test_model_only_startup_audit_requires_fresh_scaler_and_scheduler() -> None:
    scaler_state = {
        "scale": 65_536.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2_000,
        "_growth_tracker": 0,
    }
    model = SimpleNamespace(
        occupancy_grid=SimpleNamespace(
            occs=torch.zeros(4),
            binaries=torch.ones((2, 2), dtype=torch.bool),
        ),
        freq_grid=SimpleNamespace(grid=torch.zeros((2, 2, 2))),
    )
    sampler = SimpleNamespace(sample_count=0)
    pipeline = SimpleNamespace(
        model=model,
        datamanager=SimpleNamespace(train_pixel_sampler=sampler),
        fas_sample_count_state=torch.zeros((), dtype=torch.int64),
        cumulative_point_samples=torch.zeros((), dtype=torch.int64),
    )
    optimizer = SimpleNamespace(state={}, param_groups=[{"lr": 0.01}])
    scheduler = SimpleNamespace(last_epoch=0)
    trainer = SimpleNamespace(
        checkpoint_load_audit={
            "local_start_step": 0,
            "optimizer_loaded": False,
            "scheduler_loaded": False,
            "scaler_loaded": False,
            "rng_loaded": False,
            "pipeline_buffers_loaded": False,
            "source_parameter_sha256": {"field": "abc"},
            "copied_parameter_sha256": {"field": "abc"},
            "fresh_state_assertions": {
                "occupancy_occs_zero": True,
                "occupancy_binary_constructor_true_count": 4,
                "frequency_grid_zero": True,
            },
        },
        optimizers=SimpleNamespace(
            optimizers={"fields": optimizer},
            schedulers={"fields": scheduler},
        ),
        pipeline=pipeline,
        grad_scaler=SimpleNamespace(state_dict=lambda: scaler_state),
        config=SimpleNamespace(
            grad_scaler_init_scale=65_536.0,
            grad_scaler_growth_interval=2_000,
            optimizers={
                "fields": {"optimizer": SimpleNamespace(lr=0.01)}
            },
        ),
    )

    audit = v2._startup_audit(trainer, expected_mode="model_parameters_only")
    assert all(audit["required"].values())

    trainer.grad_scaler = SimpleNamespace(
        state_dict=lambda: {**scaler_state, "_growth_tracker": 1}
    )
    with pytest.raises(v2.InfrastructureError, match="startup audit"):
        v2._startup_audit(trainer, expected_mode="model_parameters_only")


@pytest.mark.parametrize(
    ("lr_init", "horizon", "step", "expected"),
    [
        (0.01, 200_000, 15_188, 0.00704888),
        (0.01, 100_000, 60_752, 0.0006094807594025235),
        (0.015, 150_000, 60_752, 0.0019712662416441727),
    ],
)
def test_expected_log_linear_exponential_lr(
    lr_init: float, horizon: int, step: int, expected: float
) -> None:
    assert v2.expected_learning_rate(lr_init, horizon, step) == pytest.approx(
        expected, rel=1e-6
    )


def test_numeric_gate_is_inclusive_and_visual_gate_is_required(tmp_path: Path) -> None:
    passing = boundary(
        tmp_path,
        15_188,
        v2.PSNR_THRESHOLD,
        v2.SSIM_THRESHOLD,
        v2.LPIPS_THRESHOLD,
    )
    assert passing.numeric_pass
    assert v2.visual_pass(passing)

    pending = boundary(
        tmp_path,
        30_376,
        v2.PSNR_THRESHOLD,
        v2.SSIM_THRESHOLD,
        v2.LPIPS_THRESHOLD,
        verdict="pending",
    )
    assert pending.numeric_pass
    assert not v2.visual_pass(pending)
    with pytest.raises(v2.QualityStop, match="pending"):
        v2.require_reviewed([pending])


def test_selector_includes_exact_007_db_window_and_ignores_ssim(tmp_path: Path) -> None:
    maximum = boundary(tmp_path, 60_752, 30.0, 0.99, 0.30)
    exact_tie = boundary(tmp_path, 45_564, 29.93, 0.01, 0.20)
    outside = boundary(tmp_path, 30_376, 29.929999, 1.0, 0.01)

    assert v2.select_boundary([maximum, exact_tie, outside]) == exact_tie


def test_plateau_requires_two_consecutive_metric_and_visual_intervals(tmp_path: Path) -> None:
    rows = [
        boundary(tmp_path, 30_376, 30.000, 0.7000, 0.2000),
        boundary(tmp_path, 45_564, 30.020, 0.7005, 0.1980),
        boundary(tmp_path, 60_752, 30.025, 0.7007, 0.1970),
    ]
    assert v2.plateau_confirmed(rows)
    improved = boundary(
        tmp_path,
        60_752,
        30.025,
        0.7007,
        0.1970,
        change="improved",
    )
    assert not v2.plateau_confirmed([rows[0], rows[1], improved])


def test_manifest_file_check_rejects_extra_and_hash_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "files"
    directory.mkdir()
    good = directory / "good.bin"
    good.write_bytes(b"good")
    expected = {"good.bin": v2.sha256_file(good)}
    assert v2._verify_manifest_files(directory, expected, label="test")["count"] == 1

    (directory / "extra.bin").write_bytes(b"extra")
    with pytest.raises(v2.InfrastructureError, match="file set mismatch"):
        v2._verify_manifest_files(directory, expected, label="test")
    (directory / "extra.bin").unlink()
    with pytest.raises(v2.InfrastructureError, match="SHA-256"):
        v2._verify_manifest_files(directory, {"good.bin": "0" * 64}, label="test")


def test_native_scratch_contact_sheet_is_three_by_two(tmp_path: Path) -> None:
    shape = (380, 420, 3)
    images = [np.full(shape, value, dtype=np.uint8) for value in range(6)]
    path = tmp_path / "sheet.png"
    roi_protocol.save_contact_sheet(
        images[0],
        images[1],
        images[4],
        images[5],
        path,
        scratch_gt=images[2],
        scratch_render=images[3],
    )
    with Image.open(path) as sheet:
        assert sheet.size == (864, 1244)


def test_protocol_accepts_historical_scratch_gt_but_binds_target_gt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    render_dir = tmp_path / "candidate"
    leader_dir = tmp_path / "leader"
    scratch_dir = tmp_path / "scratch"
    for directory in (images, render_dir, leader_dir, scratch_dir):
        directory.mkdir(parents=True)

    Image.fromarray(np.full((480, 1120, 3), 64, dtype=np.uint8)).save(
        images / "frame_eval_00001.jpg"
    )
    target_gt = np.asarray(
        Image.open(images / "frame_eval_00001.jpg").convert("RGB")
    )
    for index in range(3):
        target_render = np.full_like(target_gt, 72 + index)
        Image.fromarray(np.concatenate([target_gt, target_render], axis=1)).save(
            render_dir / f"eval_img_{index:04d}.png"
        )
    leader_gt = np.full_like(target_gt, 48)
    scratch_gt = np.full_like(target_gt, 80)
    Image.fromarray(np.concatenate([leader_gt, leader_gt], axis=1)).save(
        leader_dir / "eval_img_0000.png"
    )
    Image.fromarray(np.concatenate([scratch_gt, scratch_gt], axis=1)).save(
        scratch_dir / "eval_img_0000.png"
    )

    monkeypatch.setattr(
        roi_protocol,
        "artifact_result",
        lambda *_: {
            "serious": False,
            "artifact_score": 0.0,
            "serious_artifact_score": 0.0,
            "artifact_count": 0,
            "largest_area": 0,
        },
    )
    monkeypatch.setattr(
        roi_protocol,
        "roi_metrics",
        lambda *_: {
            "psnr": 30.0,
            "ssim": 0.7,
            "lpips": 0.2,
            "gradient_ratio": 1.0,
            "gradient_mae": 0.0,
        },
    )
    monkeypatch.setattr(
        roi_protocol,
        "LearnedPerceptualImagePatchSimilarity",
        lambda **_: object(),
    )
    result = roi_protocol.build_protocol(
        SimpleNamespace(
            frame="007747",
            dataset=dataset,
            render_dir=render_dir,
            out_dir=tmp_path / "protocol",
            leader_render_dir=leader_dir,
            scratch_render_dir=scratch_dir,
            visual_verdict="pending",
            visual_note="",
            visual_change="not_applicable",
        )
    )

    assert result["roi"]["scratch_gt_matches_target_revision"] is False
    assert Path(result["contact_sheet"]).name == "contact_hands_chain_3x2.png"


def test_active_dataset_revision_hashes_are_complete() -> None:
    result = v2.dataset_preflight()
    assert result["revision_sha256"] == v2.EXPECTED_REVISION_SHA256
    assert result["train_images"] == 66
    assert result["eval_images"] == 3
    assert result["jpeg"]["count"] == 69
    assert result["frequency_maps"]["count"] == 132
