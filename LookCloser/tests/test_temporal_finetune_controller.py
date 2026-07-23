from __future__ import annotations

from collections import namedtuple
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_lookcloser_temporal_finetune as temporal


def metric(step: int, psnr: float, ssim: float, lpips: float) -> temporal.Metrics:
    return temporal.Metrics(step, psnr, ssim, lpips)


def evidence(step: int, psnr: float, ssim: float, lpips: float, artifact: float = 1.0):
    return temporal.BoundaryEvidence(
        metric(step, psnr, ssim, lpips),
        {name: 0.2 - step / 100_000_000 for name in temporal.CRITICAL_ROIS},
        artifact,
    )


def complete_protocol() -> dict:
    return {
        "status": "complete",
        "full_views": [{"eval_idx": index} for index in range(3)],
        "full_view_serious_count": 0,
        "roi_serious_count": 0,
        "tracking": {"ambiguous": False},
        "rois": [
            {"name": name, "metrics": {"lpips": 0.1}, "artifact": {"artifact_score": 0.0}}
            for name in temporal.CRITICAL_ROIS
        ],
    }


def test_selector_includes_exact_007_boundary_and_ignores_ssim() -> None:
    maximum = metric(60_752, 30.0, 0.99, 0.30)
    exact_tie = metric(45_564, 29.93, 0.01, 0.20)
    outside = metric(30_376, 29.929999, 1.0, 0.01)

    assert temporal.select_metrics([maximum, exact_tie, outside]) == exact_tie


def test_tail_resumes_selected_phase_a_checkpoint_not_last_plateau(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interval = temporal.INTERVAL
    phase_a = [
        temporal.Metrics(
            4 * interval,
            30.00,
            0.70,
            0.30,
            checkpoint=tmp_path / "maximum.ckpt",
            run_id="screen",
        ),
        temporal.Metrics(
            5 * interval,
            29.95,
            0.69,
            0.20,
            checkpoint=tmp_path / "selected.ckpt",
            run_id="screen",
        ),
        temporal.Metrics(
            6 * interval,
            29.80,
            0.71,
            0.19,
            checkpoint=tmp_path / "last.ckpt",
            run_id="screen",
        ),
    ]
    plateau = [
        evidence(4 * interval, 30.00, 0.7000, 0.2000),
        evidence(5 * interval, 30.01, 0.7005, 0.1980),
        evidence(6 * interval, 30.02, 0.7007, 0.1970),
    ]
    monkeypatch.setattr(temporal, "_metrics_with_checkpoints", lambda *_: phase_a)
    monkeypatch.setattr(temporal, "_boundary_evidence", lambda *_: plateau)
    captured = []

    def capture_tail(_args, _store, spec):
        captured.append(spec)
        raise RuntimeError("captured tail")

    monkeypatch.setattr(temporal, "run_training", capture_tail)
    args = temporal.parse_args([])
    store = object()

    with pytest.raises(RuntimeError, match="captured tail"):
        temporal.train_frame_recipe(
            args,
            store,
            frame="007747",
            parent_checkpoint=tmp_path / "leader.ckpt",
            parent_effective_step=91_128,
            lr=0.002,
            seed=42,
            prefix="transfer",
            initial_run_id="screen",
            traversal_warmup_steps=8_192,
        )

    assert len(captured) == 1
    assert captured[0].phase == "tail"
    assert captured[0].parent_checkpoint == tmp_path / "selected.ckpt"
    assert captured[0].target_local_step == 5 * interval + 2 * interval


def test_plateau_requires_two_complete_consecutive_intervals() -> None:
    rows = [
        evidence(30_376, 30.000, 0.7000, 0.2000),
        evidence(45_564, 30.020, 0.7005, 0.1980),
        evidence(60_752, 30.025, 0.7007, 0.1970),
    ]
    assert temporal.plateau_confirmed(rows)
    assert not temporal.plateau_confirmed(rows[-2:])
    assert not temporal.plateau_confirmed([rows[0], rows[1], evidence(60_753, 30.025, 0.7007, 0.1970)])
    assert not temporal.plateau_confirmed(
        [rows[0], evidence(45_564, 30.040, 0.7005, 0.1980), rows[2]]
    )


@pytest.mark.parametrize("mutation", ["missing_view", "missing_roi", "artifact"])
def test_missing_eval_roi_and_artifact_close_gate(mutation: str) -> None:
    protocol = complete_protocol()
    if mutation == "missing_view":
        protocol["full_views"].pop()
    elif mutation == "missing_roi":
        protocol["rois"].pop()
    else:
        protocol["full_view_serious_count"] = 1

    decision = temporal.quality_gate(protocol, previous_protocol=None, visual_pass=True)

    assert decision.outcome == "fail"


def test_tracking_and_midrange_roi_regression_are_ambiguous() -> None:
    previous = complete_protocol()
    current = complete_protocol()
    current["tracking"]["ambiguous"] = True
    for row in current["rois"]:
        row["metrics"]["lpips"] = 0.115

    decision = temporal.quality_gate(current, previous_protocol=previous, visual_pass=True)

    assert decision.outcome == "ambiguous"
    assert all(value == pytest.approx(0.015) for value in decision.critical_roi_regressions.values())


def test_large_roi_regression_fails_gate() -> None:
    previous = complete_protocol()
    current = complete_protocol()
    for row in current["rois"]:
        row["metrics"]["lpips"] = 0.121
    assert temporal.quality_gate(current, previous_protocol=previous, visual_pass=True).outcome == "fail"


def test_leader_metric_gate_uses_declared_inclusive_tolerances() -> None:
    leader = metric(91_128, 29.8, 0.68, 0.23)
    boundary = metric(60_752, 29.6, 0.67, 0.245)

    assert temporal.leader_metric_gate(boundary, leader).outcome == "pass"
    assert temporal.leader_metric_gate(metric(60_752, 29.599999, 0.67, 0.245), leader).outcome == "fail"
    assert temporal.leader_metric_gate(metric(60_752, 29.6, 0.669999, 0.245), leader).outcome == "fail"
    assert temporal.leader_metric_gate(metric(60_752, 29.6, 0.67, 0.245001), leader).outcome == "fail"


def test_combined_gate_preserves_ambiguity_and_failure_precedence() -> None:
    passing = temporal.GateDecision("pass", (), {})
    ambiguous = temporal.GateDecision("ambiguous", ("tracking",), {"hand_eval0": 0.012})
    failing = temporal.GateDecision("fail", ("metric",), {})

    combined = temporal.combine_gates(passing, ambiguous)
    assert combined.outcome == "ambiguous"
    assert combined.critical_roi_regressions == {"hand_eval0": 0.012}
    assert temporal.combine_gates(ambiguous, failing).outcome == "fail"


def test_dry_run_has_factorial_deterministic_model_only_commands(tmp_path: Path) -> None:
    args = temporal.parse_args(
        ["--dry-run", "--output-dir", str(tmp_path / "runs"), "--campaign", str(tmp_path / "campaign.json")]
    )
    first = temporal.deterministic_dry_run(args)
    second = temporal.deterministic_dry_run(args)

    assert first == second
    assert [
        (row["run"]["traversal_warmup_steps"], row["run"]["lr"])
        for row in first["lr_screen"]
    ] == [
        (warmup, lr)
        for warmup in temporal.TRAVERSAL_WARMUP_CANDIDATES
        for lr in temporal.LR_CANDIDATES
    ]
    assert all(row["run"]["load_mode"] == "model_parameters_only" for row in first["lr_screen"])
    assert all(row["run"]["target_local_step"] == 60_752 for row in first["lr_screen"])
    assert len({row["run"]["run_id"] for row in first["lr_screen"]}) == 6
    assert len({row["input_config"] for row in first["lr_screen"]}) == 6


def test_screen_candidate_overrides_are_validated_and_crossed(tmp_path: Path) -> None:
    args = temporal.parse_args(
        [
            "--output-dir",
            str(tmp_path / "runs"),
            "--campaign",
            str(tmp_path / "campaign.json"),
            "--lr-candidates",
            "0.00075,0.0015",
            "--traversal-warmup-candidates",
            "2048,6144",
        ]
    )
    assert [
        (spec.traversal_warmup_steps, spec.lr) for spec in temporal.lr_screen_specs(args)
    ] == [
        (2_048, 0.00075),
        (2_048, 0.0015),
        (6_144, 0.00075),
        (6_144, 0.0015),
    ]

    with pytest.raises(SystemExit):
        temporal.parse_args(["--lr-candidates", "0.001,0.001"])
    with pytest.raises(SystemExit):
        temporal.parse_args(["--traversal-warmup-candidates", "0"])


def test_run_local_traversal_warmup_override(tmp_path: Path) -> None:
    args = temporal.parse_args(
        ["--output-dir", str(tmp_path / "runs"), "--campaign", str(tmp_path / "campaign.json")]
    )
    spec = temporal.RunSpec(
        run_id="warmup8192",
        frame="007747",
        seed=42,
        lr=0.002,
        phase="diagnostic_warmup8192",
        feature_reweighting=1.0,
        fas_strength=1.0,
        load_mode="model_parameters_only",
        parent_checkpoint=args.leader_checkpoint,
        target_local_step=60_752,
        inherited_global_step=91_128,
        traversal_warmup_steps=8_192,
    )

    config, _, _ = temporal.configured_run(args, spec)
    model = config.pipeline.model
    assert model.adaptive_warmup_steps == 8_192
    assert model.occupancy_warmup_steps == 8_192
    assert model.occupancy_binary_warmup_steps == 8_192


def test_manifest_resume_is_atomic_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "campaign.json"
    store = temporal.CampaignStore(path, resume=False)
    store.data["status"] = "running"
    store.flush()
    with pytest.raises(temporal.InfrastructureError, match="already exists"):
        temporal.CampaignStore(path, resume=False)
    resumed = temporal.CampaignStore(path, resume=True)
    assert resumed.data["status"] == "running"


def test_disk_and_vram_guards_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(temporal.shutil, "disk_usage", lambda _: Usage(200, 150, 50))
    monkeypatch.setattr(temporal, "MIN_DISK_FREE_BYTES", 100)
    with pytest.raises(temporal.InfrastructureError, match="less than 100 GiB"):
        temporal.disk_guard(tmp_path)

    monkeypatch.setattr(
        temporal,
        "command_output",
        lambda *args, **kwargs: "0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 70000",
    )
    with pytest.raises(temporal.InfrastructureError, match="needs 81920 MiB"):
        temporal.vram_guard(3, {})


def test_rejected_parent_cannot_be_forwarded(tmp_path: Path) -> None:
    with pytest.raises(temporal.QualityStop, match="cannot be forwarded"):
        temporal.require_accepted_parent(
            {"007747": {"status": "rejected", "selected_checkpoint": str(tmp_path / "x.ckpt")}},
            "007747",
        )
