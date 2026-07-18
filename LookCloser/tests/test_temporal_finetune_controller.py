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


def test_dry_run_has_three_deterministic_model_only_commands(tmp_path: Path) -> None:
    args = temporal.parse_args(
        ["--dry-run", "--output-dir", str(tmp_path / "runs"), "--campaign", str(tmp_path / "campaign.json")]
    )
    first = temporal.deterministic_dry_run(args)
    second = temporal.deterministic_dry_run(args)

    assert first == second
    assert [row["run"]["lr"] for row in first["lr_screen"]] == list(temporal.LR_CANDIDATES)
    assert all(row["run"]["load_mode"] == "model_parameters_only" for row in first["lr_screen"])
    assert all(row["run"]["target_local_step"] == 60_752 for row in first["lr_screen"])
    assert len({tuple(row["command"]) for row in first["lr_screen"]}) == 3


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
