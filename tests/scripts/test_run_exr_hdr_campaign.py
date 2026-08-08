"""Regression tests for the staged native-EXR campaign controller."""

import csv
import importlib.util
from pathlib import Path


CAMPAIGN = Path(__file__).resolve().parents[2] / "LookCloser" / "scripts" / "run_exr_hdr_campaign.py"


def _load_campaign():
    spec = importlib.util.spec_from_file_location("exr_hdr_campaign_for_test", CAMPAIGN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, name: str, rows: list[tuple[int, float, float, float]]) -> dict:
    run_dir = tmp_path / name
    run_dir.mkdir()
    summary = run_dir / "run_summary.json"
    summary.write_text("{}", encoding="utf-8")
    with (run_dir / "metrics_compact.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "eval_all_psnr", "eval_all_ssim", "eval_all_lpips"))
        writer.writeheader()
        for step, psnr, ssim, lpips in rows:
            writer.writerow(
                {
                    "step": step,
                    "eval_all_psnr": psnr,
                    "eval_all_ssim": ssim,
                    "eval_all_lpips": lpips,
                }
            )
    _, psnr, ssim, lpips = rows[-1]
    return {
        "summary_path": str(summary),
        "eval": {"hdr": {"aggregate": {"psnr": psnr, "ssim": ssim, "lpips": lpips}}},
    }


def test_one_boundary_candidate_can_be_rejected_without_wasting_second_segment(tmp_path: Path) -> None:
    campaign = _load_campaign()
    anchor = _run(tmp_path, "anchor", [(15188, 32.8, 0.88, 0.29), (30376, 33.3, 0.89, 0.26)])
    rejected = _run(tmp_path, "rejected", [(15188, 30.0, 0.80, 0.50)])
    manifest = {"runs": {"anchor": anchor, "rejected": rejected}}

    assert campaign.select_best(["anchor", "rejected"], manifest, "map-screen") == "anchor"
    gate = manifest["trajectory_gates"]["map-screen"]
    assert gate["early_rejected_after_first_boundary"] == ["rejected"]
    assert gate["observed_common_points"] == 2


def test_alias_candidate_reuses_exact_measured_summary(tmp_path: Path) -> None:
    campaign = _load_campaign()
    source = _run(tmp_path, "source", [(15188, 32.8, 0.88, 0.29), (30376, 33.3, 0.89, 0.26)])
    manifest = {"runs": {"source": source}}

    assert campaign.alias_candidate(
        manifest,
        tag="alias",
        source_tag="source",
        stage="tune",
        method="calibrated",
        extra=("--pq-linear-anchor-weight", "0.0"),
    )
    assert manifest["runs"]["alias"]["alias_of"] == "source"
    assert manifest["runs"]["alias"]["summary_path"] == source["summary_path"]
