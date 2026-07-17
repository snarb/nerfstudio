#!/usr/bin/env python3
"""Generate the temporal video training report from current run artifacts."""

from __future__ import annotations

import csv
import datetime as dt
import math
import os
import re
from pathlib import Path


REPO = Path("/home/brans/repos/nerfstudio/LookCloser")
DATASET = Path("/home/brans/temporal_perframe_stride7_45f")
SOURCE_REPORT = REPO / "experiments/temporal_lookcloser_transfer.md"
OUT_REPORT = REPO / "experiments/training_on_video.md"
DRIVER_LOG = Path("/home/brans/lookcloser_temporal_runs/temporal_transfer_resume_007803_driver.log")


def fmt_metric(value: str) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return value


def dataset_frames() -> list[str]:
    return sorted(
        p.name for p in DATASET.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)
    )


def completed_rows() -> list[dict[str, str]]:
    rows_by_frame: dict[str, dict[str, str]] = {}
    if not SOURCE_REPORT.exists():
        return []

    for line in SOURCE_REPORT.read_text(errors="replace").splitlines():
        if not line.startswith("| "):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 8 or parts[0] in {"---", "Section"}:
            continue

        section, frame, label, psnr, ssim, lpips, ckpt, renders = parts[:8]
        if not re.fullmatch(r"\d{6}", frame):
            continue
        if section == "Sanity" and frame == "007740":
            if frame in rows_by_frame:
                continue
        elif section == "LR sweep" and not (frame == "007747" and label == "const5e-4"):
            continue
        elif not section.startswith("Frame") and not (
            section == "LR sweep" and label == "const5e-4"
        ):
            continue

        rows_by_frame[frame] = {
            "frame": frame,
            "label": label,
            "psnr": fmt_metric(psnr),
            "ssim": fmt_metric(ssim),
            "lpips": fmt_metric(lpips),
            "ckpt": ckpt.strip("`"),
            "renders": renders.strip("`"),
        }

    return [rows_by_frame[k] for k in sorted(rows_by_frame)]


def active_run() -> dict[str, object] | None:
    if not DRIVER_LOG.exists():
        return None

    matches = re.findall(
        r"train frame=(\d+) label=([^ ]+) run_dir=([^\n]+)",
        DRIVER_LOG.read_text(errors="replace"),
    )
    if not matches:
        return None

    frame, label, run_dir = matches[-1]
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics_compact.csv"
    last_step = "none"
    evals: list[dict[str, str]] = []
    if metrics_path.exists():
        metric_rows = list(csv.DictReader(metrics_path.open()))
        if metric_rows:
            last_step = metric_rows[-1].get("step", "none")
        evals = [
            row
            for row in metric_rows
            if row.get("eval_all_psnr") or row.get("eval_all_ssim") or row.get("eval_all_lpips")
        ]

    ckpt_dir = run_path / "nerfstudio_models"
    ckpts = [p.name for p in sorted(ckpt_dir.glob("step-*.ckpt"))] if ckpt_dir.exists() else []
    return {
        "frame": frame,
        "label": label,
        "run_dir": str(run_path),
        "last_step": last_step,
        "evals": evals,
        "ckpts": ckpts,
    }


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def driver_is_running() -> bool:
    pid_path = DRIVER_LOG.with_name("temporal_transfer_resume_007803_driver.pid")
    if not pid_path.exists():
        return False
    try:
        return is_process_running(int(pid_path.read_text().strip()))
    except Exception:
        return False


def rough_eta_minutes(active: dict[str, object] | None, pending_after: list[str]) -> int:
    full_frame_min = 52
    if not active or not str(active.get("last_step", "")).isdigit():
        return len(pending_after) * full_frame_min

    last_step = int(str(active["last_step"]))
    active_min = full_frame_min
    match = re.search(r"_from_(\d+)_", Path(str(active["run_dir"])).name)
    if match:
        init_step = int(match.group(1))
        remaining_steps = max(0, init_step + 50_000 - last_step)
        active_min = max(8, math.ceil(remaining_steps / 1000) + 8)
    return active_min + len(pending_after) * full_frame_min


def main() -> None:
    frames = dataset_frames()
    rows = completed_rows()
    active = active_run()
    completed_video = [row for row in rows if row["frame"] != "007740"]
    completed_frames = {row["frame"] for row in completed_video}
    all_video_frames_complete = all(
        frame == "007740" or frame in completed_frames for frame in frames
    )
    if all_video_frames_complete and not driver_is_running():
        active = None

    pending_after: list[str]
    if active and active["frame"] in frames:
        pending_after = frames[frames.index(str(active["frame"])) + 1 :]
    else:
        pending_after = [frame for frame in frames if frame not in completed_frames and frame != "007740"]

    eta_min = rough_eta_minutes(active, pending_after)
    full_frame_min = 52

    lines = [
        "# Training on Video",
        "",
        f"Generated: {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Status",
        "",
        f"- Dataset: `{DATASET}`",
        f"- Dataset frame directories: {len(frames)}",
        "- Leader frame `007740` is reused from the static leader and is not retrained in the chain.",
    ]
    if active:
        lines.extend(
            [
                f"- Active frame: `{active['frame']}` (`{active['label']}`), last step `{active['last_step']}`",
                f"- Active run: `{active['run_dir']}`",
            ]
        )
    else:
        lines.append("- Active frame: None; temporal transfer chain is complete.")
    lines.extend(
        [
            f"- Completed video frames in report, excluding leader sanity: {len(completed_video)}",
            f"- Remaining frames after active: {len(pending_after)}",
        ]
    )
    if eta_min:
        lines.append(
            f"- Rough ETA to finish all remaining frames: about {eta_min / 60:.1f} hours "
            f"({eta_min} minutes), assuming ~{full_frame_min} min/full frame."
        )
    lines.append("")

    if active:
        ckpts = ", ".join(active["ckpts"]) if active["ckpts"] else "none yet"
        lines.extend(
            [
                "## Current Active Frame",
                "",
                "| Frame | Last step | Eval count | Checkpoints saved |",
                "|---|---:|---:|---|",
                f"| {active['frame']} | {active['last_step']} | {len(active['evals'])} | {ckpts} |",
            ]
        )
        if active["evals"]:
            lines.extend(["", "| Eval step | PSNR | SSIM | LPIPS | Status |", "|---:|---:|---:|---:|---|"])
            for row in active["evals"][-8:]:
                lines.append(
                    f"| {row.get('step', '')} | {fmt_metric(row.get('eval_all_psnr', ''))} | "
                    f"{fmt_metric(row.get('eval_all_ssim', ''))} | "
                    f"{fmt_metric(row.get('eval_all_lpips', ''))} | {row.get('eval_status', '')} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Metrics by Frame",
            "",
            "| Frame | Label | PSNR | SSIM | LPIPS | Selected checkpoint | Renders |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['frame']} | {row['label']} | {row['psnr']} | {row['ssim']} | {row['lpips']} | "
            f"`{row['ckpt']}` | `{row['renders']}` |"
        )

    pending_text = "`" + "`, `".join(pending_after) + "`" if pending_after else "None"
    lines.extend(
        [
            "",
            "## Pending Frames",
            "",
            pending_text,
            "",
            "## Notes",
            "",
            "- Default temporal transfer LR is constant `5e-4`, selected by the LR sweep on `007747`.",
            "- All checkpoints are intentionally preserved; selected checkpoints and hard-stop checkpoints are not deleted.",
            "- This report is regenerated periodically while the temporal transfer driver is running.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n")
    print(OUT_REPORT)


if __name__ == "__main__":
    main()
