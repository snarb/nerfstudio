#!/usr/bin/env python3
"""Build a temporal eval-view video from selected Nerfstudio eval renders."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2


REPORT = Path("/home/brans/repos/nerfstudio/LookCloser/experiments/training_on_video.md")


def parse_rows(report: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in report.read_text(errors="replace").splitlines():
        if not line.startswith("| "):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 7 or not re.fullmatch(r"\d{6}", parts[0]):
            continue
        rows.append(
            {
                "frame": parts[0],
                "label": parts[1],
                "psnr": parts[2],
                "ssim": parts[3],
                "lpips": parts[4],
                "checkpoint": parts[5].strip("`"),
                "renders": parts[6].strip("`"),
            }
        )
    return rows


def render_half(image):
    height, width = image.shape[:2]
    if width % 2 != 0:
        raise ValueError(f"Expected side-by-side eval image with even width, got {width}x{height}")
    return image[:, width // 2 :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--eval-index", type=int, default=0)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    rows = parse_rows(args.report)
    if not rows:
        raise SystemExit(f"No frame rows found in {args.report}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / f"frames_eval{args.eval_index:04d}_fps{args.fps:g}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / (
        args.output_name or f"eval_view_{args.eval_index:04d}_fps{str(args.fps).replace('.', 'p')}.mp4"
    )

    written: list[dict[str, str]] = []
    writer = None
    try:
        for i, row in enumerate(rows):
            src = Path(row["renders"]) / f"eval_img_{args.eval_index:04d}.png"
            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(src)
            frame = render_half(image)
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {video_path}")
            out_frame = frames_dir / f"frame_{i:06d}_{row['frame']}.png"
            cv2.imwrite(str(out_frame), frame)
            writer.write(frame)
            written.append({**row, "source": str(src), "frame_png": str(out_frame)})
    finally:
        if writer is not None:
            writer.release()

    metadata = {
        "video": str(video_path),
        "frames_dir": str(frames_dir),
        "report": str(args.report),
        "eval_index": args.eval_index,
        "fps": args.fps,
        "frame_count": len(written),
        "frames": written,
    }
    metadata_path = video_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(video_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
