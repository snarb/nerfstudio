#!/usr/bin/env python3
"""Build an explicit fixed-pose COLMAP PatchMatch source-view configuration.

COLMAP's ``__auto__`` source selection depends on sparse 3D tracks.  A calibrated
capture may intentionally provide poses without retriangulating sparse points,
so this utility chooses nearby train cameras directly from Nerfstudio c2w poses.
Eval images are never admitted when ``--split train`` is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np


def normalized_name(value: str) -> str:
    name = PurePosixPath(value).as_posix()
    while name.startswith("./"):
        name = name[2:]
    return name


def camera_center(frame: dict[str, object]) -> np.ndarray:
    transform = np.asarray(frame["transform_matrix"], dtype=np.float64)
    if transform.shape == (3, 4):
        transform = np.vstack((transform, [0.0, 0.0, 0.0, 1.0]))
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid transform for {frame.get('file_path')!r}")
    return transform[:3, 3]


def explicit_source_views(
    frames: list[dict[str, object]],
    *,
    source_count: int,
) -> dict[str, list[str]]:
    """Choose nearest non-identical calibrated camera centers for every frame."""

    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if len(frames) < 2:
        raise ValueError("At least two frames are required")
    names = [normalized_name(str(frame["file_path"])) for frame in frames]
    if len(set(names)) != len(names):
        raise ValueError("Frame paths must be unique")
    centers = np.stack([camera_center(frame) for frame in frames])
    if not np.isfinite(centers).all():
        raise ValueError("Camera centers must be finite")
    result: dict[str, list[str]] = {}
    count = min(source_count, len(frames) - 1)
    for index, name in enumerate(names):
        distances = np.linalg.norm(centers - centers[index], axis=-1)
        order = np.argsort(distances, kind="stable")
        sources = [names[int(other)] for other in order if int(other) != index][:count]
        result[name] = sources
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=12)
    parser.add_argument("--split", choices=("all", "train"), default="train")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads((args.data / "transforms.json").read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("transforms.json contains no frames")
    selected = frames
    if args.split == "train":
        raw_train = payload.get("train_filenames")
        if not isinstance(raw_train, list) or not raw_train or not all(isinstance(value, str) for value in raw_train):
            raise ValueError("--split train requires a non-empty explicit train_filenames list")
        train_names = {normalized_name(value) for value in raw_train}
        selected = [frame for frame in frames if normalized_name(str(frame["file_path"])) in train_names]
        if {normalized_name(str(frame["file_path"])) for frame in selected} != train_names:
            raise ValueError("train_filenames and frames do not match")
    mapping = explicit_source_views(selected, source_count=args.source_count)
    lines: list[str] = []
    for reference, sources in mapping.items():
        lines.extend((reference, ", ".join(sources)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote references={len(mapping)} sources_per_reference={len(next(iter(mapping.values())))} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
