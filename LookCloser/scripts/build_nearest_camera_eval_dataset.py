#!/usr/bin/env python3
"""Build an immutable nearest-camera train subset around one held-out eval view.

The source dataset is never modified.  Camera selection uses Euclidean camera-centre
distance and stable source order; it does not depend on scene-specific camera names,
frame counts, image dimensions, or raster extension.  Masks are deliberately rejected
because this diagnostic is intended to preserve full-frame RGB supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SUPPORTED_IMAGE_SUFFIXES = {".exr", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def resolve_dataset_file(root: Path, declared: str, *, label: str) -> tuple[Path, Path]:
    if not isinstance(declared, str) or not declared:
        raise ValueError(f"{label} must be a non-empty dataset-relative string")
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay inside the dataset: {declared!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the dataset: {declared!r}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"Declared {label} does not exist: {resolved}")
    return relative, resolved


def camera_center(frame: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
    if matrix.shape not in {(3, 4), (4, 4)} or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid camera transform for {frame.get('file_path')!r}: shape={matrix.shape}")
    return matrix[:3, 3]


def find_unique_frame(frames: list[dict[str, Any]], file_path: str) -> tuple[int, dict[str, Any]]:
    matches = [(index, frame) for index, frame in enumerate(frames) if frame.get("file_path") == file_path]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one frame with file_path={file_path!r}, got {len(matches)}")
    return matches[0]


def is_filename_eval(frame: dict[str, Any]) -> bool:
    return "eval" in Path(str(frame.get("file_path", ""))).name.lower()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-frame-file", required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument(
        "--allow-source-eval-train",
        action="store_true",
        help="Allow other filename-eval frames to become train candidates; off by default.",
    )
    parser.add_argument(
        "--preserve-normalization-context",
        action="store_true",
        help=(
            "Retain every source frame for pose normalization while explicit train/val/test filename lists "
            "restrict RGB supervision to the selected leave-one-out split."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.train_count <= 0:
        raise ValueError("train-count must be positive")
    if output.exists():
        raise FileExistsError(output)
    transforms = source / "transforms.json"
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in {transforms}")

    eval_index, eval_frame = find_unique_frame(frames, args.eval_frame_file)
    eval_center = camera_center(eval_frame)
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for index, frame in enumerate(frames):
        if index == eval_index:
            continue
        if not args.allow_source_eval_train and is_filename_eval(frame):
            continue
        candidates.append((float(np.linalg.norm(camera_center(frame) - eval_center)), index, frame))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if args.train_count > len(candidates):
        raise ValueError(f"Requested {args.train_count} train cameras, only {len(candidates)} are eligible")
    selected = candidates[: args.train_count]

    frames_to_validate = frames if args.preserve_normalization_context else [
        *[frame for _, _, frame in selected],
        eval_frame,
    ]
    for frame in frames_to_validate:
        if frame.get("mask_path"):
            raise ValueError("Nearest-camera held-out diagnostic forbids masks")
        _, image = resolve_dataset_file(source, str(frame.get("file_path", "")), label="image path")
        if image.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension for {image}")

    output.mkdir(parents=True)
    sparse_relative: str | None = None
    if payload.get("ply_file_path") is not None:
        relative, source_ply = resolve_dataset_file(source, payload["ply_file_path"], label="ply_file_path")
        link_or_copy(source_ply, output / relative)
        sparse_relative = relative.as_posix()

    train_frames = []
    train_receipt = []
    for output_index, (distance, source_index, frame) in enumerate(selected):
        _, image = resolve_dataset_file(source, frame["file_path"], label="image path")
        relative = Path("images") / f"nearest_train_{output_index:05d}{image.suffix.lower()}"
        link_or_copy(image, output / relative)
        copied = dict(frame)
        copied["file_path"] = relative.as_posix()
        train_frames.append(copied)
        train_receipt.append(
            {
                "distance_to_eval": distance,
                "source_frame_index": source_index,
                "source_file_path": frame["file_path"],
            }
        )

    _, eval_image = resolve_dataset_file(source, eval_frame["file_path"], label="image path")
    eval_relative = Path("images") / f"nearest_eval_00000{eval_image.suffix.lower()}"
    link_or_copy(eval_image, output / eval_relative)
    copied_eval = dict(eval_frame)
    copied_eval["file_path"] = eval_relative.as_posix()

    receipt: dict[str, Any] = {
        "source_dataset": str(source),
        "source_transforms_sha256": sha256(transforms),
        "eval_source_frame_index": eval_index,
        "eval_source_file_path": eval_frame["file_path"],
        "eval_image_sha256": sha256(eval_image),
        "train_camera_count": len(train_frames),
        "train_selection": "euclidean_camera_center_distance",
        "allow_source_eval_train": bool(args.allow_source_eval_train),
        "train_source_frames": train_receipt,
        "eval_is_held_out": all(row["source_frame_index"] != eval_index for row in train_receipt),
        "masks": False,
    }
    if sparse_relative is not None:
        receipt["sparse_point_cloud"] = sparse_relative

    result = dict(payload)
    if args.preserve_normalization_context:
        for frame in frames:
            relative, image = resolve_dataset_file(source, frame["file_path"], label="image path")
            link_or_copy(image, output / relative)
            if frame.get("depth_file_path") is not None:
                depth_relative, depth = resolve_dataset_file(
                    source,
                    frame["depth_file_path"],
                    label="depth_file_path",
                )
                link_or_copy(depth, output / depth_relative)
        result["frames"] = [dict(frame) for frame in frames]
        result["train_filenames"] = [row["source_file_path"] for row in train_receipt]
        result["val_filenames"] = [eval_frame["file_path"]]
        result["test_filenames"] = [eval_frame["file_path"]]
    else:
        result["frames"] = [*train_frames, copied_eval]
        result.pop("train_filenames", None)
        result.pop("val_filenames", None)
        result.pop("test_filenames", None)
    receipt["preserve_normalization_context"] = bool(args.preserve_normalization_context)
    receipt["normalization_frame_count"] = len(result["frames"])
    result["nearest_camera_eval"] = receipt
    if sparse_relative is not None:
        result["ply_file_path"] = sparse_relative
    atomic_json(output / "transforms.json", result)
    atomic_json(output / "nearest_camera_eval_manifest.json", receipt)
    print(
        f"complete train={len(train_frames)} eval=1 eval_source={eval_frame['file_path']} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
