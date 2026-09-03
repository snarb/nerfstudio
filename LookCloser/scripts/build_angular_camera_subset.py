#!/usr/bin/env python3
"""Build an immutable, geometry-selected Nerfstudio train-camera subset.

Training cameras are selected by greedy farthest-point sampling on camera-center
directions around the least-squares optical-axis focus.  This approximates
uniform angular coverage without trusting filename order or looking at image
pixels.  All source frames remain as normalization context, while explicit
filename lists restrict supervision to the selected train cameras and the
source filename-eval split.
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
    materialized = source.resolve(strict=True) if source.is_symlink() else source
    try:
        os.link(materialized, destination)
    except OSError:
        shutil.copyfile(materialized, destination)


def clone_tree(source: Path, destination: Path) -> None:
    destination.mkdir()
    for root, directories, files in os.walk(source):
        relative = Path(root).relative_to(source)
        target = destination / relative
        target.mkdir(parents=True, exist_ok=True)
        directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
        for name in files:
            if relative == Path(".") and name in {"transforms.json", "angular_subset_manifest.json"}:
                continue
            link_or_copy(Path(root) / name, target / name)


def filename_split(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("transforms.json has no frames")
    if any(frame.get("mask_path") for frame in frames):
        raise ValueError("Angular camera subsets forbid person/image masks")
    explicit_train = set(payload.get("train_filenames", []))
    explicit_eval = set(payload.get("val_filenames", [])) or set(payload.get("test_filenames", []))
    if explicit_train or explicit_eval:
        train = [frame for frame in frames if frame.get("file_path") in explicit_train]
        evaluation = [frame for frame in frames if frame.get("file_path") in explicit_eval]
    else:
        train = [frame for frame in frames if "train" in Path(str(frame.get("file_path", ""))).stem.lower()]
        evaluation = [frame for frame in frames if "eval" in Path(str(frame.get("file_path", ""))).stem.lower()]
    if not train or not evaluation:
        raise ValueError(f"Filename split is empty: train={len(train)} eval={len(evaluation)}")
    return train, evaluation


def camera_geometry(frames: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = np.asarray([frame["transform_matrix"] for frame in frames], dtype=np.float64)
    if matrices.shape != (len(frames), 4, 4) or not np.isfinite(matrices).all():
        raise ValueError("Every selected frame must have a finite 4x4 transform_matrix")
    centers = matrices[:, :3, 3]
    forward = -matrices[:, :3, 2]
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    lhs = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for center, direction in zip(centers, forward):
        projector = np.eye(3) - np.outer(direction, direction)
        lhs += projector
        rhs += projector @ center
    focus = np.linalg.solve(lhs, rhs)
    directions = centers - focus
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms <= 1e-9):
        raise ValueError("Camera center coincides with the estimated focus")
    return centers, directions / norms, focus


def select_angular_frames(
    frames: list[dict[str, Any]], count: int, required_camera: str | None
) -> tuple[list[int], dict[str, Any]]:
    if not 1 <= count <= len(frames):
        raise ValueError(f"train count must be within [1, {len(frames)}]")
    _, directions, focus = camera_geometry(frames)
    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    if required_camera is not None:
        required = [index for index, frame in enumerate(frames) if frame.get("physical_camera") == required_camera]
        if len(required) != 1:
            raise ValueError(f"Expected one train frame for required camera {required_camera!r}, got {len(required)}")
        selected = required
    else:
        # Deterministic seed furthest from the mean viewing direction.
        mean_direction = directions.mean(axis=0)
        mean_direction /= np.linalg.norm(mean_direction)
        selected = [int(np.argmin(directions @ mean_direction))]
    while len(selected) < count:
        minimum_angle = angles[:, selected].min(axis=1)
        minimum_angle[selected] = -1.0
        best_value = float(minimum_angle.max())
        candidates = np.flatnonzero(np.isclose(minimum_angle, best_value, atol=1e-10)).tolist()
        selected.append(min(candidates, key=lambda index: str(frames[index]["file_path"])))
    selected = sorted(selected, key=lambda index: str(frames[index]["file_path"]))
    selected_angles = angles[np.ix_(selected, selected)].copy()
    np.fill_diagonal(selected_angles, np.inf)
    nearest = selected_angles.min(axis=1) if len(selected) > 1 else np.asarray([180.0])
    stats = {
        "focus": focus.tolist(),
        "nearest_selected_angle_deg": {
            "min": float(nearest.min()),
            "median": float(np.median(nearest)),
            "max": float(nearest.max()),
        },
        "maximum_selected_pair_angle_deg": (
            float(angles[np.ix_(selected, selected)].max()) if len(selected) > 1 else 0.0
        ),
    }
    return selected, stats


def select_nearest_frames(
    frames: list[dict[str, Any]], eval_frame: dict[str, Any], count: int, required_camera: str | None
) -> tuple[list[int], dict[str, Any]]:
    if not 1 <= count <= len(frames):
        raise ValueError(f"train count must be within [1, {len(frames)}]")
    centers, _, focus = camera_geometry(frames)
    eval_matrix = np.asarray(eval_frame["transform_matrix"], dtype=np.float64)
    if eval_matrix.shape != (4, 4) or not np.isfinite(eval_matrix).all():
        raise ValueError("Eval frame must have a finite 4x4 transform_matrix")
    distances = np.linalg.norm(centers - eval_matrix[:3, 3], axis=1)
    selected = sorted(range(len(frames)), key=lambda index: (float(distances[index]), str(frames[index]["file_path"])))[
        :count
    ]
    if required_camera is not None and not any(
        frames[index].get("physical_camera") == required_camera for index in selected
    ):
        raise ValueError(f"Required camera {required_camera!r} is not inside the nearest-{count} selection")
    selected = sorted(selected, key=lambda index: str(frames[index]["file_path"]))
    return selected, {
        "focus": focus.tolist(),
        "distance_to_eval": {
            "min": float(distances[selected].min()),
            "median": float(np.median(distances[selected])),
            "max": float(distances[selected].max()),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--required-camera")
    parser.add_argument("--strategy", choices=("angular", "nearest-eval"), default="angular")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    transforms = source / "transforms.json"
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    train_frames, eval_frames = filename_split(payload)
    if args.strategy == "angular":
        indices, geometry = select_angular_frames(train_frames, args.train_count, args.required_camera)
        strategy = "greedy_farthest_point_on_camera_center_directions"
    else:
        if len(eval_frames) != 1:
            raise ValueError(f"nearest-eval strategy requires exactly one eval frame, got {len(eval_frames)}")
        indices, geometry = select_nearest_frames(
            train_frames, eval_frames[0], args.train_count, args.required_camera
        )
        strategy = "euclidean_camera_center_distance_to_eval"
    selected = [train_frames[index] for index in indices]
    result = json.loads(json.dumps(payload))
    result["train_filenames"] = [frame["file_path"] for frame in selected]
    result["val_filenames"] = [frame["file_path"] for frame in eval_frames]
    result["test_filenames"] = [frame["file_path"] for frame in eval_frames]
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "source_transforms_sha256": sha256(transforms),
        "selection_uses_image_pixels": False,
        "strategy": strategy,
        "required_physical_camera": args.required_camera,
        "source_train_count": len(train_frames),
        "selected_train_count": len(selected),
        "eval_count": len(eval_frames),
        "selected_train_frames": [
            {"file_path": frame["file_path"], "physical_camera": frame.get("physical_camera")} for frame in selected
        ],
        **geometry,
    }
    result["angular_camera_subset"] = manifest
    stage = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        clone_tree(source, stage)
        atomic_json(stage / "transforms.json", result)
        manifest["output"] = str(output)
        manifest["output_transforms_sha256"] = sha256(stage / "transforms.json")
        atomic_json(stage / "angular_subset_manifest.json", manifest)
        os.replace(stage, output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    print(
        f"complete train={len(selected)} eval={len(eval_frames)} "
        f"strategy={args.strategy} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
