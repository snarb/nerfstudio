#!/usr/bin/env python3
"""Create a display-referred JPEG copy of one native-EXR Nerfstudio dataset.

The conversion uses one train-split exposure for every camera, preserving
multiview photometric relationships.  Paths, camera count, split names, image
dimensions, and scene identifiers are discovered from ``transforms.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from nerfstudio.data.utils.data_utils import load_exr_image
from nerfstudio.utils.hdr import calibrate_exr_paths
from nerfstudio.utils.lookcloser_dataset import resolve_nerfstudio_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-mode", default="filename")
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--middle-gray", type=float, default=0.18)
    parser.add_argument("--exposure-mode", choices=("global", "per-image", "fixed"), default="global")
    parser.add_argument("--fixed-exposure-gain", type=float, default=None,
                        help="One positive, content-independent multiplier for the entire temporal campaign; requires --exposure-mode fixed.")
    parser.add_argument("--exposure-percentile", type=float, default=70.0)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when the immutable conversion request and all reused image hashes match.",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(value <= 0.0031308, 12.92 * value, 1.055 * np.power(value, 1.0 / 2.4) - 0.055)


def tone_map(image: np.ndarray, gain: float) -> np.ndarray:
    linear = np.maximum(image[..., :3].astype(np.float32, copy=False), 0.0) * gain
    display_linear = linear / (1.0 + linear)
    return np.uint8(np.clip(linear_to_srgb(display_linear), 0.0, 1.0) * 255.0 + 0.5)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_completed_conversion(destination: Path, request_hash: str, expected_count: int) -> bool:
    manifest_path = destination / "conversion_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("images")
    if manifest.get("request_sha256") != request_hash or not isinstance(rows, list) or len(rows) != expected_count:
        return False
    for row in rows:
        output = Path(str(row["output"]))
        if not output.is_file() or sha256(output) != row.get("sha256"):
            return False
    return (destination / "transforms.json").is_file()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.input.resolve()
    destination = args.output.resolve()
    if not 0.0 < args.middle_gray < 1.0:
        raise ValueError("--middle-gray must be in (0, 1)")
    if not 1 <= args.quality <= 100:
        raise ValueError("--quality must be in [1, 100]")
    if not 0.0 < args.exposure_percentile < 100.0:
        raise ValueError("--exposure-percentile must be in (0, 100)")
    if source == destination:
        raise ValueError("Input and output datasets must differ")
    split = resolve_nerfstudio_dataset(
        source,
        eval_mode=args.eval_mode,
        eval_interval=args.eval_interval,
        require_exr=True,
    )
    if (args.exposure_mode == "fixed") != (args.fixed_exposure_gain is not None):
        raise ValueError("Fixed mode and --fixed-exposure-gain must be supplied together")
    if args.fixed_exposure_gain is not None and (
        not np.isfinite(args.fixed_exposure_gain) or args.fixed_exposure_gain <= 0
    ):
        raise ValueError("Fixed exposure gain must be finite and positive")
    calibration = None if args.exposure_mode == "fixed" else calibrate_exr_paths(split.train_images)
    global_gain = (float(args.fixed_exposure_gain) if calibration is None else
                   args.middle_gray / (calibration.log_mean_luminance * (1.0 - args.middle_gray)))
    source_transforms = source / "transforms.json"
    payload = json.loads(source_transforms.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("transforms.json contains no frames")
    request = {
        "schema_version": 1,
        "source": str(source),
        "source_transforms_sha256": sha256(source_transforms),
        "eval_mode": args.eval_mode,
        "eval_interval": args.eval_interval,
        "middle_gray": args.middle_gray,
        "exposure_mode": args.exposure_mode,
        "exposure_percentile": args.exposure_percentile,
        "quality": args.quality,
        "jpeg_subsampling": "4:4:4",
        "curve": "global_exposure_then_reinhard_then_srgb",
    }
    if args.exposure_mode == "fixed":
        request["fixed_exposure_gain"] = global_gain
    request_hash = canonical_sha256(request)
    request["request_sha256"] = request_hash
    request_path = destination / "conversion_request.json"
    output_existed = destination.exists()
    if output_existed and not args.resume:
        if any(destination.iterdir()):
            raise RuntimeError(f"Output directory must be empty or --resume must be supplied: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if args.resume and output_existed and any(destination.iterdir()):
        if not request_path.is_file():
            raise RuntimeError(f"Cannot safely resume without {request_path}")
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous != request:
            raise RuntimeError("Refusing to resume: conversion request changed")
        if validate_completed_conversion(destination, request_hash, len(frames)):
            print(f"complete status=reused output={destination}", flush=True)
            return 0
    else:
        atomic_json(request_path, request)

    state_path = destination / "conversion_state.json"
    completed: dict[str, dict[str, object]] = {}
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("request_sha256") != request_hash:
            raise RuntimeError("Refusing to resume: conversion state request hash changed")
        state_rows = state.get("images", [])
        if not isinstance(state_rows, list):
            raise RuntimeError("Invalid conversion_state.json images")
        completed = {str(row["frame_file_path"]): row for row in state_rows}

    rows: list[dict[str, object]] = []
    images_dir = destination / "images"
    images_dir.mkdir(exist_ok=True)
    for index, frame in enumerate(frames, 1):
        original_file_path = str(frame["file_path"])
        input_path = (source / frame["file_path"]).resolve()
        if input_path.suffix.lower() != ".exr":
            raise RuntimeError(f"Expected EXR input, got {input_path}")
        output_path = images_dir / f"{input_path.stem}.jpg"
        source_hash = sha256(input_path)
        prior = completed.get(original_file_path)
        if (
            prior is not None
            and prior.get("source_sha256") == source_hash
            and output_path.is_file()
            and prior.get("sha256") == sha256(output_path)
        ):
            frame["file_path"] = str(output_path.relative_to(destination))
            rows.append(prior)
            print(f"images={index}/{len(frames)} status=reused {output_path.name}", flush=True)
            continue
        image = load_exr_image(input_path)
        if args.exposure_mode == "per-image":
            sample = np.maximum(image[::8, ::8, :3], 0.0)
            luminance = 0.2126 * sample[..., 0] + 0.7152 * sample[..., 1] + 0.0722 * sample[..., 2]
            anchor = float(np.percentile(luminance, args.exposure_percentile))
            gain = args.middle_gray / (max(anchor, 1e-8) * (1.0 - args.middle_gray))
        else:
            anchor = None if calibration is None else calibration.log_mean_luminance
            gain = global_gain
        jpeg = tone_map(image, gain)
        temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        Image.fromarray(jpeg, "RGB").save(
            temporary,
            format="JPEG",
            quality=args.quality,
            subsampling=0,
            optimize=True,
        )
        os.replace(temporary, output_path)
        frame["file_path"] = str(output_path.relative_to(destination))
        row = {
                "frame_file_path": original_file_path,
                "input": str(input_path),
                "source_sha256": source_hash,
                "output": str(output_path),
                "width": int(jpeg.shape[1]),
                "height": int(jpeg.shape[0]),
                "sha256": sha256(output_path),
                "exposure_anchor": anchor,
                "exposure_gain": gain,
            }
        rows.append(row)
        completed[original_file_path] = row
        atomic_json(
            state_path,
            {
                "schema_version": 1,
                "request_sha256": request_hash,
                "images": [completed[key] for key in sorted(completed)],
            },
        )
        print(f"images={index}/{len(frames)} {output_path.name}", flush=True)

    ply_name = payload.get("ply_file_path")
    if ply_name:
        source_ply = (source / ply_name).resolve()
        destination_ply = destination / Path(ply_name).name
        shutil.copyfile(source_ply, destination_ply)
        payload["ply_file_path"] = destination_ply.name
    payload["source_exr_dataset"] = str(source)
    payload["jpeg_tone_map"] = {
        "curve": "global_exposure_then_reinhard_then_srgb",
        "exposure_mode": args.exposure_mode,
        "global_gain": global_gain,
        "exposure_percentile": args.exposure_percentile,
        "middle_gray": args.middle_gray,
        "jpeg_quality": args.quality,
        "jpeg_subsampling": "4:4:4",
        "calibration": None if calibration is None else calibration.as_metadata(),
    }
    atomic_json(destination / "transforms.json", payload)
    atomic_json(
        destination / "conversion_manifest.json",
        {
            "schema_version": 1,
            "request_sha256": request_hash,
            "source": str(source),
            "destination": str(destination),
            "image_count": len(rows),
            "train_count": len(split.train_images),
            "eval_count": len(split.eval_images),
            "tone_map": payload["jpeg_tone_map"],
            "images": rows,
        },
    )
    state_path.unlink(missing_ok=True)
    print(
        f"complete images={len(rows)} train={len(split.train_images)} eval={len(split.eval_images)} "
        f"exposure_mode={args.exposure_mode} global_gain={global_gain:.9g} output={destination}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
