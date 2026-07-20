#!/usr/bin/env python3
"""Reproduce the accepted 007740 EXR-to-JPEG pipeline for temporal frames.

The script is deliberately split into two commands:

* ``stage`` reads EXRs and writes converted JPEGs only below red-to-exr/temp.
* ``apply`` verifies a byte-exact 007740 proof, backs up existing JPEGs, and
  atomically replaces images only for frames strictly newer than 007740.

The protected 007740 dataset is never an apply target.

Run this script with ``.venv-leader-jpeg/bin/python``.  The exact package
versions are pinned in ``requirements-leader-jpeg.txt`` and enforced below.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import OpenEXR
import PIL
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_ROOT = SCRIPT_DIR / "temp"
DEFAULT_EXR_ROOT = Path("/fsx/oregon/tank_bkup/6A_4_EXR")
DEFAULT_DATASET_ROOT = Path(
    "/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/temporal_perframe_stride7_45f"
)
DEFAULT_FFMPEG = Path("/home/ubuntu/anaconda3/envs/ngp/bin/ffmpeg")
REFERENCE_FRAME = 7740
EVAL_STEMS = ("D004_A014", "E004_B014", "I004_D014")
EXPECTED_CAMERA_COUNT = 69
EXPECTED_TRAIN_COUNT = 66
EXPECTED_EVAL_COUNT = 3
QHD_SIZE = (2560, 1440)
HD_SIZE = (1920, 1080)
PIPELINE_SCHEMA = 1
INTERMEDIATE_QUANTIZATION_SHA256 = (
    "5e945aa24d55aba9b3560867e60e283e0700efe27e83afbd6c9d6389eaf3486e"
)
FINAL_QUANTIZATION_SHA256 = (
    "a412dffd7346a1fb47fd63bd5563df629b103fea55100fa4fc616c03ed6e4d15"
)
INTERMEDIATE_LAYER = [[1, 2, 1, 0], [2, 1, 1, 1], [3, 1, 1, 1]]
FINAL_LAYER = [[1, 2, 2, 0], [2, 1, 2, 0], [3, 1, 2, 0]]

EXPECTED_VERSIONS = {
    "python_major_minor": "3.12",
    "numpy": "2.4.6",
    "Pillow": "12.2.0",
    "OpenEXR": "3.4.11",
    "ffmpeg": "5.0.1",
    "libavcodec": "59. 18.100",
}

# Frozen accepted 007740 color recipe.
GRADE = True
CHUNK_ROWS = 256
EXPOSURE_TARGET = 0.44
EXPOSURE_MAX_GAIN = 64.0
SHADOW_TINT = np.array([-0.035, +0.010, +0.045], dtype=np.float32)
HIGHLIGHT_TINT = np.array([+0.055, +0.020, -0.040], dtype=np.float32)
CONTRAST_STRENGTH = 4.0
SATURATION = 0.96
BLACK_LIFT = 0.055
DISPLAY_BRIGHTNESS = 1.10
VIGNETTE_STRENGTH = 0.16
VIGNETTE_INNER = 0.55
VIGNETTE_POWER = 1.6

ACES_IN = np.array(
    [
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777],
    ],
    dtype=np.float32,
)
ACES_OUT = np.array(
    [
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602],
    ],
    dtype=np.float32,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_below(path: Path, parent: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = parent.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must be below {root}: {resolved}") from exc
    if resolved == root:
        raise RuntimeError(f"{label} must be a child of {root}, not the root itself")
    return resolved


def frame_name(frame: int) -> str:
    if not (0 <= frame <= 999999):
        raise ValueError(f"Invalid frame number: {frame}")
    return f"{frame:06d}"


def selected_frames(args: argparse.Namespace) -> list[int]:
    explicit = list(args.frame or [])
    has_range = args.start_frame is not None or args.end_frame is not None
    if explicit and has_range:
        raise RuntimeError("Use either --frame or --start-frame/--end-frame, not both")
    if explicit:
        frames = sorted(set(explicit))
    else:
        if args.start_frame is None or args.end_frame is None:
            raise RuntimeError("Provide --frame or both --start-frame and --end-frame")
        if args.stride <= 0 or args.end_frame < args.start_frame:
            raise RuntimeError("Invalid frame range or stride")
        frames = list(range(args.start_frame, args.end_frame + 1, args.stride))
        if not frames or frames[-1] != args.end_frame:
            raise RuntimeError("Frame range end must lie exactly on the requested stride")
    for frame in frames:
        frame_name(frame)
    return frames


def rrt_odt_fit(value: np.ndarray) -> np.ndarray:
    a = value * (value + 0.0245786) - 0.000090537
    b = value * (0.983729 * value + 0.4329510) + 0.238081
    return a / b


def aces_filmic(rgb_linear: np.ndarray) -> np.ndarray:
    flat = rgb_linear.reshape(-1, 3) @ ACES_IN.T
    flat = rrt_odt_fit(flat)
    flat = flat @ ACES_OUT.T
    return np.clip(flat.reshape(rgb_linear.shape), 0.0, 1.0)


def s_curve(value: np.ndarray, strength: float = CONTRAST_STRENGTH) -> np.ndarray:
    result = 1.0 / (1.0 + np.exp(-strength * (value - 0.5)))
    y0 = 1.0 / (1.0 + np.exp(strength * 0.5))
    y1 = 1.0 / (1.0 + np.exp(-strength * 0.5))
    return (result - y0) / (y1 - y0)


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    a = 0.055
    return np.where(
        value <= 0.0031308,
        12.92 * value,
        (1 + a) * np.power(value, 1 / 2.4) - a,
    )


def auto_exposure(image: np.ndarray) -> float:
    downsampled = image[::8, ::8, :].astype(np.float32)
    np.maximum(downsampled, 0.0, out=downsampled)
    luma = (
        0.2126 * downsampled[..., 0]
        + 0.7152 * downsampled[..., 1]
        + 0.0722 * downsampled[..., 2]
    )
    anchor = float(np.percentile(luma, 70))
    return float(np.clip(EXPOSURE_TARGET / max(anchor, 1e-6), 1.0, EXPOSURE_MAX_GAIN))


def grade_chunk(
    chunk: np.ndarray,
    exposure_gain: float,
    center_x: float,
    center_y: float,
    half_width: float,
    half_height: float,
    y0: int,
    y1: int,
    width: int,
) -> np.ndarray:
    chunk = chunk.astype(np.float32)
    np.maximum(chunk, 0.0, out=chunk)
    chunk *= exposure_gain
    chunk = aces_filmic(chunk)

    if GRADE:
        luma = 0.2126 * chunk[..., 0] + 0.7152 * chunk[..., 1] + 0.0722 * chunk[..., 2]
        shadow_mask = np.clip(1.0 - luma * 1.6, 0.0, 1.0) ** 1.2
        highlight_mask = np.clip((luma - 0.35) / 0.6, 0.0, 1.0) ** 1.1
        chunk += shadow_mask[..., None] * SHADOW_TINT
        chunk += highlight_mask[..., None] * HIGHLIGHT_TINT
        np.clip(chunk, 0.0, 1.0, out=chunk)

        chunk = s_curve(chunk)
        gray = 0.2126 * chunk[..., 0] + 0.7152 * chunk[..., 1] + 0.0722 * chunk[..., 2]
        chunk = gray[..., None] + (chunk - gray[..., None]) * SATURATION
        chunk = chunk * (1.0 - BLACK_LIFT) + BLACK_LIFT

        yy = np.arange(y0, y1, dtype=np.float32)[:, None]
        xx = np.arange(width, dtype=np.float32)[None, :]
        radius = np.sqrt(
            ((xx - center_x) / half_width) ** 2 + ((yy - center_y) / half_height) ** 2
        )
        vignette = 1.0 - VIGNETTE_STRENGTH * np.clip(
            (radius - VIGNETTE_INNER) / (1.0 - VIGNETTE_INNER), 0.0, 1.0
        ) ** VIGNETTE_POWER
        chunk *= vignette[..., None]
        np.clip(chunk, 0.0, 1.0, out=chunk)
    return chunk


def read_exr_rgb(path: Path) -> np.ndarray:
    exr = OpenEXR.File(str(path))
    channels = exr.channels()
    if "RGB" in channels:
        image = channels["RGB"].pixels
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"Unexpected RGB layout in {path}: {image.shape}")
        return image
    if "RGBA" in channels:
        image = channels["RGBA"].pixels
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise RuntimeError(f"Unexpected RGBA layout in {path}: {image.shape}")
        return image[..., :3]
    keys = {key.upper(): key for key in channels}
    try:
        return np.stack(
            [channels[keys[name]].pixels for name in ("R", "G", "B")], axis=-1
        )
    except KeyError as exc:
        raise RuntimeError(f"No RGB channels in {path}; found {sorted(channels)}") from exc


def center_crop_box(width: int, height: int, target_size: tuple[int, int]) -> tuple[int, int, int, int]:
    target_width, target_height = target_size
    target_aspect = target_width / target_height
    aspect = width / height
    if aspect > target_aspect:
        crop_width = int(round(height * target_aspect))
        crop_height = height
        left = (width - crop_width) // 2
        top = 0
    else:
        crop_width = width
        crop_height = int(round(width / target_aspect))
        left = 0
        top = (height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


def render_intermediate_jpeg(source: Path, destination: Path) -> dict[str, Any]:
    started = time.monotonic()
    image = read_exr_rgb(source)
    height, width, channels = image.shape
    if channels != 3:
        raise RuntimeError(f"Expected three channels in {source}, got {image.shape}")
    exposure_gain = auto_exposure(image)
    output = np.empty((height, width, 3), dtype=np.uint8)
    center_x, center_y = width / 2.0, height / 2.0
    for y0 in range(0, height, CHUNK_ROWS):
        y1 = min(y0 + CHUNK_ROWS, height)
        graded = grade_chunk(
            image[y0:y1],
            exposure_gain,
            center_x,
            center_y,
            width / 2.0,
            height / 2.0,
            y0,
            y1,
            width,
        )
        srgb = linear_to_srgb(np.clip(graded * DISPLAY_BRIGHTNESS, 0.0, 1.0))
        output[y0:y1] = (srgb * 255.0 + 0.5).astype(np.uint8)
    del image

    qhd = Image.fromarray(output, "RGB")
    qhd = qhd.crop(center_crop_box(width, height, QHD_SIZE))
    qhd = qhd.resize(QHD_SIZE, Image.Resampling.LANCZOS)
    hd = qhd.resize(HD_SIZE, Image.Resampling.LANCZOS)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    hd.save(
        temporary,
        format="JPEG",
        quality=95,
        subsampling=1,
        progressive=True,
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "intermediate": str(destination),
        "intermediate_sha256": sha256_file(destination),
        "width": width,
        "height": height,
        "exposure_gain": exposure_gain,
        "exposure_ev": math.log2(exposure_gain),
        "seconds": time.monotonic() - started,
    }


def jpeg_profile(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        quantization = tuple(tuple(table) for _, table in sorted((image.quantization or {}).items()))
        return {
            "size": list(image.size),
            "mode": image.mode,
            "layer": [list(row) for row in getattr(image, "layer", ())],
            "quantization_sha256": hashlib.sha256(repr(quantization).encode()).hexdigest(),
            "quantization_table_count": len(quantization),
            "comment": (image.info.get("comment") or b"").rstrip(b"\x00").decode(
                "utf-8", errors="replace"
            ),
        }


def run_ffmpeg_batch(ffmpeg: Path, source_pattern: Path, output_pattern: Path) -> None:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-noautorotate",
        "-i",
        str(source_pattern),
        "-filter_complex",
        "[0:v]split=1[t0];[t0]scale=iw/1:ih/1[out0]",
        "-map",
        "[out0]",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)


def runtime_fingerprint(ffmpeg: Path, allow_mismatch: bool) -> dict[str, Any]:
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    result = subprocess.run(
        [str(ffmpeg), "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    version_text = result.stdout
    actual = {
        "python": sys.version,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
        "Pillow": PIL.__version__,
        "OpenEXR": str(OpenEXR.__version__).removeprefix("b'").removesuffix("'"),
        "ffmpeg": version_text.splitlines()[0],
        "libavcodec": next(
            (line.strip() for line in version_text.splitlines() if line.strip().startswith("libavcodec")),
            "",
        ),
    }
    mismatches = []
    for key in ("python_major_minor", "numpy", "Pillow", "OpenEXR"):
        if actual[key] != EXPECTED_VERSIONS[key]:
            mismatches.append(f"{key}: {actual[key]} != {EXPECTED_VERSIONS[key]}")
    if f"ffmpeg version {EXPECTED_VERSIONS['ffmpeg']}" not in actual["ffmpeg"]:
        mismatches.append(f"ffmpeg: {actual['ffmpeg']}")
    if EXPECTED_VERSIONS["libavcodec"] not in actual["libavcodec"]:
        mismatches.append(f"libavcodec: {actual['libavcodec']}")
    if mismatches and not allow_mismatch:
        raise RuntimeError("Runtime fingerprint mismatch: " + "; ".join(mismatches))
    return {
        "actual": actual,
        "expected": EXPECTED_VERSIONS,
        "mismatches": mismatches,
        "allow_version_mismatch": allow_mismatch,
        "ffmpeg_path": str(ffmpeg.resolve()),
    }


def target_mapping(stems: Iterable[str]) -> dict[str, str]:
    ordered = sorted(stems)
    if len(ordered) != EXPECTED_CAMERA_COUNT or len(set(ordered)) != EXPECTED_CAMERA_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_CAMERA_COUNT} unique cameras, got {len(set(ordered))}")
    missing_eval = sorted(set(EVAL_STEMS) - set(ordered))
    if missing_eval:
        raise RuntimeError(f"Missing canonical eval cameras: {missing_eval}")
    train = [stem for stem in ordered if stem not in EVAL_STEMS]
    if len(train) != EXPECTED_TRAIN_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_TRAIN_COUNT} train cameras, got {len(train)}")
    mapping = {
        stem: f"frame_eval_{index:05d}.jpg" for index, stem in enumerate(EVAL_STEMS, 1)
    }
    mapping.update(
        {stem: f"frame_train_{index:05d}.jpg" for index, stem in enumerate(train, 1)}
    )
    return mapping


def expected_target_names(mapping: dict[str, str]) -> set[str]:
    return set(mapping.values())


def validate_dataset_frame(dataset_root: Path, frame: int, expected_names: set[str]) -> None:
    directory = dataset_root / frame_name(frame)
    transforms = directory / "transforms.json"
    images = directory / "images"
    if not transforms.is_file() or not images.is_dir():
        raise FileNotFoundError(f"Incomplete dataset frame: {directory}")
    data = json.loads(transforms.read_text(encoding="utf-8"))
    transform_names = {Path(row["file_path"]).name for row in data.get("frames", [])}
    disk_names = {path.name for path in images.glob("*.jpg")}
    if transform_names != expected_names or disk_names != expected_names:
        raise RuntimeError(
            f"Dataset binding mismatch for {frame_name(frame)}: "
            f"transforms={len(transform_names)} images={len(disk_names)} expected={len(expected_names)}"
        )


def convert_task(task: tuple[str, str]) -> dict[str, Any]:
    return render_intermediate_jpeg(Path(task[0]), Path(task[1]))


def build_final_frame(
    ffmpeg: Path,
    intermediate_dir: Path,
    final_dir: Path,
    expected_names: set[str],
) -> None:
    build_dir = final_dir.parent / f"images.build-{os.getpid()}"
    if build_dir.exists():
        raise RuntimeError(f"Stale build directory exists: {build_dir}")
    build_dir.mkdir(parents=True)
    try:
        run_ffmpeg_batch(
            ffmpeg,
            intermediate_dir / "frame_eval_%05d.jpg",
            build_dir / "frame_eval_%05d.jpg",
        )
        run_ffmpeg_batch(
            ffmpeg,
            intermediate_dir / "frame_train_%05d.jpg",
            build_dir / "frame_train_%05d.jpg",
        )
        built_names = {path.name for path in build_dir.glob("*.jpg")}
        if built_names != expected_names:
            raise RuntimeError(f"FFmpeg output mismatch: {len(built_names)} != {len(expected_names)}")
        final_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(expected_names):
            os.replace(build_dir / name, final_dir / name)
    finally:
        if build_dir.exists() and not any(build_dir.iterdir()):
            build_dir.rmdir()


def stage(args: argparse.Namespace) -> int:
    frames = selected_frames(args)
    staging_dir = require_below(args.staging_dir, TEMP_ROOT, "--staging-dir")
    exr_root = args.exr_root.resolve()
    dataset_root = args.dataset_root.resolve()
    runtime = runtime_fingerprint(args.ffmpeg, args.allow_version_mismatch)
    reference_stems = sorted(path.stem for path in (exr_root / frame_name(REFERENCE_FRAME)).glob("*.exr"))
    mapping = target_mapping(reference_stems)
    names = expected_target_names(mapping)
    script_sha256 = sha256_file(Path(__file__))
    staging_dir.mkdir(parents=True, exist_ok=True)

    frame_records = []
    for frame in frames:
        source_dir = exr_root / frame_name(frame)
        sources = sorted(source_dir.glob("*.exr"))
        source_stems = sorted(path.stem for path in sources)
        if source_stems != reference_stems:
            raise RuntimeError(f"Camera set mismatch in {source_dir}")
        validate_dataset_frame(dataset_root, frame, names)
        frame_dir = staging_dir / frame_name(frame)
        intermediate_dir = frame_dir / "intermediate_hd_q95_422"
        final_dir = frame_dir / "images"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        tasks = [(str(path), str(intermediate_dir / mapping[path.stem])) for path in sources]
        print(f"stage frame={frame_name(frame)} cameras={len(tasks)} workers={args.workers}", flush=True)
        started = time.monotonic()
        records = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(convert_task, task): task for task in tasks}
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                record = future.result()
                records.append(record)
                print(
                    f"  frame={frame_name(frame)} done={completed}/{len(tasks)} "
                    f"camera={Path(record['source']).stem} ev={record['exposure_ev']:+.3f} "
                    f"seconds={record['seconds']:.2f}",
                    flush=True,
                )
        intermediate_names = {path.name for path in intermediate_dir.glob("*.jpg")}
        if intermediate_names != names:
            raise RuntimeError(f"Intermediate output mismatch for frame {frame_name(frame)}")
        intermediate_profiles = {json.dumps(jpeg_profile(path), sort_keys=True) for path in intermediate_dir.glob("*.jpg")}
        if len(intermediate_profiles) != 1:
            raise RuntimeError(f"Mixed intermediate JPEG profiles in frame {frame_name(frame)}")
        intermediate_profile = json.loads(next(iter(intermediate_profiles)))
        expected_intermediate_profile = {
            "size": list(HD_SIZE),
            "mode": "RGB",
            "layer": INTERMEDIATE_LAYER,
            "quantization_sha256": INTERMEDIATE_QUANTIZATION_SHA256,
            "quantization_table_count": 2,
            "comment": "",
        }
        if intermediate_profile != expected_intermediate_profile:
            raise RuntimeError(
                f"Unexpected intermediate JPEG profile in {frame_name(frame)}: "
                f"{intermediate_profile}"
            )

        build_final_frame(args.ffmpeg, intermediate_dir, final_dir, names)
        final_profiles = {json.dumps(jpeg_profile(path), sort_keys=True) for path in final_dir.glob("*.jpg")}
        if len(final_profiles) != 1:
            raise RuntimeError(f"Mixed final JPEG profiles in frame {frame_name(frame)}")
        final_profile = json.loads(next(iter(final_profiles)))
        expected_final_profile = {
            "size": list(HD_SIZE),
            "mode": "RGB",
            "layer": FINAL_LAYER,
            "quantization_sha256": FINAL_QUANTIZATION_SHA256,
            "quantization_table_count": 1,
            "comment": "Lavc59.18.100",
        }
        if final_profile != expected_final_profile:
            raise RuntimeError(
                f"Unexpected final JPEG profile in {frame_name(frame)}: {final_profile}"
            )

        outputs = []
        records_by_source = {Path(record["source"]).stem: record for record in records}
        for stem, target_name in sorted(mapping.items(), key=lambda item: item[1]):
            final_path = final_dir / target_name
            outputs.append(
                {
                    "camera": stem,
                    "target_name": target_name,
                    "source": records_by_source[stem]["source"],
                    "source_sha256": records_by_source[stem]["source_sha256"],
                    "intermediate_sha256": records_by_source[stem]["intermediate_sha256"],
                    "output": str(final_path),
                    "output_sha256": sha256_file(final_path),
                    "exposure_gain": records_by_source[stem]["exposure_gain"],
                }
            )
        reference = None
        if frame == REFERENCE_FRAME:
            reference_dir = dataset_root / frame_name(frame) / "images"
            mismatches = [
                row["target_name"]
                for row in outputs
                if row["output_sha256"] != sha256_file(reference_dir / row["target_name"])
            ]
            reference = {
                "directory": str(reference_dir),
                "byte_exact_count": len(outputs) - len(mismatches),
                "expected_count": EXPECTED_CAMERA_COUNT,
                "mismatches": mismatches,
            }
            if mismatches:
                raise RuntimeError(f"007740 byte reproduction failed: {mismatches[:10]}")
        frame_record = {
            "frame": frame,
            "frame_name": frame_name(frame),
            "source_dir": str(source_dir),
            "staged_dir": str(final_dir),
            "seconds": time.monotonic() - started,
            "intermediate_profile": intermediate_profile,
            "final_profile": final_profile,
            "outputs": outputs,
            "reference_verification": reference,
        }
        atomic_json(frame_dir / "frame_manifest.json", frame_record)
        frame_records.append(frame_record)
        print(
            f"staged frame={frame_name(frame)} outputs={len(outputs)} seconds={frame_record['seconds']:.1f}",
            flush=True,
        )

    manifest = {
        "schema_version": PIPELINE_SCHEMA,
        "created_at": utc_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": script_sha256,
        "command": "stage",
        "runtime": runtime,
        "exr_root": str(exr_root),
        "dataset_root": str(dataset_root),
        "reference_frame": REFERENCE_FRAME,
        "eval_stems": list(EVAL_STEMS),
        "frames": frame_records,
    }
    atomic_json(staging_dir / "conversion_manifest.json", manifest)
    print(f"manifest={staging_dir / 'conversion_manifest.json'}", flush=True)
    return 0


def load_and_verify_manifest(staging_dir: Path) -> dict[str, Any]:
    path = staging_dir / "conversion_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PIPELINE_SCHEMA:
        raise RuntimeError(f"Unsupported manifest schema: {path}")
    for frame in manifest.get("frames", []):
        for row in frame.get("outputs", []):
            output = Path(row["output"])
            if not output.is_file() or sha256_file(output) != row["output_sha256"]:
                raise RuntimeError(f"Staged output hash mismatch: {output}")
    return manifest


def validate_reference_proof(path: Path, script_sha256: str) -> dict[str, Any]:
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("script_sha256") != script_sha256:
        raise RuntimeError("Reference proof was produced by a different script revision")
    reference_rows = [
        frame for frame in proof.get("frames", []) if int(frame.get("frame", -1)) == REFERENCE_FRAME
    ]
    if len(reference_rows) != 1:
        raise RuntimeError("Reference proof must contain exactly one 007740 frame")
    verification = reference_rows[0].get("reference_verification") or {}
    if verification.get("byte_exact_count") != EXPECTED_CAMERA_COUNT or verification.get("mismatches"):
        raise RuntimeError("Reference proof is not 69/69 byte exact")
    return proof


def apply_staged(args: argparse.Namespace) -> int:
    staging_dir = require_below(args.staging_dir, TEMP_ROOT, "--staging-dir")
    backup_dir = require_below(args.backup_dir, TEMP_ROOT, "--backup-dir")
    manifest = load_and_verify_manifest(staging_dir)
    script_sha256 = sha256_file(Path(__file__))
    if manifest.get("script_sha256") != script_sha256:
        raise RuntimeError("Staged outputs were produced by a different script revision")
    validate_reference_proof(args.reference_proof.resolve(), script_sha256)
    frames = [int(row["frame"]) for row in manifest.get("frames", [])]
    if not frames or any(frame <= REFERENCE_FRAME for frame in frames):
        raise RuntimeError("Apply accepts only frames strictly newer than protected 007740")
    if not args.confirm_overwrite_after_007740:
        raise RuntimeError("Pass --confirm-overwrite-after-007740 after reviewing the plan")

    dataset_root = args.dataset_root.resolve()
    expected_names = {row["target_name"] for row in manifest["frames"][0]["outputs"]}
    for frame in frames:
        validate_dataset_frame(dataset_root, frame, expected_names)
    if backup_dir.exists():
        raise RuntimeError(f"Backup directory already exists: {backup_dir}")

    total = sum(len(frame["outputs"]) for frame in manifest["frames"])
    print(f"apply plan frames={len(frames)} images={total} backup={backup_dir}", flush=True)
    if not args.execute:
        print("dry-run only; pass --execute to create backup and replace files", flush=True)
        return 0

    backup_records = []
    for frame_record in manifest["frames"]:
        frame = int(frame_record["frame"])
        source_images = dataset_root / frame_name(frame) / "images"
        destination = backup_dir / frame_name(frame) / "images"
        destination.mkdir(parents=True, exist_ok=False)
        for row in frame_record["outputs"]:
            original = source_images / row["target_name"]
            backup = destination / row["target_name"]
            original_sha256 = sha256_file(original)
            shutil.copy2(original, backup)
            backup_sha256 = sha256_file(backup)
            if backup_sha256 != original_sha256:
                raise RuntimeError(f"Backup verification failed: {backup}")
            backup_records.append(
                {
                    "frame": frame,
                    "target": str(original),
                    "original_sha256": original_sha256,
                    "backup": str(backup),
                    "backup_sha256": backup_sha256,
                    "replacement_sha256": row["output_sha256"],
                }
            )
    backup_manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source_manifest": str(staging_dir / "conversion_manifest.json"),
        "source_manifest_sha256": sha256_file(staging_dir / "conversion_manifest.json"),
        "protected_reference_frame": REFERENCE_FRAME,
        "records": backup_records,
        "replacement_started": False,
    }
    atomic_json(backup_dir / "backup_manifest.json", backup_manifest)

    backup_manifest["replacement_started"] = True
    backup_manifest["replacement_started_at"] = utc_now()
    atomic_json(backup_dir / "backup_manifest.json", backup_manifest)
    for frame_record in manifest["frames"]:
        frame = int(frame_record["frame"])
        target_dir = dataset_root / frame_name(frame) / "images"
        for row in frame_record["outputs"]:
            staged = Path(row["output"])
            target = target_dir / row["target_name"]
            temporary = target.with_name(f".{target.name}.leader-jpeg-tmp-{os.getpid()}")
            shutil.copy2(staged, temporary)
            if sha256_file(temporary) != row["output_sha256"]:
                raise RuntimeError(f"Replacement copy verification failed: {temporary}")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)

    mismatches = []
    for frame_record in manifest["frames"]:
        frame = int(frame_record["frame"])
        target_dir = dataset_root / frame_name(frame) / "images"
        for row in frame_record["outputs"]:
            target = target_dir / row["target_name"]
            if sha256_file(target) != row["output_sha256"]:
                mismatches.append(str(target))
    if mismatches:
        raise RuntimeError(f"Post-apply verification failed: {mismatches[:10]}")
    backup_manifest["replacement_completed"] = True
    backup_manifest["replacement_completed_at"] = utc_now()
    backup_manifest["post_apply_mismatches"] = mismatches
    atomic_json(backup_dir / "backup_manifest.json", backup_manifest)
    print(f"applied frames={len(frames)} images={total} backup={backup_dir}", flush=True)
    return 0


def add_frame_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frame", type=int, action="append")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--stride", type=int, default=7)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage_parser = subparsers.add_parser("stage", help="Convert EXRs into a temp staging directory")
    add_frame_selection(stage_parser)
    stage_parser.add_argument("--staging-dir", type=Path, required=True)
    stage_parser.add_argument("--exr-root", type=Path, default=DEFAULT_EXR_ROOT)
    stage_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    stage_parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    stage_parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    stage_parser.add_argument("--allow-version-mismatch", action="store_true")

    apply_parser = subparsers.add_parser("apply", help="Back up and atomically install staged JPEGs")
    apply_parser.add_argument("--staging-dir", type=Path, required=True)
    apply_parser.add_argument("--reference-proof", type=Path, required=True)
    apply_parser.add_argument("--backup-dir", type=Path, required=True)
    apply_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    apply_parser.add_argument("--confirm-overwrite-after-007740", action="store_true")
    apply_parser.add_argument("--execute", action="store_true")

    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    if args.command == "stage":
        return stage(args)
    if args.command == "apply":
        return apply_staged(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
