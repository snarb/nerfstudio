#!/usr/bin/env python3
"""Shared fail-closed helpers for the temporal LookCloser campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from PIL import Image, ImageDraw


DATA_ROOT = Path("/home/brans/temporal_perframe_stride7_45f")
CAMPAIGN_ROOT = Path("/mnt/data/lookcloser_temporal_perframe_stride7_45f_v2")
METRICS_PATH = DATA_ROOT / "metrics.csv"
FRAME_NAMES = tuple(f"{value:06d}" for value in range(7_740, 8_049, 7))
TARGET_FRAMES = FRAME_NAMES[2:]
SEEDS = (42, 43, 44)

INTERVAL = 15_188
INITIAL_FINAL_STEP = 60_752
INITIAL_TARGET_STEP = 151_880
PROCESS_BOUNDARIES = tuple(range(INTERVAL, INITIAL_TARGET_STEP + 1, INTERVAL))
INITIAL_PROCESS_TARGETS = (
    INITIAL_FINAL_STEP,
    75_940,
    91_128,
    106_316,
    121_504,
    136_692,
    INITIAL_TARGET_STEP,
)

INITIAL_LR = 0.015
FINAL_LR = 0.0001
SCHEDULER_MAX_STEPS = 300_000
PSNR_TIE_DB = 0.07
PSNR_MIN = 29.7
SSIM_MIN = 0.668
LPIPS_MAX = 0.217
PREFERRED_PSNR = 29.88
PREFERRED_SSIM = 0.676
PREFERRED_LPIPS = 0.215
PLATEAU_PSNR_GAIN = 0.03
PLATEAU_SSIM_GAIN = 0.001
PLATEAU_LPIPS_IMPROVEMENT = 0.003
CROP_BOX = (700, 100, 1120, 480)

METRICS_COLUMNS = (
    "frame",
    "seed",
    "parent_frame",
    "selected_step",
    "psnr",
    "ssim",
    "lpips",
    "visual_gate",
    "checkpoint",
    "checkpoint_sha256",
)


class InfrastructureError(RuntimeError):
    """The campaign cannot safely interpret or reproduce an artifact."""


class QualityFailure(RuntimeError):
    """A complete frame failed a declared quality or visual gate."""


@dataclass(frozen=True)
class Boundary:
    seed: int
    local_step: int
    psnr: float
    ssim: float
    lpips: float
    checkpoint: Path
    checkpoint_sha256: str
    eval_json: Path
    render_dir: Path
    completed_wall_time_ns: int

    @property
    def numeric_pass(self) -> bool:
        return (
            math.isfinite(self.psnr)
            and math.isfinite(self.ssim)
            and math.isfinite(self.lpips)
            and self.psnr >= PSNR_MIN
            and self.ssim >= SSIM_MIN
            and self.lpips <= LPIPS_MAX
        )

    @property
    def preferred_pass(self) -> bool:
        return (
            self.numeric_pass
            and self.psnr >= PREFERRED_PSNR
            and self.ssim >= PREFERRED_SSIM
            and self.lpips <= PREFERRED_LPIPS
        )


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in METRICS_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("-")[-1])
    except ValueError as error:
        raise InfrastructureError(f"Cannot parse checkpoint step: {path}") from error


def frame_index(frame: str) -> int:
    if frame not in FRAME_NAMES:
        raise InfrastructureError(f"Frame is outside the canonical stride-7 chain: {frame}")
    return FRAME_NAMES.index(frame)


def previous_frame(frame: str) -> str:
    index = frame_index(frame)
    if index == 0:
        raise InfrastructureError("007740 has no temporal parent")
    return FRAME_NAMES[index - 1]


def _file_manifest(directory: Path, names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for name in sorted(names):
        path = directory / name
        if not path.is_file():
            raise InfrastructureError(f"Manifest source is missing: {path}")
        result[name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def compute_dataset_manifest(frame: str, dataset: Path) -> Dict[str, Any]:
    if dataset.resolve() != (DATA_ROOT / frame).resolve():
        raise InfrastructureError(f"Target dataset is not canonical for {frame}: {dataset}")
    images = dataset / "images"
    maps = dataset / "lookcloser_frequencies"
    if not images.is_dir() or not maps.is_dir():
        raise InfrastructureError(f"Dataset lacks canonical images/maps: {dataset}")
    forbidden = (
        dataset / "lookcloser_frequencies_chroma422",
        dataset / "lookcloser_frequencies_probe",
    )
    if maps.name != "lookcloser_frequencies" or any(
        "_probe" in part for part in maps.resolve().parts
    ):
        raise InfrastructureError(f"Nonstandard frequency-map path: {maps}")
    if any(maps.resolve() == path.resolve() for path in forbidden if path.exists()):
        raise InfrastructureError(f"Forbidden frequency-map path: {maps}")

    jpeg_names = sorted(path.name for path in images.iterdir() if path.is_file())
    pt_names = sorted(path.name for path in maps.glob("*.pt") if path.is_file())
    json_names = sorted(path.name for path in maps.glob("*.json") if path.is_file())
    other_map_names = sorted(
        path.name
        for path in maps.iterdir()
        if path.is_file() and path.suffix not in {".pt", ".json"}
    )
    if len(jpeg_names) != 69 or any(Path(name).suffix.lower() not in {".jpg", ".jpeg"} for name in jpeg_names):
        raise InfrastructureError(f"{frame} expected exactly 69 JPEGs, got {len(jpeg_names)}")
    if len(pt_names) != 66 or len(json_names) != 66 or other_map_names:
        raise InfrastructureError(
            f"{frame} expected 66 PT+JSON map pairs, got "
            f"{len(pt_names)} PT, {len(json_names)} JSON, extras={other_map_names}"
        )
    pt_stems = {Path(name).stem for name in pt_names}
    json_stems = {Path(name).stem for name in json_names}
    expected_train_stems = {f"frame_train_{index:05d}" for index in range(1, 67)}
    if pt_stems != expected_train_stems or json_stems != expected_train_stems:
        raise InfrastructureError(f"{frame} standard-map stems do not match the 66 train views")

    transforms = dataset / "transforms.json"
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    transform_names = [Path(str(row["file_path"])).name for row in payload.get("frames", [])]
    train = [name for name in transform_names if "_train_" in name]
    evaluate = [name for name in transform_names if "_eval_" in name]
    if (len(train), len(evaluate)) != (66, 3):
        raise InfrastructureError(
            f"{frame} transforms split is {len(train)}+{len(evaluate)}, expected 66+3"
        )
    if set(transform_names) != set(jpeg_names):
        raise InfrastructureError(f"{frame} transforms and JPEG file sets differ")

    return {
        "schema_version": 1,
        "frame": frame,
        "dataset": str(dataset.resolve()),
        "transforms": {
            "path": "transforms.json",
            "size_bytes": transforms.stat().st_size,
            "sha256": sha256_file(transforms),
            "train_images": 66,
            "eval_images": 3,
        },
        "jpeg": {"directory": "images", "files": _file_manifest(images, jpeg_names)},
        "frequency_maps": {
            "directory": "lookcloser_frequencies",
            "files": _file_manifest(maps, [*pt_names, *json_names]),
        },
    }


def freeze_dataset_manifest(frame: str, dataset: Path, path: Path) -> Dict[str, Any]:
    current = compute_dataset_manifest(frame, dataset)
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != current:
            raise InfrastructureError(f"Frozen dataset manifest changed: {path}")
    else:
        atomic_json(path, current)
    return current


def snapshot_checkpoint(snapshot: Path) -> Path:
    expected_root = snapshot / "lookcloser" / "final" / "nerfstudio_models"
    checkpoints = sorted(expected_root.glob("step-*.ckpt"))
    if len(checkpoints) != 1:
        raise InfrastructureError(
            f"Snapshot must contain exactly one checkpoint, found {len(checkpoints)}: {snapshot}"
        )
    return checkpoints[0]


def validate_snapshot_files(snapshot: Path, *, expected_frame: Optional[str] = None) -> Dict[str, Any]:
    required = ("config.yml", "selection.json", "provenance.json", "validation.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise InfrastructureError(f"Snapshot is incomplete ({missing}): {snapshot}")
    checkpoint = snapshot_checkpoint(snapshot)
    selection = json.loads((snapshot / "selection.json").read_text(encoding="utf-8"))
    provenance = json.loads((snapshot / "provenance.json").read_text(encoding="utf-8"))
    validation = json.loads((snapshot / "validation.json").read_text(encoding="utf-8"))
    frame = str(selection.get("frame"))
    if expected_frame is not None and frame != expected_frame:
        raise InfrastructureError(
            f"Snapshot frame mismatch: expected {expected_frame}, found {frame}"
        )
    checkpoint_hash = sha256_file(checkpoint)
    recorded = provenance.get("checkpoint_sha256")
    if recorded is not None and recorded != checkpoint_hash:
        raise InfrastructureError(f"Snapshot checkpoint hash mismatch: {snapshot}")
    selected_step = int(selection.get("selected_step", -1))
    if selected_step != checkpoint_step(checkpoint):
        raise InfrastructureError(f"Snapshot selected step does not match checkpoint: {snapshot}")
    metrics = validation.get("results", {})
    for name in ("psnr", "ssim", "lpips"):
        if name not in metrics or not math.isfinite(float(metrics[name])):
            raise InfrastructureError(f"Snapshot validation lacks finite {name}: {snapshot}")
    return {
        "frame": frame,
        "snapshot": str(snapshot.resolve()),
        "config": str((snapshot / "config.yml").resolve()),
        "config_sha256": sha256_file(snapshot / "config.yml"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": checkpoint_step(checkpoint),
        "selection_sha256": sha256_file(snapshot / "selection.json"),
        "provenance_sha256": sha256_file(snapshot / "provenance.json"),
        "validation_sha256": sha256_file(snapshot / "validation.json"),
        "metrics": {name: float(metrics[name]) for name in ("psnr", "ssim", "lpips")},
    }


def discover_boundaries(seed: int, run_dir: Path) -> list[Boundary]:
    boundaries: list[Boundary] = []
    for eval_json in sorted(run_dir.glob("evaluations/step-*/eval.json")):
        payload = json.loads(eval_json.read_text(encoding="utf-8"))
        step = int(payload["local_step"])
        checkpoint = run_dir / "nerfstudio_models" / f"step-{step:09d}.ckpt"
        if not checkpoint.is_file():
            raise InfrastructureError(f"Evaluated boundary lacks checkpoint: {checkpoint}")
        results = payload["results"]
        boundaries.append(
            Boundary(
                seed=seed,
                local_step=step,
                psnr=float(results["psnr"]),
                ssim=float(results["ssim"]),
                lpips=float(results["lpips"]),
                checkpoint=checkpoint,
                checkpoint_sha256=sha256_file(checkpoint),
                eval_json=eval_json,
                render_dir=Path(payload["render_dir"]),
                completed_wall_time_ns=int(payload["completed_wall_time_ns"]),
            )
        )
    return boundaries


def load_visual_decisions(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: Dict[str, Dict[str, str]] = {}
    for key, value in payload.items():
        if not isinstance(value, Mapping):
            raise InfrastructureError(f"Invalid visual decision for {key}: {value!r}")
        verdict = str(value.get("verdict", "pending"))
        change = str(value.get("change_from_previous", "not_applicable"))
        if verdict not in {"pending", "pass", "fail"}:
            raise InfrastructureError(f"Invalid visual verdict for {key}: {verdict}")
        if change not in {"not_applicable", "improved", "no_improvement", "regressed"}:
            raise InfrastructureError(f"Invalid visual change for {key}: {change}")
        result[str(key)] = {
            "verdict": verdict,
            "change_from_previous": change,
            "note": str(value.get("note", "")),
        }
    return result


def visual_key(frame: str, seed: int, step: int) -> str:
    return f"{frame}:seed-{seed}:step-{step:09d}"


def final_visual_key(frame: str, checkpoint_sha256: str) -> str:
    return f"{frame}:final:{checkpoint_sha256}"


def decision_for(
    decisions: Mapping[str, Mapping[str, str]], frame: str, boundary: Boundary
) -> Mapping[str, str]:
    return decisions.get(
        visual_key(frame, boundary.seed, boundary.local_step),
        {"verdict": "pending", "change_from_previous": "not_applicable", "note": ""},
    )


def boundary_is_valid(
    frame: str,
    boundary: Boundary,
    decisions: Mapping[str, Mapping[str, str]],
) -> bool:
    return boundary.numeric_pass and decision_for(decisions, frame, boundary)["verdict"] == "pass"


def select_boundary(boundaries: Sequence[Boundary]) -> Boundary:
    if not boundaries:
        raise QualityFailure("No valid checkpoint is available for selection")
    maximum = max(row.psnr for row in boundaries)
    tied = [row for row in boundaries if maximum - row.psnr <= PSNR_TIE_DB + 1e-12]
    return min(tied, key=lambda row: (row.lpips, -row.psnr, row.local_step, row.seed))


def contender_seeds(boundaries: Sequence[Boundary]) -> tuple[int, ...]:
    if not boundaries:
        return ()
    maximum = max(row.psnr for row in boundaries)
    return tuple(
        sorted(
            {
                row.seed
                for row in boundaries
                if maximum - row.psnr <= PSNR_TIE_DB + 1e-12
            }
        )
    )


def hard_gate_bootstrap_seeds(
    frame: str,
    boundaries_by_seed: Mapping[int, Sequence[Boundary]],
    decisions: Mapping[str, Mapping[str, str]],
) -> tuple[int, ...]:
    """Choose promising trajectories for one more pre-acceptance interval."""

    latest = []
    for rows in boundaries_by_seed.values():
        if not rows:
            continue
        ordered = sorted(rows, key=lambda row: row.local_step)
        boundary = ordered[-1]
        decision = decision_for(decisions, frame, boundary)
        prior_psnr_pass = any(
            row.psnr >= PSNR_MIN
            and row.ssim >= SSIM_MIN
            and decision_for(decisions, frame, row)["verdict"] == "pass"
            for row in ordered
        )
        lpips_still_converging = (
            len(ordered) >= 2 and boundary.lpips < ordered[-2].lpips
        )
        if (
            prior_psnr_pass
            and (boundary.psnr >= PSNR_MIN or lpips_still_converging)
            and boundary.ssim >= SSIM_MIN
            and decision["verdict"] == "pass"
            and boundary.local_step + INTERVAL <= SCHEDULER_MAX_STEPS
        ):
            latest.append(boundary)
    if not latest:
        return ()
    maximum = max(row.psnr for row in latest)
    return tuple(
        sorted(
            row.seed
            for row in latest
            if maximum - row.psnr <= PSNR_TIE_DB + 1e-12
        )
    )


def plateau_confirmed(
    frame: str,
    boundaries: Sequence[Boundary],
    decisions: Mapping[str, Mapping[str, str]],
) -> bool:
    ordered = sorted(boundaries, key=lambda row: row.local_step)
    if len(ordered) < 3:
        return False
    last = ordered[-3:]
    if not all(boundary_is_valid(frame, row, decisions) for row in last):
        return False
    for previous, current in zip(last, last[1:]):
        if current.local_step - previous.local_step != INTERVAL:
            return False
        if current.psnr - previous.psnr >= PLATEAU_PSNR_GAIN:
            return False
        if current.ssim - previous.ssim >= PLATEAU_SSIM_GAIN:
            return False
        if previous.lpips - current.lpips >= PLATEAU_LPIPS_IMPROVEMENT:
            return False
        if decision_for(decisions, frame, current)["change_from_previous"] not in {
            "no_improvement",
            "regressed",
        }:
            return False
    return True


def _split_pair(path: Path) -> tuple[Image.Image, Image.Image]:
    if not path.is_file():
        raise InfrastructureError(f"Missing render pair: {path}")
    image = Image.open(path).convert("RGB")
    if image.width % 2:
        raise InfrastructureError(f"Expected even-width GT|render pair: {path}")
    width = image.width // 2
    return image.crop((0, 0, width, image.height)), image.crop(
        (width, 0, image.width, image.height)
    )


def _crop_pair(path: Path) -> tuple[Image.Image, Image.Image]:
    ground_truth, render = _split_pair(path)
    left, top, right, bottom = CROP_BOX
    if right > ground_truth.width or bottom > ground_truth.height:
        raise InfrastructureError(
            f"Crop {CROP_BOX} does not fit {ground_truth.width}x{ground_truth.height}: {path}"
        )
    return ground_truth.crop(CROP_BOX), render.crop(CROP_BOX)


def build_native_comparison(
    *,
    frame: str,
    seed: int,
    step: int,
    target_render: Path,
    previous_accepted_render: Path,
    leader_render: Path,
    output_dir: Path,
    previous_boundary_render: Optional[Path] = None,
) -> Dict[str, Any]:
    sources: list[tuple[str, Path]] = [
        ("leader 007740", leader_render),
        (f"accepted {previous_frame(frame)}", previous_accepted_render),
    ]
    if previous_boundary_render is not None:
        sources.append((f"seed {seed} previous boundary", previous_boundary_render))
    sources.append((f"target {frame} seed {seed} step {step}", target_render))

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[str, Image.Image, Image.Image]] = []
    crop_hashes: Dict[str, Dict[str, str]] = {}
    for index, (label, path) in enumerate(sources):
        gt, render = _crop_pair(path)
        stem = f"{index:02d}_{label.lower().replace(' ', '_')}"
        gt_path = output_dir / f"{stem}_gt.png"
        render_path = output_dir / f"{stem}_render.png"
        gt.save(gt_path)
        render.save(render_path)
        crop_hashes[label] = {
            "source": str(path),
            "source_sha256": sha256_file(path),
            "gt": str(gt_path),
            "gt_sha256": sha256_file(gt_path),
            "render": str(render_path),
            "render_sha256": sha256_file(render_path),
        }
        prepared.append((label, gt, render))

    crop_width = CROP_BOX[2] - CROP_BOX[0]
    crop_height = CROP_BOX[3] - CROP_BOX[1]
    gap = 8
    label_height = 24
    canvas = Image.new(
        "RGB",
        (
            crop_width * 2 + gap * 3,
            len(prepared) * (crop_height + label_height + gap) + gap,
        ),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    for row, (label, gt, render) in enumerate(prepared):
        y = gap + row * (crop_height + label_height + gap)
        draw.text((gap + 3, y + 4), f"{label} GT", fill="white")
        draw.text((gap * 2 + crop_width + 3, y + 4), f"{label} render", fill="white")
        canvas.paste(gt, (gap, y + label_height))
        canvas.paste(render, (gap * 2 + crop_width, y + label_height))
    comparison = output_dir / "native_comparison.png"
    canvas.save(comparison)
    payload = {
        "schema_version": 1,
        "frame": frame,
        "seed": seed,
        "local_step": step,
        "crop_xyxy": list(CROP_BOX),
        "native_crop_size": [crop_width, crop_height],
        "comparison": str(comparison),
        "comparison_sha256": sha256_file(comparison),
        "sources": crop_hashes,
        "visual_requirements": [
            "fingers remain separated and sharp",
            "chain remains continuous and gap-free",
            "chain and contact detail remain unblurred",
        ],
    }
    atomic_json(output_dir / "comparison.json", payload)
    return payload


def boundary_payload(boundary: Boundary) -> Dict[str, Any]:
    payload = asdict(boundary)
    for key in ("checkpoint", "eval_json", "render_dir"):
        payload[key] = str(payload[key])
    payload["numeric_pass"] = boundary.numeric_pass
    payload["preferred_pass"] = boundary.preferred_pass
    return payload


def boundary_from_payload(payload: Mapping[str, Any]) -> Boundary:
    return Boundary(
        seed=int(payload["seed"]),
        local_step=int(payload["local_step"]),
        psnr=float(payload["psnr"]),
        ssim=float(payload["ssim"]),
        lpips=float(payload["lpips"]),
        checkpoint=Path(str(payload["checkpoint"])),
        checkpoint_sha256=str(payload["checkpoint_sha256"]),
        eval_json=Path(str(payload["eval_json"])),
        render_dir=Path(str(payload["render_dir"])),
        completed_wall_time_ns=int(payload["completed_wall_time_ns"]),
    )


def read_metrics_rows(path: Path = METRICS_PATH) -> list[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != METRICS_COLUMNS:
            raise InfrastructureError(
                f"Metrics header changed: {reader.fieldnames}, expected {METRICS_COLUMNS}"
            )
        rows = [dict(row) for row in reader]
    frames = [row["frame"] for row in rows]
    if len(frames) != len(set(frames)):
        raise InfrastructureError("metrics.csv contains duplicate frame rows")
    return rows
