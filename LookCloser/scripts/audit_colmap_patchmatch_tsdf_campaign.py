#!/usr/bin/env python3
"""Independently audit and optionally build review sheets for the 50-frame campaign."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from colmap_patchmatch_tsdf_campaign_common import (
    CSV_FIELDS,
    FRAME_COUNT,
    canonical_sha256,
    atomic_json,
    discover_frames,
    load_json,
    sha256,
    validate_hash_manifest,
)


def assert_no_full_frame_metric_keys(payload: object, location: str = "root") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"psnr", "ssim", "lpips", "full_frame_psnr", "full_frame_ssim", "full_frame_lpips"}:
                raise ValueError(f"Forbidden non-face metric key {key!r} at {location}")
            assert_no_full_frame_metric_keys(value, f"{location}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_no_full_frame_metric_keys(value, f"{location}[{index}]")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("metrics.csv header/order differs from the campaign schema")
        return list(reader)


def verify_final_campaign_manifest(campaign_manifest: dict, ordered: list[str], results: list[dict]) -> None:
    if campaign_manifest.get("status") != "complete":
        raise ValueError("Final campaign manifest status is not complete")
    frame_states = campaign_manifest.get("frame_states", {})
    if list(frame_states) != ordered:
        raise ValueError("Final campaign manifest frame inventory/order differs from the request")
    for result in results:
        frame_id = result["frame_id"]
        state = frame_states[frame_id]
        for key, expected in (
            ("state", result["status"]),
            ("metric_status", result["metric_status"]),
            ("visual_status", result["visual_status"]),
        ):
            if state.get(key) != expected:
                raise ValueError(f"Final campaign manifest/result mismatch {frame_id}:{key}")


def verify_render_revision(root: Path, result: dict) -> None:
    revision = result.get("render_revision")
    if not isinstance(revision, dict):
        raise ValueError(f"Published result has no render revision: {result.get('frame_id')}")
    correction_id = revision.get("correction_id")
    if not isinstance(correction_id, str):
        raise ValueError("Render revision has no correction ID")
    base = root / ".diagnostics" / f"render_revision_{correction_id}"
    correction = load_json(base / "correction_manifest.json")
    if correction.get("status") != "complete" or correction.get("correction_id") != correction_id:
        raise ValueError(f"Invalid base render correction: {correction_id}")
    arguments = revision.get("render_arguments", {})
    if (
        arguments.get("uses_eval_rgb_for_prediction") is not False
        or arguments.get("uses_masks") is not False
        or arguments.get("averages_sources") is not False
        or arguments.get("primary_color_continuation") is not True
    ):
        raise ValueError(f"Invalid render correction policy for {result['frame_id']}")
    before = Path(revision["remote_render_preserved_in"]) / "eval_pred_0000.png"
    current = root / "frames" / result["frame_id"] / "render/eval_pred_0000.png"
    if not before.is_file() or sha256(before) != revision["before_render_sha256"]:
        raise ValueError(f"Superseded remote render/hash is missing for {result['frame_id']}")
    if sha256(current) != revision["after_render_sha256"]:
        raise ValueError(f"Revised render/hash mismatch for {result['frame_id']}")
    if revision.get("extension") is True:
        extension = root / ".diagnostics" / f"render_revision_{correction_id}_extensions" / result["frame_id"]
        state = load_json(extension / "state.json")
        receipt = load_json(extension / "extension_receipt.json")
        if state.get("status") != "complete" or receipt.get("frame_id") != result["frame_id"]:
            raise ValueError(f"Incomplete render correction extension for {result['frame_id']}")
        if receipt.get("after_render_sha256") != result["render_sha256"] or receipt.get("mesh_unchanged") is not True:
            raise ValueError(f"Invalid render correction extension receipt for {result['frame_id']}")
    else:
        receipt = load_json(base / "frame_receipts" / f"{result['frame_id']}.json")
        if receipt.get("after_render_sha256") != result["render_sha256"] or receipt.get("mesh_unchanged") is not True:
            raise ValueError(f"Invalid base render correction receipt for {result['frame_id']}")


def verify_frame(root: Path, frame_id: str, row: dict[str, str]) -> dict:
    frame = root / "frames" / frame_id
    result = load_json(frame / "result.json")
    metrics = load_json(frame / "metrics.json")
    visual = load_json(frame / "visual_review.json")
    remote = load_json(frame / "remote_result.json")
    retained_manifest = load_json(frame / "retained_manifest.json")
    validate_hash_manifest(frame, retained_manifest)
    if result["frame_id"] != frame_id or row["frame_id"] != frame_id:
        raise ValueError(f"Frame identity mismatch for {frame_id}")
    for key in CSV_FIELDS:
        expected = result[key]
        actual = row[key]
        if isinstance(expected, bool):
            if actual.lower() != str(expected).lower():
                raise ValueError(f"CSV/result mismatch {frame_id}:{key}")
        elif str(expected) != actual:
            raise ValueError(f"CSV/result mismatch {frame_id}:{key}: {actual!r} != {expected!r}")
    values = [float(result[key]) for key in ("face_psnr", "face_ssim", "face_lpips")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite face metrics for {frame_id}")
    if visual.get("visual_status") not in {"pass", "fail"} or result.get("visual_status") not in {"pass", "fail"}:
        raise ValueError(f"Pending/uncertain visual result for {frame_id}")
    if metrics.get("protocol", {}).get("candidate_surface_mask") is not False:
        raise ValueError(f"Candidate-defined metric mask detected for {frame_id}")
    if metrics.get("protocol", {}).get("roi") != "manual_polygon_on_heldout_gt_only":
        raise ValueError(f"Non-manual face ROI detected for {frame_id}")
    assert_no_full_frame_metric_keys(metrics)
    assert_no_full_frame_metric_keys(load_json(frame / "render" / "reprojection_audit.json"))
    mesh = frame / "mesh" / "colmap_patchmatch_tsdf.ply"
    render = frame / "render" / "eval_pred_0000.png"
    if sha256(mesh) != result["mesh_sha256"] or sha256(render) != result["render_sha256"]:
        raise ValueError(f"Published mesh/render hash mismatch for {frame_id}")
    verify_render_revision(root, result)
    if int(remote["depth_map_count"]) != 62 or remote["depth_shape"] != [1080, 1920]:
        raise ValueError(f"Depth inventory/shape mismatch for {frame_id}")
    if not math.isfinite(float(remote["depth_coverage_mean"])) or float(remote["depth_coverage_min"]) <= 0:
        raise ValueError(f"Invalid depth coverage for {frame_id}")
    if int(remote["mesh_vertices"]) <= 0 or int(remote["mesh_triangles"]) <= 0 or int(remote["mesh_components"]) <= 0:
        raise ValueError(f"Empty mesh for {frame_id}")
    gt_audit = load_json(frame / "render" / "eval_ground_truth.json")
    source_exr = Path(gt_audit["source_exr"])
    if not source_exr.is_file() or sha256(source_exr) != gt_audit["source_exr_sha256"]:
        raise ValueError(f"Immutable held-out EXR reference/hash mismatch for {frame_id}")
    for crop in ("face_ear_hair", "ear_native", "lipstick_lips_hand", "actor_overview"):
        if not (frame / "visual" / f"{crop}_gt_pred.png").is_file():
            raise FileNotFoundError(f"Missing visual crop {frame_id}:{crop}")
    return result


def labeled(image: Image.Image, frame_id: str) -> Image.Image:
    bar = 32
    result = Image.new("RGB", (image.width, image.height + bar), "white")
    result.paste(image, (0, bar))
    ImageDraw.Draw(result).text((8, 8), f"{frame_id}  GT | prediction", fill="black")
    return result


def stack_images(paths: list[tuple[str, Path]], output: Path) -> None:
    images = [labeled(Image.open(path).convert("RGB"), frame_id) for frame_id, path in paths]
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    sheet = Image.new("RGB", (width, height), (32, 32, 32))
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    sheet.save(temporary, format="PNG", compress_level=3)
    os.replace(temporary, output)


def build_contact_sheets(root: Path, ordered: list[str]) -> dict:
    complete = [frame_id for frame_id in ordered if (root / "frames" / frame_id / "result.json").is_file()]
    sheets = []
    for start in range(0, len(complete), 10):
        batch = complete[start:start + 10]
        if not batch:
            continue
        for crop in ("ear_native", "lipstick_lips_hand", "face_ear_hair", "actor_overview"):
            output = root / "contact_sheets" / f"{batch[0]}_{batch[-1]}_{crop}_gt_pred.png"
            paths = [(frame_id, root / "frames" / frame_id / "visual" / f"{crop}_gt_pred.png") for frame_id in batch]
            stack_images(paths, output)
            sheets.append({"frames": batch, "crop": crop, "path": str(output), "sha256": sha256(output)})
    receipt = {"schema_version": 1, "built_at": datetime.now(timezone.utc).isoformat(), "sheets": sheets}
    atomic_json(root / "contact_sheets" / "manifest.json", receipt)
    campaign_manifest_path = root / "campaign_manifest.json"
    campaign_manifest = load_json(campaign_manifest_path)
    boundaries = [3, 10, 20, 30, 40, 50]
    campaign_manifest["visual_batches_completed"] = [value for value in boundaries if len(complete) >= value]
    campaign_manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(campaign_manifest_path, campaign_manifest)
    return receipt


def audit(root: Path, *, allow_incomplete: bool) -> dict:
    request = load_json(root / "campaign_request.json")
    request_without_hash = {key: value for key, value in request.items() if key != "request_sha256"}
    if canonical_sha256(request_without_hash) != request.get("request_sha256"):
        raise ValueError("campaign_request.json self-hash mismatch")
    ordered = request.get("ordered_frame_ids")
    discovered = [path.name for path in discover_frames(Path(request["source_root"]))]
    if ordered != discovered or len(set(ordered)) != FRAME_COUNT:
        raise ValueError("Immutable request frame inventory/order differs from source discovery")
    for source in request["source_inventory"]:
        transforms = Path(source["source_dataset"]) / "transforms.json"
        if sha256(transforms) != source["source_transforms_sha256"]:
            raise ValueError(f"Source transforms changed: {transforms}")
    for script in request["scripts"]:
        configured = root / "config" / "code" / script["name"]
        if sha256(configured) != script["sha256"]:
            raise ValueError(f"Configured campaign code hash mismatch: {configured}")
    campaign_manifest = load_json(root / "campaign_manifest.json")
    for row in campaign_manifest.get("render_corrections", []):
        path = Path(row["manifest"])
        if not path.is_file() or sha256(path) != row["manifest_sha256"]:
            raise ValueError(f"Render correction manifest hash mismatch: {path}")
    for row in campaign_manifest.get("render_correction_extensions", []):
        path = Path(row["receipt"])
        if not path.is_file() or sha256(path) != row["receipt_sha256"]:
            raise ValueError(f"Render correction extension receipt hash mismatch: {path}")
    rows = read_csv(root / "metrics.csv")
    ids = [row["frame_id"] for row in rows]
    if len(ids) != len(set(ids)) or ids != ordered[:len(ids)]:
        raise ValueError("metrics.csv contains duplicates or is not a strict ordered prefix")
    frame_directories = sorted(path.name for path in (root / "frames").iterdir() if path.is_dir())
    if frame_directories != sorted(ids):
        raise ValueError("Published frame directory inventory differs from metrics.csv")
    results = [verify_frame(root, frame_id, row) for frame_id, row in zip(ids, rows)]
    if not allow_incomplete and len(results) != FRAME_COUNT:
        raise ValueError(f"Final audit requires {FRAME_COUNT} rows, got {len(results)}")
    if not allow_incomplete:
        verify_final_campaign_manifest(campaign_manifest, ordered, results)
        contacts = load_json(root / "contact_sheets" / "manifest.json")
        if len(contacts.get("sheets", [])) != 20:
            raise ValueError("Final campaign requires four contact sheets for each of five ten-frame batches")
        for row in contacts["sheets"]:
            path = Path(row["path"])
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Contact-sheet hash mismatch: {path}")
    metrics = np.asarray([[row[key] for key in ("face_psnr", "face_ssim", "face_lpips")] for row in results], dtype=float) if results else np.empty((0, 3))
    visual_fail = sum(row["visual_status"] == "fail" for row in results)
    all_pass = len(results) == FRAME_COUNT and visual_fail == 0 and all(row["status"] == "pass" for row in results)
    summary = {
        "schema_version": 1,
        "status": "pass" if all_pass else ("complete_with_failures" if len(results) == FRAME_COUNT else "incomplete"),
        "frame_count": len(results),
        "visual_pass": sum(row["visual_status"] == "pass" for row in results),
        "visual_fail": visual_fail,
        "face_metrics_min_median_max": (
            {} if not len(results) else {
                key: [float(np.min(metrics[:, index])), float(np.median(metrics[:, index])), float(np.max(metrics[:, index]))]
                for index, key in enumerate(("face_psnr", "face_ssim", "face_lpips"))
            }
        ),
        "no_full_frame_metrics": True,
    }
    atomic_json(root / "campaign_audit.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50"))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--build-contact-sheets", action="store_true")
    args = parser.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    request = load_json(args.output_root / "campaign_request.json")
    if args.build_contact_sheets:
        build_contact_sheets(args.output_root, request["ordered_frame_ids"])
    summary = audit(args.output_root, allow_incomplete=args.allow_incomplete)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
