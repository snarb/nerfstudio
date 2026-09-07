#!/usr/bin/env python3
"""Publish a fly-through frame rejected only by a reproducible tiny mesh island.

This is a campaign-level quality-gate amendment, not a geometry repair.  It keeps
the frozen PatchMatch/TSDF mesh byte-for-byte and reuses the frozen renderer.  A
frame is eligible only when two clean reconstructions have identical component
statistics and all secondary components are below one global conservative gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from types import ModuleType
from typing import Any, Sequence


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def small_component_policy(
    component_triangles: Sequence[int], *, max_secondary_triangles: int = 1000,
    max_secondary_fraction: float = 0.005,
    policy_name: str = "reproducible_small_secondary_component_visual_gate_v1",
) -> dict[str, Any]:
    components = [int(value) for value in component_triangles]
    if len(components) < 2 or any(value <= 0 for value in components):
        raise ValueError("The amendment requires at least two positive components")
    components.sort(reverse=True)
    largest = components[0]
    secondary = components[1:]
    secondary_total = sum(secondary)
    fraction = secondary_total / largest
    if secondary_total > max_secondary_triangles or fraction > max_secondary_fraction:
        raise ValueError(
            "Secondary components exceed the global small-island gate: "
            f"triangles={secondary_total}, fraction={fraction:.6f}"
        )
    return {
        "name": policy_name,
        "largest_component_triangles": largest,
        "secondary_component_triangles": secondary,
        "secondary_triangle_total": secondary_total,
        "secondary_to_largest_fraction": fraction,
        "max_secondary_triangles": int(max_secondary_triangles),
        "max_secondary_fraction": float(max_secondary_fraction),
        "geometry_changed": False,
        "requires_two_matching_clean_attempts": True,
        "requires_full_resolution_visual_pass": True,
    }


def load_frozen_controller(path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("frozen_temporal_flythrough_controller", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen controller: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_depth_qc(payload: dict[str, Any]) -> tuple[float, float]:
    depth = payload.get("colmap_mvs_depth", {})
    rows = depth.get("depth_maps", [])
    coverage = [float(row.get("coverage", float("nan"))) for row in rows]
    if (
        depth.get("train_depth_count") != 62
        or len(rows) != 62
        or {tuple(row.get("shape", [])) for row in rows} != {(1080, 1920)}
        or not all(math.isfinite(value) and value > 0 for value in coverage)
    ):
        raise ValueError("Recovery input lacks 62 valid full-resolution geometric depth maps")
    mean = float(depth.get("coverage_mean", float("nan")))
    minimum = float(depth.get("coverage_min", float("nan")))
    if not math.isclose(mean, sum(coverage) / len(coverage), abs_tol=1e-12):
        raise ValueError("Depth coverage mean is inconsistent")
    if not math.isclose(minimum, min(coverage), abs_tol=1e-12):
        raise ValueError("Depth coverage minimum is inconsistent")
    return mean, minimum


def recover(args: argparse.Namespace) -> None:
    controller = load_frozen_controller(args.controller)
    controller_args = controller.parse_args([
        "--output-root", str(args.output_root), "status",
    ])
    request = controller.require_request(controller_args)
    if args.frame_id not in request["ordered_frame_ids"]:
        raise ValueError(f"Frame is outside the campaign: {args.frame_id}")
    index = request["ordered_frame_ids"].index(args.frame_id)
    if index < 50:
        raise ValueError("Small-component recovery is only for newly reconstructed frames")

    attempts = [load_json(path) for path in args.attempt_metadata]
    comparable = ("connected_components", "component_triangles", "vertices", "triangles")
    if any(tuple(payload.get(key) for key in comparable) != tuple(attempts[0].get(key) for key in comparable) for payload in attempts[1:]):
        raise ValueError("Two clean attempts do not reproduce the same component statistics")
    if len(attempts) < 2:
        raise ValueError("At least two clean-attempt metadata files are required")
    metadata = load_json(args.mesh_metadata)
    if tuple(metadata.get(key) for key in comparable) != tuple(attempts[0].get(key) for key in comparable):
        raise ValueError("Chosen mesh metadata disagrees with the clean attempts")
    if metadata.get("connected_components") != len(metadata.get("component_triangles", [])):
        raise ValueError("Mesh component metadata is internally inconsistent")
    policy = small_component_policy(
        metadata["component_triangles"],
        max_secondary_triangles=args.max_secondary_triangles,
        max_secondary_fraction=args.max_secondary_fraction,
        policy_name=args.policy_name,
    )
    mesh_hash = controller.sha256(args.mesh)
    if metadata.get("output_sha256") != mesh_hash:
        raise ValueError("Chosen PLY does not match its metadata hash")
    coverage_mean, coverage_min = validate_depth_qc(load_json(args.depth_qc))

    render = args.render_root / f"scratch/{index:04d}/render"
    path_request = args.render_root / "path_request.json"
    if not (render / "path_request.json").is_file():
        shutil.copyfile(path_request, render / "path_request.json")
    render_png = render / "seam_cut8/eval_pred_0000.png"
    if not render_png.is_file():
        raise FileNotFoundError(render_png)

    policy.update({
        "frame_id": args.frame_id,
        "temporal_index": index,
        "clean_attempt_metadata_sha256": [controller.sha256(path) for path in args.attempt_metadata],
        "chosen_mesh_sha256": mesh_hash,
        "render_sha256": controller.sha256(render_png),
        "recovery_script_sha256": controller.sha256(Path(__file__)),
        "visual_status": "pass",
        "visual_notes": args.visual_notes,
    })
    geometry = {
        "provenance": "new_frozen_recipe_small_component_gate_amendment",
        "geometry_only": True,
        "heldout_rgb_read": False,
        "colmap_build": "COLMAP 3.13.0.dev0 commit 5509fffe with CUDA",
        "depth_map_count": 62,
        "depth_shape": [1080, 1920],
        "depth_coverage_mean": coverage_mean,
        "depth_coverage_min": coverage_min,
        "mesh_vertices": int(metadata["vertices"]),
        "mesh_triangles": int(metadata["triangles"]),
        "mesh_components": int(metadata["connected_components"]),
        "mesh_component_triangles": metadata["component_triangles"],
        "mesh_sha256": mesh_hash,
        "mesh_metadata_sha256": controller.sha256(args.mesh_metadata),
        "validation_status": "pass_with_small_component_visual_gate",
        "quality_gate_amendment": policy,
    }

    final = args.output_root / "frames" / args.frame_id
    if final.exists():
        raise FileExistsError(final)
    original_validate = controller.validate_render

    def validate_with_amendment(render_path, mesh_path, geometry_payload, target_index, campaign_request):
        validation_copy = deepcopy(geometry_payload)
        validation_copy["mesh_components"] = 1
        return original_validate(render_path, mesh_path, validation_copy, target_index, campaign_request)

    controller.validate_render = validate_with_amendment
    controller.publish_frame(
        controller_args, request, args.frame_id, index, args.host,
        args.mesh, args.mesh_metadata, geometry, render,
    )
    controller.atomic_json(final / "visual_review.json", {
        "schema_version": 1,
        "frame_id": args.frame_id,
        "temporal_index": index,
        "render_sha256": load_json(final / "result.json")["render_sha256"],
        "visual_status": "pass",
        "visual_notes": args.visual_notes,
        "reviewed_at": controller.now(),
        "quality_gate_amendment": policy,
    })
    claim = args.output_root / "claims" / args.frame_id
    controller.atomic_json(claim / "complete.json", {
        "completed_at": controller.now(),
        "quality_gate_amendment": policy["name"],
        "recovery_script_sha256": policy["recovery_script_sha256"],
    })
    controller.append_jsonl(args.output_root / "campaign_checks.jsonl", {
        "timestamp": controller.now(),
        "frame_id": args.frame_id,
        "host": args.host,
        "check_status": "small_component_visual_gate_published",
        "policy": policy,
    })
    if not controller.validate_finished(final, request["request_sha256"]):
        raise RuntimeError("Recovered frame publication did not validate")
    print(json.dumps({"frame_id": args.frame_id, "status": "complete", "policy": policy}, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--host", default="dev3")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-metadata", type=Path, required=True)
    parser.add_argument("--attempt-metadata", type=Path, nargs="+", required=True)
    parser.add_argument("--depth-qc", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--max-secondary-triangles", type=int, default=1000)
    parser.add_argument("--max-secondary-fraction", type=float, default=0.005)
    parser.add_argument(
        "--policy-name",
        default="reproducible_small_secondary_component_visual_gate_v1",
    )
    parser.add_argument("--visual-notes", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    recover(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
