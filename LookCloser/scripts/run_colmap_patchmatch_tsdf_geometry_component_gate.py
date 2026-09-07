#!/usr/bin/env python3
"""Run a frozen geometry worker with a post-hoc multi-component visual gate.

The wrapped worker still executes the byte-identical PatchMatch and TSDF commands.
This shim changes only the terminal topology validator: a bounded secondary
component is retained for a later full-resolution visual verdict instead of
causing an automatic repeat of the expensive geometry computation.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Sequence

import numpy as np


MAX_SECONDARY_TRIANGLES = 40_000
MAX_SECONDARY_FRACTION = 0.25
POLICY_NAME = "bounded_multicomponent_pending_visual_gate_v2"
FROZEN_WORKER_NAME = "run_colmap_patchmatch_tsdf_geometry_worker.py"


def component_policy(
    component_triangles: Sequence[int],
    *,
    max_secondary_triangles: int = MAX_SECONDARY_TRIANGLES,
    max_secondary_fraction: float = MAX_SECONDARY_FRACTION,
) -> dict[str, Any]:
    components = sorted((int(value) for value in component_triangles), reverse=True)
    if len(components) < 2 or any(value <= 0 for value in components):
        raise ValueError("The component gate requires at least two positive components")
    secondary_total = sum(components[1:])
    fraction = secondary_total / components[0]
    if secondary_total > max_secondary_triangles or fraction > max_secondary_fraction:
        raise ValueError(
            "Secondary components exceed the campaign-wide visual gate: "
            f"triangles={secondary_total}, fraction={fraction:.6f}"
        )
    return {
        "name": POLICY_NAME,
        "largest_component_triangles": components[0],
        "secondary_component_triangles": components[1:],
        "secondary_triangle_total": secondary_total,
        "secondary_to_largest_fraction": fraction,
        "max_secondary_triangles": int(max_secondary_triangles),
        "max_secondary_fraction": float(max_secondary_fraction),
        "geometry_changed": False,
        "automatic_geometry_retry": False,
        "requires_full_resolution_visual_verdict": True,
    }


def load_frozen_worker(path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("frozen_patchmatch_geometry_worker", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen geometry worker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_geometry_with_component_gate(
    worker: ModuleType,
    frame_id: str,
    data: Path,
    output: Path,
    colmap_build: str,
) -> dict[str, Any]:
    """Repeat the frozen validation, relaxing only bounded component count."""

    split = worker.load_json(data / "transforms.json")
    if len(split.get("train_filenames", [])) != 62 or len(split.get("val_filenames", [])) != 1:
        raise ValueError("Geometry worker input must retain the explicit 62/1 split")
    geometric = sorted((output / "dense/stereo/depth_maps").glob("**/*.geometric.bin"))
    if len(geometric) != 62:
        raise ValueError(f"Expected 62 geometric maps, got {len(geometric)}")
    depth = worker.load_json(output / "depth_dataset/transforms.json")["colmap_mvs_depth"]
    rows = depth.get("depth_maps", [])
    coverages = [float(row["coverage"]) for row in rows]
    if (
        depth.get("train_depth_count") != 62
        or len(rows) != 62
        or {tuple(row.get("shape", [])) for row in rows} != {(1080, 1920)}
        or not all(math.isfinite(value) and value > 0 for value in coverages)
        or not math.isclose(float(depth["coverage_mean"]), float(np.mean(coverages)), abs_tol=1e-12)
        or not math.isclose(float(depth["coverage_min"]), min(coverages), abs_tol=1e-12)
    ):
        raise ValueError("Full-resolution geometric depth validation failed")

    mesh = output / "colmap_patchmatch_tsdf.ply"
    metadata_path = mesh.with_suffix(".json")
    metadata = worker.load_json(metadata_path)
    components = [int(value) for value in metadata.get("component_triangles", [])]
    vertices = int(metadata.get("vertices", 0))
    triangles = int(metadata.get("triangles", 0))
    connected = int(metadata.get("connected_components", 0))
    mesh_hash = worker.sha256(mesh)
    if (
        vertices <= 0
        or triangles <= 0
        or connected != len(components)
        or sum(components) != triangles
        or metadata.get("output_sha256") != mesh_hash
    ):
        raise ValueError("TSDF mesh metadata/hash validation failed")
    result = {
        "schema_version": 1,
        "frame_id": frame_id,
        "geometry_only": True,
        "heldout_rgb_read": False,
        "colmap_build": colmap_build,
        "depth_map_count": 62,
        "depth_shape": [1080, 1920],
        "depth_coverage_mean": float(depth["coverage_mean"]),
        "depth_coverage_min": float(depth["coverage_min"]),
        "mesh_vertices": vertices,
        "mesh_triangles": triangles,
        "mesh_components": connected,
        "mesh_component_triangles": components,
        "mesh_sha256": mesh_hash,
        "mesh_metadata_sha256": worker.sha256(metadata_path),
        "validation_status": "pass",
    }
    if connected == 1:
        return result

    policy = component_policy(components)
    policy.update({
        "frame_id": frame_id,
        "gate_script_sha256": worker.sha256(Path(__file__)),
        "visual_status": "pending",
    })
    result["validation_status"] = "pass_pending_multicomponent_visual_gate"
    result["quality_gate_amendment"] = policy
    return result


def main() -> int:
    default_worker = Path(__file__).resolve().parents[1] / "reconstruct_code" / FROZEN_WORKER_NAME
    frozen_worker = Path(os.environ.get("LOOKCLOSER_FROZEN_GEOMETRY_WORKER", default_worker)).resolve()
    if not frozen_worker.is_file() or frozen_worker == Path(__file__).resolve():
        raise FileNotFoundError(f"Frozen geometry worker is unavailable: {frozen_worker}")
    worker = load_frozen_worker(frozen_worker)
    worker.validate_geometry = lambda frame_id, data, output, colmap_build: validate_geometry_with_component_gate(
        worker, frame_id, data, output, colmap_build
    )
    return int(worker.main())


if __name__ == "__main__":
    raise SystemExit(main())
