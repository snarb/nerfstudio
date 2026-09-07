#!/usr/bin/env python3
"""Resume a frozen temporal campaign with an opt-in component visual gate."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Sequence


POLICY_NAME = "bounded_multicomponent_pending_visual_gate_v2"


def load_controller(path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("frozen_temporal_flythrough_controller", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen controller: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def component_geometry_is_eligible(geometry: dict[str, Any], gate_sha256: str) -> bool:
    policy = geometry.get("quality_gate_amendment", {})
    components = [int(value) for value in geometry.get("mesh_component_triangles", [])]
    if len(components) < 2 or any(value <= 0 for value in components):
        return False
    components.sort(reverse=True)
    secondary = sum(components[1:])
    return (
        geometry.get("validation_status") == "pass_pending_multicomponent_visual_gate"
        and int(geometry.get("mesh_components", 0)) == len(components)
        and int(geometry.get("mesh_triangles", 0)) == sum(components)
        and policy.get("name") == POLICY_NAME
        and policy.get("gate_script_sha256") == gate_sha256
        and policy.get("geometry_changed") is False
        and policy.get("automatic_geometry_retry") is False
        and policy.get("visual_status") == "pending"
        and secondary <= int(policy.get("max_secondary_triangles", -1))
        and secondary / components[0] <= float(policy.get("max_secondary_fraction", -1))
    )


def patch_controller(
    controller: ModuleType, gate_worker: Path, gate_hash: str | None = None,
) -> None:
    gate_hash = controller.sha256(gate_worker) if gate_hash is None else gate_hash
    controller.GEOMETRY_WORKER_NAME = gate_worker
    original_validate_render = controller.validate_render

    def validate_render_with_component_gate(
        render: Path,
        mesh: Path,
        geometry: dict[str, Any],
        index: int,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        if int(geometry.get("mesh_components", 0)) == 1:
            return original_validate_render(render, mesh, geometry, index, request)
        if not component_geometry_is_eligible(geometry, gate_hash):
            raise ValueError("Multi-component geometry lacks the bounded pending-visual gate")
        validation_copy = deepcopy(geometry)
        validation_copy["mesh_components"] = 1
        return original_validate_render(render, mesh, validation_copy, index, request)

    controller.validate_render = validate_render_with_component_gate

    def safe_rsync(source: str, destination: str, *, delete: bool = False) -> None:
        command = ["rsync", "-rl"]
        if delete:
            command.append("--delete")
        subprocess.run([*command, source, destination], check=True)

    controller.rsync = safe_rsync
    original_join = shlex.join
    controller.shlex.join = lambda values: original_join([str(value) for value in values])

    def safe_quarantine(args: argparse.Namespace, path: Path, label: str) -> Path:
        destination = args.output_root / "quarantine" / f"{label}.{int(time.time())}.{os.getpid()}"
        destination.parent.mkdir(exist_ok=True)
        if path.stat().st_dev == destination.parent.stat().st_dev:
            os.replace(path, destination)
            return destination
        native = path.parent / ".quarantine"
        native.mkdir(exist_ok=True)
        retained = native / destination.name
        os.replace(path, retained)
        controller.atomic_json(destination.with_suffix(destination.suffix + ".json"), {
            "schema_version": 1,
            "label": label,
            "retained_native_path": str(retained),
            "reason": "cross_device_scratch_quarantine",
            "timestamp": controller.now(),
        })
        return retained

    controller.quarantine_path = safe_quarantine


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--geometry-worker", type=Path, required=True)
    parser.add_argument(
        "--geometry-worker-hash-source", type=Path, default=None,
        help="Local byte-identical copy used to validate a remote --geometry-worker path.",
    )
    parser.add_argument("--expected-geometry-worker-sha256", required=True)
    args, remainder = parser.parse_known_args(argv)
    args.controller = args.controller.expanduser().resolve()
    args.geometry_worker = args.geometry_worker.expanduser()
    if not args.geometry_worker.is_absolute():
        args.geometry_worker = args.geometry_worker.absolute()
    if args.geometry_worker_hash_source is not None:
        args.geometry_worker_hash_source = args.geometry_worker_hash_source.expanduser().resolve()
    if not remainder or remainder[0] != "process":
        parser.error("The wrapper accepts only a frozen controller process action")
    return args, remainder


def main(argv: Sequence[str] | None = None) -> int:
    args, remainder = parse_args(argv)
    controller = load_controller(args.controller)
    hash_source = args.geometry_worker if args.geometry_worker_hash_source is None else args.geometry_worker_hash_source
    if controller.sha256(hash_source) != args.expected_geometry_worker_sha256:
        raise ValueError("Component-gate geometry worker hash mismatch")
    patch_controller(controller, args.geometry_worker, args.expected_geometry_worker_sha256)
    sys.argv = [str(args.controller), *remainder]
    return int(controller.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
