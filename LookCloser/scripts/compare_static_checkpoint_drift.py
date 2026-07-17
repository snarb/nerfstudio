#!/usr/bin/env python3
"""Compare static LookCloser model drift without modifying checkpoints.

The first positional checkpoint is the reference.  Every later checkpoint is
compared with it independently.  Tensor arithmetic is performed in bounded
CPU chunks so that float64 accumulation does not duplicate a large hash table.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch


FIELD_KEYS = (
    "_model.field.encoding.params",
    "_model.field.mlp_geo.params",
    "_model.field.mlp_color.params",
)
OCCUPANCY_PREFIXES = (
    "_model.occupancy_grid.",
    "_model.adaptive_sampler.occupancy_grid.",
)
DEFAULT_CHUNK_ELEMENTS = 1 << 20


class CheckpointValidationError(ValueError):
    """Raised when checkpoints cannot be compared exactly."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        nargs="+",
        type=Path,
        help="Reference checkpoint followed by one or more candidates",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination JSON report")
    parser.add_argument(
        "--chunk-elements",
        type=int,
        default=DEFAULT_CHUNK_ELEMENTS,
        help=f"Maximum elements converted to float64 at once (default: {DEFAULT_CHUNK_ELEMENTS})",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        load_mode = "mmap"
    except RuntimeError as exc:
        # Legacy (pre-zip) torch.save files do not support mmap.  Retain a
        # compatibility path while reporting the less memory-efficient mode.
        if "mmap" not in str(exc).lower():
            raise
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        load_mode = "cpu_fallback"
    if not isinstance(checkpoint, dict):
        raise CheckpointValidationError(f"{path}: checkpoint root must be a dict")
    return checkpoint, load_mode


def _trainer_step(checkpoint: Mapping[str, Any], path: Path) -> int:
    if "step" not in checkpoint:
        raise CheckpointValidationError(f"{path}: missing trainer step")
    step = checkpoint["step"]
    if isinstance(step, torch.Tensor):
        if step.numel() != 1:
            raise CheckpointValidationError(f"{path}: trainer step tensor must be scalar")
        step = step.item()
    if isinstance(step, bool) or not isinstance(step, int):
        raise CheckpointValidationError(f"{path}: trainer step must be an integer, got {type(step).__name__}")
    return step


def _pipeline(checkpoint: Mapping[str, Any], path: Path) -> Mapping[str, torch.Tensor]:
    pipeline = checkpoint.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise CheckpointValidationError(f"{path}: pipeline must be a mapping")
    return pipeline


def _require_tensor(pipeline: Mapping[str, Any], key: str, path: Path) -> torch.Tensor:
    if key not in pipeline:
        raise CheckpointValidationError(f"{path}: missing required tensor {key!r}")
    tensor = pipeline[key]
    if not isinstance(tensor, torch.Tensor):
        raise CheckpointValidationError(f"{path}: pipeline entry {key!r} is not a tensor")
    if tensor.is_complex():
        raise CheckpointValidationError(f"{path}: complex tensor {key!r} is unsupported")
    return tensor


def _occupancy_keys(pipeline: Mapping[str, Any], path: Path) -> tuple[str, ...]:
    suffix_sets: list[set[str]] = []
    keys: list[str] = []
    for prefix in OCCUPANCY_PREFIXES:
        prefix_keys = sorted(key for key in pipeline if key.startswith(prefix))
        if not prefix_keys:
            raise CheckpointValidationError(f"{path}: no occupancy tensors under {prefix!r}")
        suffixes: set[str] = set()
        for key in prefix_keys:
            _require_tensor(pipeline, key, path)
            suffixes.add(key.removeprefix(prefix))
            keys.append(key)
        suffix_sets.append(suffixes)
    if suffix_sets[0] != suffix_sets[1]:
        raise CheckpointValidationError(
            f"{path}: model/adaptive occupancy suffix keys differ: "
            f"{sorted(suffix_sets[0])!r} != {sorted(suffix_sets[1])!r}"
        )
    if "occs" not in suffix_sets[0]:
        raise CheckpointValidationError(f"{path}: occupancy state is missing required 'occs' tensors")
    return tuple(sorted(keys))


def _validate_duplicate_occupancy(
    pipeline: Mapping[str, Any], occupancy_keys: tuple[str, ...], path: Path
) -> dict[str, bool]:
    suffixes = sorted(
        {
            key.removeprefix(prefix)
            for key in occupancy_keys
            for prefix in OCCUPANCY_PREFIXES
            if key.startswith(prefix)
        }
    )
    result: dict[str, bool] = {}
    for suffix in suffixes:
        model = _require_tensor(pipeline, OCCUPANCY_PREFIXES[0] + suffix, path)
        sampler = _require_tensor(pipeline, OCCUPANCY_PREFIXES[1] + suffix, path)
        if model.shape != sampler.shape or model.dtype != sampler.dtype:
            raise CheckpointValidationError(
                f"{path}: duplicate occupancy metadata differs for suffix {suffix!r}"
            )
        result[suffix] = bool(torch.equal(model, sampler))
    return result


def _paired_chunks(
    reference: torch.Tensor, candidate: torch.Tensor, max_elements: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield matching tensor views with at most ``max_elements`` each."""

    if reference.numel() <= max_elements:
        yield reference, candidate
        return

    split_dim = next((index for index, size in enumerate(reference.shape) if size > 1), None)
    if split_dim is None:
        # A tensor with more than one element necessarily has a splittable
        # dimension.  Keep this guard explicit for malformed tensor subclasses.
        raise RuntimeError("Unable to split tensor into bounded chunks")
    elements_per_index = reference.numel() // reference.shape[split_dim]
    indices_per_chunk = max(1, max_elements // elements_per_index)
    for start in range(0, reference.shape[split_dim], indices_per_chunk):
        length = min(indices_per_chunk, reference.shape[split_dim] - start)
        reference_slice = reference.narrow(split_dim, start, length)
        candidate_slice = candidate.narrow(split_dim, start, length)
        yield from _paired_chunks(reference_slice, candidate_slice, max_elements)


def _tensor_sums(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    chunk_elements: int,
    key: str,
) -> tuple[float, float, float]:
    reference_squares: list[float] = []
    candidate_squares: list[float] = []
    difference_squares: list[float] = []
    for reference_chunk, candidate_chunk in _paired_chunks(reference, candidate, chunk_elements):
        # copy=True is important when a checkpoint already stores float64: the
        # in-place subtraction below must never mutate its mmap-backed storage.
        reference_f64 = reference_chunk.detach().to(dtype=torch.float64, device="cpu", copy=True).reshape(-1)
        candidate_f64 = candidate_chunk.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        if not bool(torch.isfinite(reference_f64).all()):
            raise CheckpointValidationError(f"reference tensor {key!r} contains non-finite values")
        if not bool(torch.isfinite(candidate_f64).all()):
            raise CheckpointValidationError(f"candidate tensor {key!r} contains non-finite values")
        reference_squares.append(float(torch.dot(reference_f64, reference_f64)))
        candidate_squares.append(float(torch.dot(candidate_f64, candidate_f64)))
        reference_f64.sub_(candidate_f64)
        difference_squares.append(float(torch.dot(reference_f64, reference_f64)))
    return math.fsum(reference_squares), math.fsum(candidate_squares), math.fsum(difference_squares)


def _norm_payload(reference_sq: float, candidate_sq: float, difference_sq: float) -> dict[str, float]:
    reference_l2 = math.sqrt(reference_sq)
    candidate_l2 = math.sqrt(candidate_sq)
    difference_l2 = math.sqrt(difference_sq)
    denominator = reference_l2 + candidate_l2
    relative = 0.0 if denominator == 0.0 else 2.0 * difference_l2 / denominator
    return {
        "reference_l2": reference_l2,
        "candidate_l2": candidate_l2,
        "difference_l2": difference_l2,
        "symmetric_relative_l2": relative,
    }


def _compare_group(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    reference_path: Path,
    candidate_path: Path,
    chunk_elements: int,
    include_suffix_aggregates: bool = False,
) -> dict[str, Any]:
    per_key: dict[str, dict[str, Any]] = {}
    aggregate_sums = [0.0, 0.0, 0.0]
    suffix_sums: dict[str, list[float]] = {}
    for key in keys:
        reference_tensor = _require_tensor(reference, key, reference_path)
        candidate_tensor = _require_tensor(candidate, key, candidate_path)
        if reference_tensor.shape != candidate_tensor.shape:
            raise CheckpointValidationError(
                f"shape mismatch for {key!r}: {list(reference_tensor.shape)} != {list(candidate_tensor.shape)}"
            )
        if reference_tensor.dtype != candidate_tensor.dtype:
            raise CheckpointValidationError(
                f"dtype mismatch for {key!r}: {reference_tensor.dtype} != {candidate_tensor.dtype}"
            )
        sums = _tensor_sums(
            reference_tensor,
            candidate_tensor,
            chunk_elements=chunk_elements,
            key=key,
        )
        for index, value in enumerate(sums):
            aggregate_sums[index] += value
        payload: dict[str, Any] = {
            "shape": list(reference_tensor.shape),
            "dtype": str(reference_tensor.dtype),
            "numel": reference_tensor.numel(),
            **_norm_payload(*sums),
        }
        per_key[key] = payload
        if include_suffix_aggregates:
            prefix = next(prefix for prefix in OCCUPANCY_PREFIXES if key.startswith(prefix))
            suffix = key.removeprefix(prefix)
            accumulator = suffix_sums.setdefault(suffix, [0.0, 0.0, 0.0])
            for index, value in enumerate(sums):
                accumulator[index] += value

    result: dict[str, Any] = {
        "keys": list(keys),
        "aggregate": _norm_payload(*aggregate_sums),
        "per_key": per_key,
    }
    if include_suffix_aggregates:
        result["by_suffix"] = {
            suffix: {
                "keys": [prefix + suffix for prefix in OCCUPANCY_PREFIXES],
                "aggregate": _norm_payload(*sums),
            }
            for suffix, sums in sorted(suffix_sums.items())
        }
    return result


def compare_checkpoints(paths: list[Path], *, chunk_elements: int) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("At least two checkpoints are required")
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    normalized_paths = [path.expanduser().resolve() for path in paths]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("Checkpoint paths must be unique")

    reference_path = normalized_paths[0]
    reference_checkpoint, reference_load_mode = _load_checkpoint(reference_path)
    reference_pipeline = _pipeline(reference_checkpoint, reference_path)
    for key in FIELD_KEYS:
        _require_tensor(reference_pipeline, key, reference_path)
    reference_occupancy_keys = _occupancy_keys(reference_pipeline, reference_path)
    reference_duplicates = _validate_duplicate_occupancy(
        reference_pipeline, reference_occupancy_keys, reference_path
    )
    reference_step = _trainer_step(reference_checkpoint, reference_path)

    comparisons: list[dict[str, Any]] = []
    for candidate_path in normalized_paths[1:]:
        candidate_checkpoint, candidate_load_mode = _load_checkpoint(candidate_path)
        candidate_pipeline = _pipeline(candidate_checkpoint, candidate_path)
        for key in FIELD_KEYS:
            _require_tensor(candidate_pipeline, key, candidate_path)
        candidate_occupancy_keys = _occupancy_keys(candidate_pipeline, candidate_path)
        if candidate_occupancy_keys != reference_occupancy_keys:
            missing = sorted(set(reference_occupancy_keys) - set(candidate_occupancy_keys))
            extra = sorted(set(candidate_occupancy_keys) - set(reference_occupancy_keys))
            raise CheckpointValidationError(
                f"{candidate_path}: occupancy key mismatch; missing={missing!r}, extra={extra!r}"
            )
        candidate_duplicates = _validate_duplicate_occupancy(
            candidate_pipeline, candidate_occupancy_keys, candidate_path
        )
        candidate_step = _trainer_step(candidate_checkpoint, candidate_path)
        comparisons.append(
            {
                "candidate": {
                    "path": str(candidate_path),
                    "bytes": candidate_path.stat().st_size,
                    "trainer_step": candidate_step,
                    "load_mode": candidate_load_mode,
                    "occupancy_duplicate_equal": candidate_duplicates,
                },
                "trainer_step_delta": candidate_step - reference_step,
                "validation": {
                    "keys": "exact_match",
                    "shapes": "exact_match",
                    "dtypes": "exact_match",
                    "finite": True,
                },
                "groups": {
                    "field": _compare_group(
                        reference_pipeline,
                        candidate_pipeline,
                        FIELD_KEYS,
                        reference_path=reference_path,
                        candidate_path=candidate_path,
                        chunk_elements=chunk_elements,
                    ),
                    "occupancy": _compare_group(
                        reference_pipeline,
                        candidate_pipeline,
                        reference_occupancy_keys,
                        reference_path=reference_path,
                        candidate_path=candidate_path,
                        chunk_elements=chunk_elements,
                        include_suffix_aggregates=True,
                    ),
                },
            }
        )
        del candidate_pipeline, candidate_checkpoint
        gc.collect()

    return {
        "schema_version": 1,
        "metric": {
            "name": "symmetric_relative_l2",
            "formula": "2 * ||candidate - reference||_2 / (||reference||_2 + ||candidate||_2)",
            "both_zero_value": 0.0,
            "accumulation_dtype": "torch.float64",
        },
        "chunk_elements": chunk_elements,
        "reference": {
            "path": str(reference_path),
            "bytes": reference_path.stat().st_size,
            "trainer_step": reference_step,
            "load_mode": reference_load_mode,
            "occupancy_duplicate_equal": reference_duplicates,
        },
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    payload = compare_checkpoints(args.checkpoints, chunk_elements=args.chunk_elements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output.resolve())
    for comparison in payload["comparisons"]:
        print(
            f"candidate={comparison['candidate']['path']} "
            f"step_delta={comparison['trainer_step_delta']} "
            f"field={comparison['groups']['field']['aggregate']['symmetric_relative_l2']:.9g} "
            f"occupancy_occs="
            f"{comparison['groups']['occupancy']['by_suffix']['occs']['aggregate']['symmetric_relative_l2']:.9g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
