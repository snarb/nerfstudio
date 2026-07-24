#!/usr/bin/env python3
"""Expand a LookCloser TCNN hash checkpoint without changing its initial function.

For a power-of-two hash table, increasing ``log2_hashmap_size`` changes the
modulus used by the hash lookup.  Copying an old table into only the first half
of a larger table is therefore incorrect.  Each saturated source-level table
must be repeated across every target-sized modulus partition.  This preserves
the source encoding exactly at conversion time while allowing the repeated
entries to diverge during subsequent optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence

import torch
from torch import Tensor


ENCODING_KEY = "_model.field.encoding.params"
FIELDS_OPTIMIZER = "fields"
TCNN_ROW_ALIGNMENT = 8


def level_resolutions(*, num_levels: int, min_res: float, max_res: float) -> List[int]:
    """Return the per-level TCNN grid resolutions used by LookCloser."""

    if num_levels < 2:
        raise ValueError("num_levels must be at least 2")
    if not (0 < min_res < max_res):
        raise ValueError("expected 0 < min_res < max_res")
    scale = math.exp(math.log(max_res / min_res) / (num_levels - 1))
    return [math.ceil(min_res * (scale**level)) for level in range(num_levels)]


def aligned_level_rows(
    *,
    log2_hashmap_size: int,
    num_levels: int,
    min_res: float,
    max_res: float,
    alignment: int = TCNN_ROW_ALIGNMENT,
) -> List[int]:
    """Return aligned hash rows per level, excluding the feature dimension."""

    if log2_hashmap_size <= 0:
        raise ValueError("log2_hashmap_size must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    cap = 1 << log2_hashmap_size
    rows = []
    for resolution in level_resolutions(
        num_levels=num_levels, min_res=min_res, max_res=max_res
    ):
        unaligned = min(cap, resolution**3)
        rows.append(((unaligned + alignment - 1) // alignment) * alignment)
    return rows


def expected_parameter_count(
    *,
    log2_hashmap_size: int,
    num_levels: int,
    min_res: float,
    max_res: float,
    features_per_level: int,
) -> int:
    if features_per_level <= 0:
        raise ValueError("features_per_level must be positive")
    return (
        sum(
            aligned_level_rows(
                log2_hashmap_size=log2_hashmap_size,
                num_levels=num_levels,
                min_res=min_res,
                max_res=max_res,
            )
        )
        * features_per_level
    )


def expand_hash_tensor(
    tensor: Tensor,
    *,
    source_log2: int,
    target_log2: int,
    num_levels: int,
    min_res: float,
    max_res: float,
    features_per_level: int,
) -> Tensor:
    """Expand one flat TCNN hash tensor with modulus-preserving repetition."""

    if tensor.ndim != 1:
        raise ValueError(f"expected a flat hash tensor, got shape {tuple(tensor.shape)}")
    if target_log2 < source_log2:
        raise ValueError("target_log2 must be greater than or equal to source_log2")
    source_rows = aligned_level_rows(
        log2_hashmap_size=source_log2,
        num_levels=num_levels,
        min_res=min_res,
        max_res=max_res,
    )
    target_rows = aligned_level_rows(
        log2_hashmap_size=target_log2,
        num_levels=num_levels,
        min_res=min_res,
        max_res=max_res,
    )
    expected_source = sum(source_rows) * features_per_level
    if tensor.numel() != expected_source:
        raise ValueError(
            f"source tensor has {tensor.numel()} values; expected {expected_source}"
        )

    expanded: List[Tensor] = []
    offset = 0
    for level, (old_rows, new_rows) in enumerate(zip(source_rows, target_rows)):
        count = old_rows * features_per_level
        table = tensor[offset : offset + count].reshape(old_rows, features_per_level)
        offset += count
        if new_rows % old_rows != 0:
            raise ValueError(
                f"level {level} target rows {new_rows} are not a multiple of source rows {old_rows}"
            )
        repeat_factor = new_rows // old_rows
        expanded.append(table.repeat((repeat_factor, 1)).reshape(-1))
    if offset != tensor.numel():
        raise AssertionError("hash tensor split did not consume the full source tensor")
    return torch.cat(expanded)


def _expand_optimizer_moments(
    optimizers: MutableMapping[str, Any],
    *,
    source_count: int,
    expansion_kwargs: Dict[str, Any],
) -> List[str]:
    fields = optimizers.get(FIELDS_OPTIMIZER)
    if not isinstance(fields, MutableMapping):
        raise ValueError("checkpoint lacks the fields optimizer state")
    state = fields.get("state")
    if not isinstance(state, MutableMapping):
        raise ValueError("fields optimizer lacks a state mapping")

    expanded_names: List[str] = []
    matching_states = []
    for parameter_id, parameter_state in state.items():
        if not isinstance(parameter_state, MutableMapping):
            continue
        exp_avg = parameter_state.get("exp_avg")
        if isinstance(exp_avg, Tensor) and exp_avg.ndim == 1 and exp_avg.numel() == source_count:
            matching_states.append((parameter_id, parameter_state))
    if len(matching_states) != 1:
        raise ValueError(
            "expected exactly one fields optimizer parameter matching the hash encoding, "
            f"found {len(matching_states)}"
        )

    parameter_id, parameter_state = matching_states[0]
    for moment_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        moment = parameter_state.get(moment_name)
        if moment is None:
            continue
        if not isinstance(moment, Tensor) or moment.ndim != 1 or moment.numel() != source_count:
            raise ValueError(
                f"optimizer state {parameter_id}.{moment_name} does not match the encoding shape"
            )
        parameter_state[moment_name] = expand_hash_tensor(moment, **expansion_kwargs)
        expanded_names.append(f"{parameter_id}.{moment_name}")
    return expanded_names


def expand_checkpoint_state(
    checkpoint: MutableMapping[str, Any],
    *,
    source_log2: int,
    target_log2: int,
    num_levels: int,
    min_res: float,
    max_res: float,
    features_per_level: int,
) -> Dict[str, Any]:
    """Mutate an in-memory Nerfstudio checkpoint and return an audit record."""

    pipeline = checkpoint.get("pipeline")
    if not isinstance(pipeline, MutableMapping):
        raise ValueError("checkpoint lacks a pipeline state mapping")
    encoding = pipeline.get(ENCODING_KEY)
    if not isinstance(encoding, Tensor):
        raise ValueError(f"checkpoint lacks tensor {ENCODING_KEY!r}")

    expansion_kwargs = {
        "source_log2": source_log2,
        "target_log2": target_log2,
        "num_levels": num_levels,
        "min_res": min_res,
        "max_res": max_res,
        "features_per_level": features_per_level,
    }
    source_count = encoding.numel()
    pipeline[ENCODING_KEY] = expand_hash_tensor(encoding, **expansion_kwargs)
    target_count = pipeline[ENCODING_KEY].numel()

    optimizers = checkpoint.get("optimizers")
    if not isinstance(optimizers, MutableMapping):
        raise ValueError("checkpoint lacks optimizer state")
    expanded_moments = _expand_optimizer_moments(
        optimizers,
        source_count=source_count,
        expansion_kwargs=expansion_kwargs,
    )
    audit = {
        "algorithm": "per_level_modulus_partition_repeat",
        "encoding_key": ENCODING_KEY,
        "source_log2_hashmap_size": source_log2,
        "target_log2_hashmap_size": target_log2,
        "num_levels": num_levels,
        "min_res": min_res,
        "max_res": max_res,
        "features_per_level": features_per_level,
        "source_parameter_count": source_count,
        "target_parameter_count": target_count,
        "expanded_optimizer_moments": expanded_moments,
    }
    checkpoint["lookcloser_hash_expansion"] = audit
    return audit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--source-log2", type=int, default=23)
    parser.add_argument("--target-log2", type=int, default=24)
    parser.add_argument("--num-levels", type=int, default=16)
    parser.add_argument("--min-res", type=float, default=16.0)
    parser.add_argument("--max-res", type=float, default=8192.0)
    parser.add_argument("--features-per-level", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source = args.source_checkpoint.resolve()
    output = args.output_checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if source == output:
        raise ValueError("source and output checkpoints must differ")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    audit = expand_checkpoint_state(
        checkpoint,
        source_log2=args.source_log2,
        target_log2=args.target_log2,
        num_levels=args.num_levels,
        min_res=args.min_res,
        max_res=args.max_res,
        features_per_level=args.features_per_level,
    )
    audit["source_checkpoint"] = str(source)
    audit["source_sha256"] = sha256_file(source)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"wrote={output}")
    print(f"sha256={sha256_file(output)}")
    print(f"audit={audit}")


if __name__ == "__main__":
    main()
