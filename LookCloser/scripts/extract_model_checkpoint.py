#!/usr/bin/env python3
"""Extract the evaluation-required model state from a full trainer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metric-step", type=int, default=None)
    parser.add_argument("--selection", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False, mmap=True)
    compact = {key: checkpoint[key] for key in ("step", "pipeline")}
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(compact, temporary)
    temporary.replace(args.output)
    sidecar = {
        "source": str(args.source),
        "source_step": int(checkpoint["step"]),
        "metric_step": args.metric_step,
        "selection": args.selection,
        "keys": sorted(compact),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
