#!/usr/bin/env python3
"""Run a fixed-batch or dynamic-point-budget speed candidate from the frozen leader recipe."""

from __future__ import annotations

import sys

import run_static_leader_e2e as leader


def has_option(argv: list[str], name: str) -> bool:
    return any(value == name or value.startswith(f"{name}=") for value in argv)


def main() -> int:
    argv = sys.argv[1:]
    has_batch_scale = has_option(argv, "--batch-scale")
    has_target_points = has_option(argv, "--target-points")
    if has_batch_scale == has_target_points:
        raise SystemExit("Specify exactly one of --batch-scale {1,2,4} or --target-points N")
    if "--no-speed-stop-at-accepted-boundary" in argv:
        raise SystemExit("The speed controller must stop at the point-normalized accepted boundary")
    if not has_option(argv, "--speed-stop-at-accepted-boundary"):
        argv.append("--speed-stop-at-accepted-boundary")
    if not has_option(argv, "--eval-num-rays-per-chunk"):
        argv.extend(["--eval-num-rays-per-chunk", "16384"])
    original = sys.argv
    try:
        sys.argv = [original[0], *argv]
        return leader.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
