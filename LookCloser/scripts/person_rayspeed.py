#!/usr/bin/env python
"""Extract median 'Train Rays / Sec' from a nerfstudio train_stdout.log (ANSI-stripped)."""
import re
import sys
import statistics

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Matches the rightmost throughput token, e.g. "8.05 K", "82.10 K", "1.20 M", or a bare number.
TOK = re.compile(r"([0-9]+\.?[0-9]*)\s*([KM]?)\s*$")


def to_rays(num: float, suffix: str) -> float:
    return num * (1000.0 if suffix == "K" else 1_000_000.0 if suffix == "M" else 1.0)


def main(path: str) -> None:
    vals = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = ANSI.sub("", line).rstrip()
            # Throughput rows contain a percent-done marker and end with the rays/sec token.
            if "%" not in line:
                continue
            m = TOK.search(line)
            if not m:
                continue
            try:
                r = to_rays(float(m.group(1)), m.group(2))
            except ValueError:
                continue
            if r > 100:  # ignore tiny artifacts / ms values
                vals.append(r)
    if vals:
        vals = vals[2:] if len(vals) > 4 else vals  # drop warmup rows
        print(f"rays/sec median={statistics.median(vals):.0f} n={len(vals)} "
              f"min={min(vals):.0f} max={max(vals):.0f}")
    else:
        print("rays/sec n/a")


if __name__ == "__main__":
    main(sys.argv[1])
