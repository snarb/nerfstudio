#!/usr/bin/env python3
"""Run three concurrent from-scratch leader-recipe campaigns on frame 007810."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/home/brans/repos/nerfstudio_007810_scratch3")
PYTHON = Path("/home/brans/repos/nerfstudio/.venv/bin/python")
CONTROLLER = REPO / "LookCloser/scripts/run_static_target_from_scratch.py"
DATA = Path("/home/brans/temporal_perframe_stride7_45f/007810")
OUTPUT = Path("/mnt/data/lookcloser_007810_from_scratch_seed_sweep")
BRANCH = "scratch-007810-seed-sweep"
SEEDS = (42, 43, 44)
PLATEAU_INTERVALS = 2
MAX_STEP = 303_760
PSNR_TIE_DB = 0.07
MONITOR_SECONDS = 60


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def campaign_name(seed: int) -> str:
    return f"007810_leader_recipe_seed{seed}"


def manifest_path(seed: int) -> Path:
    return OUTPUT / "campaigns" / campaign_name(seed) / "campaign.json"


def controller_command(seed: int, *, resume: bool) -> list[str]:
    command = [
        str(PYTHON),
        str(CONTROLLER),
        "--campaign-name",
        campaign_name(seed),
        "--frame",
        "007810",
        "--expected-branch",
        BRANCH,
        "--data",
        str(DATA),
        "--output-dir",
        str(OUTPUT),
        "--venv",
        str(PYTHON.parents[1]),
        "--variant",
        "canonical",
        "--seed",
        str(seed),
        "--poll-seconds",
        "15",
    ]
    if resume:
        command.extend(["--resume", "--tail-intervals", "1"])
    return command


def process_snapshot() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,stat=,etimes=,cmd="], text=True
    )
    rows = []
    for line in output.splitlines():
        if (
            "007810_leader_recipe_seed" not in line
            and "run_007810_scratch_seed_sweep.py" not in line
        ):
            continue
        fields = line.strip().split(None, 4)
        if len(fields) == 5:
            rows.append(
                {
                    "pid": int(fields[0]),
                    "ppid": int(fields[1]),
                    "stat": fields[2],
                    "elapsed_seconds": int(fields[3]),
                    "command": fields[4],
                }
            )
    return rows


def gpu_snapshot() -> list[dict[str, Any]]:
    query = (
        "index,name,memory.total,memory.used,memory.free,"
        "utilization.gpu,utilization.memory,temperature.gpu"
    )
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    keys = (
        "index",
        "name",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "gpu_util_percent",
        "memory_util_percent",
        "temperature_c",
    )
    return [
        dict(zip(keys, (value.strip() for value in line.split(","))))
        for line in output.splitlines()
        if line.strip()
    ]


def oom_evidence() -> list[str]:
    hits: list[str] = []
    if not OUTPUT.exists():
        return hits
    for path in OUTPUT.glob("campaigns/*/*.log"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines()[-300:]:
            lowered = line.lower()
            if "out of memory" in lowered or "cuda error" in lowered:
                hits.append(f"{path}: {line[-500:]}")
    return hits[-20:]


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def monitor_wave(
    wave: int,
    active: dict[int, tuple[subprocess.Popen[str], Any, Path]],
) -> dict[int, int]:
    supervision = OUTPUT / "supervision.jsonl"
    while True:
        statuses = {seed: process.poll() for seed, (process, _, _) in active.items()}
        snapshot = {
            "at": now(),
            "wave": wave,
            "controllers": {
                str(seed): {"pid": active[seed][0].pid, "returncode": status}
                for seed, status in statuses.items()
            },
            "processes": process_snapshot(),
            "gpu": gpu_snapshot(),
            "oom_evidence": oom_evidence(),
        }
        append_jsonl(supervision, snapshot)
        gpu = snapshot["gpu"][0]
        alive = sum(status is None for status in statuses.values())
        print(
            f"monitor at={snapshot['at']} wave={wave} alive={alive}/{len(active)} "
            f"gpu_used_mib={gpu['memory_used_mib']} gpu_util={gpu['gpu_util_percent']} "
            f"oom_hits={len(snapshot['oom_evidence'])}",
            flush=True,
        )
        if all(status is not None for status in statuses.values()):
            break
        time.sleep(MONITOR_SECONDS)
    results: dict[int, int] = {}
    for seed, (process, stream, _) in active.items():
        results[seed] = int(process.returncode)
        stream.close()
    return results


def run_wave(seeds: list[int], wave: int, *, resume: bool) -> dict[int, int]:
    active: dict[int, tuple[subprocess.Popen[str], Any, Path]] = {}
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    for seed in seeds:
        log_path = OUTPUT / "campaigns" / campaign_name(seed) / f"supervisor_wave_{wave}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("w", encoding="utf-8")
        command = controller_command(seed, resume=resume)
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active[seed] = (process, stream, log_path)
        print(
            f"launch wave={wave} seed={seed} pid={process.pid} resume={resume} "
            f"log={log_path}",
            flush=True,
        )
    return monitor_wave(wave, active)


def load_manifest(seed: int) -> dict[str, Any]:
    return json.loads(manifest_path(seed).read_text(encoding="utf-8"))


def progress(seed: int) -> dict[str, Any]:
    manifest = load_manifest(seed)
    candidates = list(manifest.get("candidates", {}).values())
    latest_step = max(int(row["step"]) for row in candidates)
    selected = manifest["selected"]
    return {
        "seed": seed,
        "latest_step": latest_step,
        "trailing_numeric_plateau_intervals": int(
            manifest["plateau"]["trailing_numeric_plateau_intervals"]
        ),
        "selected_step": int(selected["step"]),
        "psnr": float(selected["metrics"]["psnr"]),
        "ssim": float(selected["metrics"]["ssim"]),
        "lpips": float(selected["metrics"]["lpips"]),
    }


def select_global() -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        manifest = load_manifest(seed)
        for candidate in manifest["candidates"].values():
            rows.append(
                {
                    "seed": seed,
                    "step": int(candidate["step"]),
                    "psnr": float(candidate["metrics"]["psnr"]),
                    "ssim": float(candidate["metrics"]["ssim"]),
                    "lpips": float(candidate["metrics"]["lpips"]),
                    "checkpoint": candidate["checkpoint"],
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "eval_json": candidate["eval_json"],
                    "render_dir": candidate["render_dir"],
                    "roi_protocol": candidate["roi_protocol"],
                }
            )
    maximum_psnr = max(row["psnr"] for row in rows)
    tied = [row for row in rows if maximum_psnr - row["psnr"] <= PSNR_TIE_DB]
    selected = min(
        tied,
        key=lambda row: (row["lpips"], -row["psnr"], row["step"], row["seed"]),
    )
    payload = {
        "created_at": now(),
        "policy": (
            "maximum full-eval PSNR, then minimum LPIPS within inclusive "
            "0.07 dB of maximum, then PSNR, earliest step and seed"
        ),
        "maximum_psnr": maximum_psnr,
        "selected": selected,
        "campaign_best": [progress(seed) for seed in SEEDS],
    }
    path = OUTPUT / "selection_numeric.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT / "sweep_state.json"
    if any(manifest_path(seed).exists() for seed in SEEDS):
        raise RuntimeError(
            "A seed campaign already exists; this entrypoint only starts a new sweep"
        )
    state = {
        "started_at": now(),
        "repo": str(REPO),
        "branch": BRANCH,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "data": str(DATA),
        "output": str(OUTPUT),
        "seeds": list(SEEDS),
        "recipe": "canonical static leader, from scratch",
        "status": "running",
        "waves": [],
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    wave = 0
    active = list(SEEDS)
    while active:
        results = run_wave(active, wave, resume=wave > 0)
        if any(code not in (0, 2) for code in results.values()):
            state["status"] = "failed"
            state["failure"] = {"wave": wave, "returncodes": results}
            state_path.write_text(
                json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return 3
        rows = [progress(seed) for seed in active]
        state["waves"].append(
            {"wave": wave, "completed_at": now(), "returncodes": results, "progress": rows}
        )
        state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for row in rows:
            print(
                f"progress seed={row['seed']} latest={row['latest_step']} "
                f"plateau={row['trailing_numeric_plateau_intervals']} "
                f"best={row['selected_step']} psnr={row['psnr']:.6f} "
                f"ssim={row['ssim']:.6f} lpips={row['lpips']:.6f}",
                flush=True,
            )
        active = [
            row["seed"]
            for row in rows
            if row["trailing_numeric_plateau_intervals"] < PLATEAU_INTERVALS
            and row["latest_step"] < MAX_STEP
        ]
        wave += 1

    selection = select_global()
    state["status"] = "numeric_complete"
    state["completed_at"] = now()
    state["selection"] = selection
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = selection["selected"]
    print(
        f"selected seed={selected['seed']} step={selected['step']} "
        f"psnr={selected['psnr']:.6f} ssim={selected['ssim']:.6f} "
        f"lpips={selected['lpips']:.6f} renders={selected['render_dir']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
