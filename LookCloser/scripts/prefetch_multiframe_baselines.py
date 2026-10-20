"""Native baseline coverage for every verified frozen frame, without training.

Uses the same campaign-only GPU lease and memory guard as transfer inference.
Heldout measurements remain evaluation-only; this script builds no hybrid.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(root, chunk):
    scripts = Path(__file__).resolve().parent
    inventory = json.loads((root / "preflight/completed_frames.json").read_text())
    pending = [
        item for item in inventory["frames"]
        if not (root / "frames" / f'{item["frame"]:06d}' /
                "evaluation/baseline_metrics.json").exists()
    ]
    plan = dict(created=time.time(), frames=[i["frame"] for i in pending],
                purpose="Native frozen baseline coverage; not hybrid acceptance")
    (root / "preflight/native_baseline_queue.json").write_text(
        json.dumps(plan, indent=2) + "\n")
    for item in pending:
        frame = item["frame"]
        directory = root / "frames" / f"{frame:06d}"
        context = directory / "context.json"
        assert context.exists()
        env = {
            **os.environ, "LUSTER_HYBRID_FRAME_MANIFEST": str(context),
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_HOME": str(Path(sys.executable).parents[1]),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        }

        def command(stage, args, gpu=False):
            if shutil.disk_usage(root).free < 768 * 1024**3:
                raise RuntimeError("Preserve unrelated training disk reserve")
            print(json.dumps(dict(frame=frame, stage=stage, time=time.time())), flush=True)
            with (root / "logs" / f"{frame:06d}_baseline_queue_{stage}.log").open("a") as log:
                subprocess.run(
                    [sys.executable, str(scripts / args[0]), *args[1:]],
                    env={**env, "CUDA_VISIBLE_DEVICES": "0" if gpu else ""},
                    cwd=scripts.parents[1], stdout=log, stderr=subprocess.STDOUT,
                    check=True,
                )

        if not (directory / "preflight/manifest.json").exists():
            command("prepare", ["luster_hybrid_common.py", "--root", str(directory)])
        command("targets", [
            "guarded_transfer_inference.py", "--root", str(directory),
            "--cameras", "7", "12", "95", "150", "--occupancy-only",
            "--chunk", str(chunk),
        ], gpu=True)
        command("protocol", ["protocol_multiframe_transfer.py", "--root", str(directory)])
        if not (directory / "evaluation/baseline_metrics.json").exists():
            command("metrics", ["evaluate_luster_hybrid.py", "--root", str(directory),
                                "--device", "cpu"])
        print(json.dumps(dict(frame=frame, stage="baseline_complete", time=time.time())),
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--chunk", type=int, choices=[256, 512, 1024, 2048, 4096],
                        default=4096)
    args = parser.parse_args()
    run(args.root, args.chunk)
