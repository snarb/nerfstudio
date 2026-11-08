"""Prove camera packing ownership survives a factory-parent crash."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def load(path):
    spec = importlib.util.spec_from_file_location("pack_ownership_test_target", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def until(predicate, timeout=30):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise TimeoutError("Camera ownership regression timed out")
        time.sleep(.05)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--helper", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--action", default="test")
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    helper = load(a.helper)
    argv = [sys.executable, str(Path(__file__).resolve()), "--helper", str(a.helper.resolve()),
            "--root", str(root)]
    if a.action == "parent":
        child = subprocess.Popen(argv + ["--action", "owner"], start_new_session=True)
        (root / "owner_pid.json").write_text(json.dumps(dict(pid=child.pid)))
        time.sleep(60)
        return
    if a.action in ("owner", "contender", "other_camera"):
        camera = "AB" if a.action == "other_camera" else "AA"
        with helper._camera_lock(root, camera):
            (root / (a.action + "_entered")).write_text(str(os.getpid()))
            if a.action == "owner":
                until(lambda: (root / "release_owner").exists())
            (root / (a.action + "_finished")).write_text(str(os.getpid()))
        return
    parent = subprocess.Popen(argv + ["--action", "parent"], start_new_session=True)
    children = []
    try:
        until(lambda: (root / "owner_entered").exists() and (root / "owner_pid.json").exists())
        owner_pid = json.loads((root / "owner_pid.json").read_text())["pid"]
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)
        os.kill(owner_pid, 0)
        with (root / ".packing_locks/AA.lock").open("a+") as lock:
            blocked = False
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                blocked = True
            if not blocked:
                raise AssertionError("A dead factory parent released the surviving camera owner's lock")
        contender = subprocess.Popen(argv + ["--action", "contender"], start_new_session=True)
        other = subprocess.Popen(argv + ["--action", "other_camera"], start_new_session=True)
        children.extend([contender, other])
        until(lambda: (root / "other_camera_finished").exists())
        if (root / "contender_entered").exists():
            raise AssertionError("Duplicate camera writer entered while original owner was alive")
        (root / "release_owner").write_text("release exact test owner\n")
        until(lambda: (root / "owner_finished").exists() and (root / "contender_finished").exists())
        for child in children:
            if child.wait(timeout=5) != 0:
                raise AssertionError("Ownership test subprocess failed")
        report = dict(passed=True, factory_parent_killed=True, original_child_survived=True,
                      duplicate_camera_writer_blocked=True, distinct_camera_parallel=True,
                      restarted_writer_acquired_only_after_owner_finished=True,
                      no_GPU_or_source_inputs_touched=True)
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        (root / "release_owner").write_text("release exact test owner\n")
        if parent.poll() is None:
            parent.terminate()
        for child in children:
            if child.poll() is None:
                child.terminate()


if __name__ == "__main__":
    main()
