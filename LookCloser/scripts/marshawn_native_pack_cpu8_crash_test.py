"""Actual canonical packing crash/restart and wrong-source adoption regression."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_for(predicate, timeout=60):
    until = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= until:
            raise TimeoutError("Canonical packing crash regression timed out")
        time.sleep(.05)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--helper-directory", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--action", default="test")
    a = p.parse_args()
    sys.path[:0] = [str(a.helper_directory.parent), str(a.campaign / "operations/sam3_assist/code")]
    helper = __import__(a.helper_directory.name + ".marshawn_native_pack_cpu8", fromlist=["pack_records"])
    import marshawn_sam3_assist as io
    release = io.read(a.campaign / "operations/sam3_assist/v5/native_future_release_v7.json")
    baseline = a.campaign / "operations/sam3_assist/v5/frame001695"
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = {name: sha(a.campaign / "operations/sam3_assist/code" / name)
               for name in ("marshawn_sam3_assist.py", "marshawn_sam3_assist_v5.py", "marshawn_sam3_union_capture_v5.py")}
    item = dict(campaign=str(a.snapshot), campaign_identity=str(a.campaign), frame=1695,
                camera="AA", task_root=str(root / "transaction"), logical_root=str(baseline),
                release=release, adapter_sources_sha256=sources)
    if a.action == "crash_child":
        original = io.write

        def observe(path, value):
            original(path, value)
            if Path(path).name == "task.json" and "v5_adapter_sha256" not in value:
                (root / "stopped_after_base_manifest.json").write_text(json.dumps(dict(path=str(path), pid=os.getpid())))
                os.kill(os.getpid(), signal.SIGSTOP)

        io.write = observe
        helper._pack_camera(item)
        raise AssertionError("Expected canonical packing observer to stop child")
    argv = [sys.executable, str(Path(__file__).resolve()), "--campaign", str(a.campaign),
            "--snapshot", str(a.snapshot), "--helper-directory", str(a.helper_directory),
            "--root", str(root), "--action", "crash_child"]
    child = subprocess.Popen(argv, start_new_session=True)
    wait_for(lambda: (root / "stopped_after_base_manifest.json").exists())
    orphan = Path(io.read(root / "stopped_after_base_manifest.json")["path"]).parent
    if (root / "transaction/tasks/AA").exists():
        raise AssertionError("Incomplete canonical base task became externally visible")
    os.kill(child.pid, signal.SIGKILL)
    child.wait(timeout=5)
    orphan_hashes = {p.name: sha(p) for p in orphan.iterdir() if p.is_file() and not p.is_symlink()}
    rows, execution = helper.pack_records(a.snapshot, 1695, ["AA"], root / "transaction", release,
                                         workers=1, campaign_identity=a.campaign, logical_root=baseline)
    names = ("task.json", "reference.npz", "source_original_receipt.json", "source_portable_receipt.json",
             "seed_policy_v5.json", "union_resolver.py")
    equality = {folder: {name: sha(root / "transaction" / folder / "AA" / name) ==
                              sha(baseline / folder / "AA" / name) for name in names}
                for folder in ("tasks", "tasks_local")}
    if not all(all(row.values()) for row in equality.values()):
        raise AssertionError("Cold restarted pack differs from actual baseline task bytes")
    if orphan_hashes != {p.name: sha(p) for p in orphan.iterdir() if p.is_file() and not p.is_symlink()}:
        raise AssertionError("Interrupted bundle evidence changed")
    rejections = {}
    for label in ("wrong_frame", "wrong_reference"):
        badroot = root / label
        bad = badroot / "tasks/AA"
        shutil.copytree(root / "transaction/tasks/AA", bad, symlinks=True)
        spec = io.read(bad / "task.json")
        if label == "wrong_frame":
            spec["frame"] = 1690
        else:
            spec["reference"]["frame"] = 1685
        io.write(bad / "task.json", spec)
        held_digest = sha(bad / "task.json")
        try:
            helper.pack_records(a.snapshot, 1695, ["AA"], badroot, release, workers=1,
                                campaign_identity=a.campaign, logical_root=baseline)
        except ValueError as error:
            rejections[label] = str(error)
        else:
            raise AssertionError("Wrong-source packed input was adopted")
        if sha(bad / "task.json") != held_digest:
            raise AssertionError("Rejected packed task was overwritten")
    report = dict(passed=True, actual_original_pack_interrupted_after_base_task=True,
                  unfinished_bundle_not_published=True, killed_child_lock_released=True,
                  original_orphan_bytes_preserved=True, resumed_both_host_tasks_byte_equal=equality,
                  normal_namespace_import_and_spawn_API=True, source_adoption_rejections=rejections,
                  helper_sha256=sha(a.helper_directory / "marshawn_native_pack_cpu8.py"),
                  original_adapters_sha256=sources, source_or_GPU_mutated=False,
                  replay_seconds=execution["finished_at"] - execution["started_at"])
    io.write(root / "report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
