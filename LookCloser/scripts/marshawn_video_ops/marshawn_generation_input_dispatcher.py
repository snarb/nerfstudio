"""Generation-aware I/O wrapper over unchanged all66 source/grid transactions.

The old1675 corrected-grid canary remains startup evidence. Every NEW source
revision still needs its own ready/grid/sparse/dense proofs before reservation.
No model, grid classifier, mask, renderer, or GPU scheduling math is changed.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(config):
    c = read(config)
    if c.get("authorized") is not True:
        raise ValueError("Generation source dispatcher not released")
    for path, digest in c["generation_files_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Generation I/O helper changed: " + path)
    legacy = Path(c["legacy_operations"])
    sys.path[:0] = [str(legacy), str(Path(c["root"]) / "code"), str(Path(c["root"]) / "code/LookCloser/scripts")]
    dense = load_module("original_dense_source_guard", legacy / "advanced_v4_dense_guarded_launcher.py")
    startup, transaction = dense.load(Path(c["legacy_startup_config"]))
    # All original code, canary and source-seal guards remain active. Only I/O
    # roots/namespace/physical-host model destinations come from this release.
    combined = dict(startup, **c)
    combined["files_sha256"] = dict(startup["files_sha256"], **c["generation_files_sha256"])
    transaction.guard(combined)
    original_read = transaction.read
    inventory = Path(c["root"]) / "source_inventory.json"
    def scoped_read(path):
        result = original_read(path)
        if Path(path) == inventory:
            result = dict(result, frames=[row for row in result["frames"] if row["frame"] in c["frame_scope"]])
        return result
    transaction.read = scoped_read
    original_advance = transaction.advance
    def generation_advance(caller, frame):
        plan = read(Path(caller["fleet10_root"]) / "video_plan.json")
        mapping = plan["frame_generations"][str(frame)]
        if mapping["source_campaign"] != caller["root"]:
            raise ValueError("Generation source differs before original transaction")
        local = copy.deepcopy(caller)
        local["production_root"] = mapping["model_campaign_by_host"]["olive"]
        marker = Path(local["production_root"]) / "frames" / ("%06d" % frame) / "dispatch_ready.json"
        if marker.exists():
            published = read(marker)
            assignment_path = Path(plan["assignments"]) / ("%06d.json" % frame)
            assignment = read(assignment_path)
            expected = dict(frame=frame, model_generation=mapping["model_generation"], source_generation=mapping["source_generation"],
                            host=assignment["host"], host_root=mapping["model_campaign_by_host"][assignment["host"]],
                            remote_root=mapping["model_campaign_by_host"][assignment["host"]], assignment_sha256=sha(assignment_path),
                            assignment_nonce=assignment["assignment_nonce"], input_manifest_sha256=assignment["input_manifest_sha256"],
                            source_ready_sha256=assignment["data_ready_sha256"], source_root=caller["root"], hard_max_iterations=140000)
            if any(published.get(k) != value for k, value in expected.items()):
                raise ValueError("Existing dispatch marker belongs to a retired generation/root/source/assignment")
        return original_advance(local, frame)
    transaction.advance = generation_advance
    route = load_module("advanced_cap_host_routing", Path(c["generation_route_module"]))
    sys.modules["advanced_cap_host_routing"] = route
    dense.install(combined, transaction)
    guarded = route.publish

    def per_frame(caller, frame, gate, stage, read_, write_, sha_):
        plan = read(Path(caller["fleet10_root"]) / "video_plan.json")
        mapping = plan["frame_generations"][str(frame)]
        if mapping["source_campaign"] != caller["root"]:
            raise ValueError("Wrong source generation before dense proof")
        local = copy.deepcopy(caller)
        local["production_root"] = mapping["model_campaign_by_host"]["olive"]
        return guarded(local, frame, gate, stage, read_, write_, sha_)

    route.publish = per_frame
    return combined, transaction


def watchdog(c, transaction, config):
    held = transaction.lock(Path(c["operations"]) / "watchdog.lock")
    if held is None:
        return
    transaction.write(Path(c["operations"]) / "watchdog_identity.json", transaction.identity())
    while not (Path(c["operations"]) / "complete.json").exists():
        owner = Path(c["operations"]) / "controller_identity.json"
        previous = read(owner) if owner.exists() else None
        if not transaction.alive(previous):
            with (Path(c["operations"]) / "controller.log").open("ab") as log:
                subprocess.Popen([sys.executable, __file__, "_daemon", "--config", str(config)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        time.sleep(20)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["check", "launch", "_daemon", "_watchdog"])
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    c, transaction = load(a.config)
    if a.action == "check":
        print(json.dumps(dict(valid=True, generation_routing=True, dense_all66_required=True,
                              no_model_or_GPU_started=True)))
    elif a.action == "_daemon":
        transaction.daemon(c, a.config)
    elif a.action == "_watchdog":
        watchdog(c, transaction, a.config)
    else:
        with (Path(c["operations"]) / "watchdog.log").open("ab") as log:
            child = subprocess.Popen([sys.executable, __file__, "_watchdog", "--config", str(a.config)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps(dict(watchdog_pid=child.pid)))
