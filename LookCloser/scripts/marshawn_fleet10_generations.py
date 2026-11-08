"""CPU-only generation routing over the qualified ten physical GPU fleet.

Scientific workers remain the sealed140k workers. Two known-invalid model
inputs receive new campaign roots/job names; all other frame outputs remain
at their original roots. No checkpoint/config relocation or warm transfer.
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

BASE_SHA = "b832673cf20b341c3d227373744b34ea3bc8d3077d61d21332901559b806bc44"
SEAL = "d873cd61ad5c00f9501f2f6c477cb42f962d4d15591f5785830e0aeae381ec14"
V = Path("/fsx/tmp/marshawn_video_1660_2015_24fps_20261107")
C = Path("/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107")
CG1 = Path("/fsx/tmp/marshawn_video_1660_2015_24fps_140k_gcrole_g1_20261108")
SOURCE_G1 = Path("/fsx/tmp/marshawn_video_source_gcrole_g1_20261108")
OLD_F = Path("/fsx/tmp/marshawn_fleet10_140k_20261107")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


BASE_FILE = Path(__file__).with_name("marshawn_fleet10_140k_base.py")
if sha(BASE_FILE) != BASE_SHA:
    raise ValueError("Qualified base fleet CPU source changed")
spec = importlib.util.spec_from_file_location("_qualified_fleet10_base", BASE_FILE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
original_configuration = base.configuration
original_reserve = base.reserve
original_modules = base.modules
base.__file__ = str(Path(__file__).resolve())  # Only watchdog/slot CLI locator.
_campaign_jobs = None


def frame_route(plan, frame):
    route = plan["frame_generations"][str(int(frame))]
    expected_generation = int(int(frame) in (1695, 1700))
    if route["model_generation"] != expected_generation:
        raise ValueError("Known-invalid model generation cannot fall back to old output")
    if route["source_generation"] != int(int(frame) >= 1695):
        raise ValueError("Wrong source generation")
    model = CG1 if expected_generation else C
    expected_roots = dict(olive=str(model), dev3=str(model),
                          silver="/cache/ubuntu/lookcloser/campaigns/" + model.name)
    source = SOURCE_G1 if int(frame) >= 1695 else V
    if route["model_campaign_by_host"] != expected_roots or route["source_campaign"] != str(source):
        raise ValueError("Generation roots cannot fall back to retired model/source inputs")
    return route


def modules(root):
    global _campaign_jobs
    fleet, campaign, stage = original_modules(root)
    if _campaign_jobs is None:
        _campaign_jobs = campaign.jobs
    campaign.eligible = eligible
    campaign.retry_failed = retry_failed
    return fleet, campaign, stage


def configuration(F, campaign, source):
    if Path(campaign) != C or Path(source) != V:
        raise ValueError("Generation migration requires the exact qualified parents")
    result = original_configuration(F, campaign, source)
    result.update(generation_router=1, base_controller_sha256=BASE_SHA,
                  controller_source_sha256=sha(__file__), legacy_controller_root=str(OLD_F),
                  assignments=str(V / "operations/fleet10_gcrole_g1/assignments"),
                  demands=str(V / "operations/fleet10_gcrole_g1/demands"),
                  source_generation_authorization=str(SOURCE_G1 / "operations/ROOT_source_generation_authorization.json"))
    routes = {}
    for frame in result["frames"]:
        model = CG1 if frame in (1695, 1700) else C
        source_root = SOURCE_G1 if frame >= 1695 else V
        routes[str(frame)] = dict(model_generation=int(frame in (1695, 1700)),
                                 source_generation=int(frame >= 1695), source_campaign=str(source_root),
                                 source_remote_campaign="/cache/ubuntu/lookcloser/campaigns/" + source_root.name,
                                 model_campaign_by_host={"olive": str(model), "dev3": str(model),
                                                        "silver": "/cache/ubuntu/lookcloser/campaigns/" + model.name})
    result["frame_generations"] = routes
    for host in result["hosts"].values():
        host["operational_controller"] = str(Path(host["root"]) / "operations/fleet10_generations/marshawn_fleet10_generations.py")
    return result


def guard(F):
    plan = read(F / "video_plan.json")
    if plan["controller_source_sha256"] != sha(__file__) or plan["base_controller_sha256"] != BASE_SHA:
        raise ValueError("Generation fleet source changed")
    if sha(C / "code_seal.json") != SEAL or sha(CG1 / "code_seal.json") != SEAL:
        raise ValueError("Generation neural code seal differs")
    for name in ("recipe.json", "pipeline.json", "factory_proof.json", "common_box.json"):
        if sha(CG1 / name) != sha(C / name):
            raise ValueError("Generation changed scientific recipe: " + name)
    if read(C / "recipe.json")["hard_max_iterations"] != 140000:
        raise ValueError("Hard140k required")
    if sha(V / "video_plan.json") != plan["baseline_plan_sha256"]:
        raise ValueError("Original source plan changed")
    if set(plan["hosts"]) != {"silver", "olive", "dev3"} or sum(len(h["gpus"]) for h in plan["hosts"].values()) != 10:
        raise ValueError("Exactly ten real physical slots required")
    for frame in plan["frames"]:
        frame_route(plan, frame)
    return plan, V


def variant(plan, frame):
    route = frame_route(plan, frame)
    result = copy.deepcopy(plan)
    for host_key, host in result["hosts"].items():
        host["root"] = route["model_campaign_by_host"][host_key]
        host["controller"] = str(Path(host["root"]) / "code/LookCloser/scripts/marshawn_tuning_controller.py")
    result["scientific_campaign"] = route["model_campaign_by_host"]["olive"]
    result["source_campaign"] = route["source_campaign"]
    return result


def jobs(plan, state, campaign):
    result = []
    for frame, row in state["frames"].items():
        if not row.get("host"):
            continue
        one = dict(state, frames={frame: row})
        generated = _campaign_jobs(variant(plan, frame), one)
        generated[0]["gpu"] = row["gpu"]
        result.extend(generated)
    return result


def source_authorization(plan, frame):
    per_frame = Path(plan["source_generation_authorization"]).parent / "ROOT_frame_source_authorizations" / ("%06d.json" % frame)
    if per_frame.exists():
        return per_frame, read(per_frame)
    if int(frame) in (1695, 1700):
        return None, {}
    path = Path(plan["source_generation_authorization"])
    return path, read(path, {})


def release_ready(plan, frame):
    route = frame_route(plan, frame)
    if not route["source_generation"]:
        return True
    path, release = source_authorization(plan, frame)
    allowed = int(frame) == release.get("frame") or int(frame) in release.get("frames", [])
    if not (release.get("authorized") is True and release.get("source_campaign") == str(SOURCE_G1)
            and release.get("source_generation") == 1
            and release.get("model_code_seal_sha256", release.get("code_seal_sha256")) == SEAL and allowed):
        return False
    if path.name != "%06d.json" % frame:
        return True  # Immutable continuous policy; per-frame QA/dense still required below.
    root = Path(route["source_campaign"])
    ops = root / "operations/advanced_inputs" / ("%06d" % frame)
    marker = root / "frames" / ("%06d" % frame) / "native_semantic_ready.json"
    ready = marker.parent / "data/ready.json"
    dense = ops / "dense_guard_before_dispatch.json"
    safe = read(ops / "safe_gate_before_dispatch.json", {})
    return (marker.exists() and ready.exists() and dense.exists()
            and release.get("source_marker_sha256") == sha(marker)
            and release.get("final_ready_sha256") == sha(ready)
            and release.get("gate_manifest_sha256") == safe.get("gate_manifest_sha256")
            and release.get("dense_guard_sha256") == sha(dense))


def eligible(plan, state, job):
    frame = next(int(f) for f, row in state["frames"].items() if row["name"] == job["name"])
    row = state["frames"][str(frame)]
    if state["jobs"][row["name"]]["status"] in ("complete", "held_input_generation"):
        return False
    route = frame_route(plan, frame)
    if not release_ready(plan, frame):
        return False
    local = Path(route["model_campaign_by_host"]["olive"])
    marker = read(local / "frames" / ("%06d" % frame) / "dispatch_ready.json", {})
    if any(marker.get(k) != route[k] for k in ("model_generation", "source_generation")):
        return False
    if marker.get("frame") != frame or marker.get("host") != job["host"]:
        return False
    if marker.get("host_root") != route["model_campaign_by_host"][job["host"]]:
        return False
    release = read(local / "release.json", {})
    return release.get("production_enabled") is True and release.get("hard_max_iterations") == 140000


def retry_failed(plan, state, now=None):
    now = time.time() if now is None else now
    changed = []
    for frame, row in state["frames"].items():
        current = state["jobs"][row["name"]]
        if current["status"] != "failed":
            continue
        snapshot = state.get("host_snapshots", {}).get(row.get("host"), {})
        remote = snapshot.get("jobs", {}).get(row["name"])
        if "error" in snapshot or not remote or remote.get("supervisor_alive") or remote.get("worker_alive"):
            continue
        if row["attempt"] + 1 >= plan["max_attempts"]:
            continue
        delay = min(1800, plan.get("retry_base_seconds", 60) * 2 ** row["attempt"])
        if now < current.get("finished", now) + delay:
            continue
        row["attempt"] += 1
        generation = frame_route(plan, frame)["model_generation"]
        row["name"] = ("frame%06d_g%02d_a%02d" % (int(frame), generation, row["attempt"])) if generation else ("frame%06d_a%02d" % (int(frame), row["attempt"]))
        state["jobs"][row["name"]] = dict(status="pending")
        changed.append(int(frame))
    return changed


def reserve(F, frame, source_proof=None, force_host=None, force_gpu=None, resume_authorization=None):
    plan, _ = guard(F)
    route = frame_route(plan, frame)
    if resume_authorization is not None:
        raise ValueError("Generation router never imports legacy model state")
    if not release_ready(plan, frame):
        return dict(status="held_source_generation_not_released", frame=frame)
    source = Path(route["source_campaign"])
    ops = source / "operations/advanced_inputs" / ("%06d" % frame)
    safe = read(source_proof or ops / "safe_gate_before_dispatch.json", {})
    dense = read(ops / "dense_guard_before_dispatch.json", {})
    data = source / "frames" / ("%06d" % frame) / "data"
    marker = data.parent / "native_semantic_ready.json"
    if (not marker.exists() or safe.get("source_marker_sha256") != sha(marker)
            or dense.get("ready_sha256") != safe.get("final_ready_sha256")
            or dense.get("manifest_sha256") != safe.get("gate_manifest_sha256")
            or dense.get("dense_all66_passed") is not True):
        return dict(status="awaiting_current_generation_source_grid_dense", frame=frame)
    fleet, _, _ = modules(V)
    previous_guard, original_atomic = base.guard, fleet.atomic
    assignment_path = Path(plan["assignments"]) / ("%06d.json" % frame)
    authorization_path, _ = source_authorization(plan, frame)
    annotations = dict(model_generation=route["model_generation"], source_generation=route["source_generation"],
                       source_root=str(source), source_data_root=str(data),
                       source_generation_authorization=str(authorization_path),
                       source_generation_authorization_sha256=sha(authorization_path))
    def annotated_atomic(path, value):
        if Path(path) == assignment_path:
            value = dict(value, **annotations)
        return original_atomic(path, value)
    base.guard = lambda root: (variant(plan, frame), source)
    fleet.atomic = annotated_atomic
    try:
        result = original_reserve(F, frame, source_proof, force_host, force_gpu, None)
    finally:
        base.guard = previous_guard
        fleet.atomic = original_atomic
    if result.get("status") == "reserved":
        path = Path(result["assignment_path"])
        result["assignment"] = read(path)
        if any(result["assignment"].get(k) != value for k, value in annotations.items()):
            raise ValueError("Existing assignment belongs to a different input generation")
    return result


def qualify_frame_permissions(plan, model_root, frame, host_key):
    if host_key != "dev3":
        return None
    # The GPU supervisor's job ledger stays in C; only model output is CG1.
    root = Path(model_root) / "frames" / ("%06d" % frame)
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["setfacl", "-R", "-P", "-m", "u:1000:rwX", str(root)], check=True, capture_output=True)
    for directory in [root] + [p for p in root.rglob("*") if p.is_dir() and not p.is_symlink()]:
        subprocess.run(["setfacl", "-m", "d:u:1000:rwx,d:u:1001:rwx", str(directory)], check=True, capture_output=True)
    generation = frame_route(plan, frame)["model_generation"]
    for attempt in range(plan["max_attempts"]):
        name = ("frame%06d_g%02d_a%02d" % (frame, generation, attempt)) if generation else ("frame%06d_a%02d" % (frame, attempt))
        directory = C / "jobs" / name
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(["setfacl", "-m", "u:1000:rwx,d:u:1000:rwx,d:u:1001:rwx", str(directory)], check=True, capture_output=True)
    proof = root / ".fleet10_dev3_write_qualification.json"
    code = ("import pathlib,json,os;p=pathlib.Path(%r);"
            "p.write_text(json.dumps(dict(uid=os.getuid(),frame=%d,generation=%d,scope='assignedframeonly'))+'\\n');print(p.read_text())") % (str(proof), frame, generation)
    host = plan["hosts"]["dev3"]
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host["address"],
                             base.shlex.join([host["python"], "-c", code])], check=True, capture_output=True, text=True, timeout=30)
    record = json.loads(result.stdout)
    if record.get("uid") != 1000 or record != read(proof):
        raise ValueError("Generation DEV3 frame write qualification failed")
    return dict(receipt=str(proof), sha256=sha(proof), uid=1000)


base.modules = modules
base.configuration = configuration
base.guard = guard
base.jobs = jobs
base.reserve = reserve
base.qualify_frame_permissions = qualify_frame_permissions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "preflight", "reserve", "launch", "_dispatch", "_watchdog", "_slots"])
    p.add_argument("--root", type=Path)
    p.add_argument("--campaign", type=Path, default=C)
    p.add_argument("--source-campaign", type=Path, default=V)
    p.add_argument("--frame", type=int)
    p.add_argument("--host-root", type=Path)
    p.add_argument("--expected-source-sha")
    p.add_argument("--source-proof", type=Path)
    p.add_argument("--force-host")
    p.add_argument("--force-gpu", type=int)
    a = p.parse_args()
    if a.expected_source_sha and sha(__file__) != a.expected_source_sha:
        raise ValueError("Slot probe code changed")
    if a.action == "_slots":
        print(json.dumps(base.probe_slots(a.host_root)))
        return
    F = a.root.resolve()
    if a.action == "prepare":
        print(json.dumps(base.prepare(F, a.campaign, a.source_campaign)))
    elif a.action == "preflight":
        print(json.dumps(base.preflight(F)))
    elif a.action == "reserve":
        print(json.dumps(reserve(F, a.frame, a.source_proof, a.force_host, a.force_gpu)))
    else:
        authorization = read(F / "ROOT_cpu_authorization.json", {})
        if authorization.get("authorized") is not True or authorization.get("controller_source_sha256") != sha(__file__) or authorization.get("plan_sha256") != sha(F / "video_plan.json"):
            raise ValueError("Versioned generation CPU controller not released")
        if a.action == "_dispatch":
            base.dispatch(F)
        elif a.action == "_watchdog":
            base.watchdog(F)
        else:
            guard(F)
            with (F / "watchdog.log").open("ab") as stream:
                child = subprocess.Popen([sys.executable, __file__, "_watchdog", "--root", str(F)], stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps(dict(watchdog_pid=child.pid)))


if __name__ == "__main__":
    main()
