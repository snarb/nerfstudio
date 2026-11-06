"""Serial, restartable HD/4K native continuations with cache release at every gate."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".%d.tmp" % os.getpid())
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_files(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError("Bound file changed: " + path)


def select_best(evaluations):
    minimum = min(row["metrics"]["lpips"] for row in evaluations)
    return min((row for row in evaluations if row["metrics"]["lpips"] <= minimum + .001),
               key=lambda row: (-row["metrics"]["psnr"], row["step"]))


def initial_state(release, release_sha):
    from marshawn_tuning_metrics import PlateauTracker
    baseline = release["baseline"]
    assert baseline["step"] == release["anchor_step"]
    tracker = PlateauTracker(anchor_step=release["anchor_step"], interval=release["interval"],
                             patience=4, loss_relative_delta=.001, lpips_absolute_delta=.001)
    tracker.observe(baseline["step"], baseline["metrics"]["validation_loss"], baseline["metrics"]["lpips"])
    arm = dict(evaluations=[baseline], selection=baseline, plateau=tracker.to_dict(),
               cursor=release["source"])
    return dict(schema_version=1, release_sha256=release_sha,
                arms={name: copy.deepcopy(arm) for name in release["arm_order"]}, pending=None)


def next_arm(state, order):
    active = [name for name in order if not state["arms"][name]["plateau"]["stopped"]]
    return min(active, key=lambda name: (state["arms"][name]["plateau"]["last_step"], order.index(name))) if active else None


def commit_gate(state, arm_name, evaluation, cursor):
    from marshawn_tuning_metrics import PlateauTracker
    result = copy.deepcopy(state)
    arm = result["arms"][arm_name]
    existing = next((row for row in arm["evaluations"] if row["step"] == evaluation["step"]), None)
    if existing is not None:
        if existing != evaluation:
            raise ValueError("Conflicting gate replay")
        return result
    tracker = PlateauTracker.from_dict(arm["plateau"])
    tracker.observe(evaluation["step"], evaluation["metrics"]["validation_loss"], evaluation["metrics"]["lpips"])
    arm["evaluations"].append(evaluation)
    arm["selection"] = select_best(arm["evaluations"])
    arm["plateau"] = tracker.to_dict()
    arm["cursor"] = cursor
    result["pending"] = None
    return result


def matched_metrics(state):
    tables = {name: {row["step"]: row for row in arm["evaluations"]} for name, arm in state["arms"].items()}
    common = sorted(set.intersection(*(set(table) for table in tables.values())))
    return dict(common_steps=common,
                rows=[dict(step=step, arms={name: {key: table[step]["metrics"][key] for key in ["psnr", "ssim", "lpips"]}
                                           for name, table in tables.items()}) for step in common],
                unmatched_steps={name: sorted(set(table)-set(common)) for name, table in tables.items()},
                limitation="Only common additional-step gates isolate the resolution factor; unmatched suffixes also differ in optimization time.")


def validate_release(path, restore_only=False):
    release = json.loads(Path(path).read_text())
    if not (release.get("production_launch_authorized") is True or
            (restore_only and release.get("restoration_probe_authorized") is True)):
        raise ValueError("Reviewed paired production or restoration-only release required")
    if release["anchor_step"] != 100000 or release["interval"] != 5000:
        raise ValueError("Unreviewed anchor or cadence")
    if release["arm_order"] != ["HD", "TRAIN4K"] or set(release["arms"]) != {"HD", "TRAIN4K"}:
        raise ValueError("Unreviewed paired arms")
    if [release["arms"][name]["train_scale"] for name in release["arm_order"]] != [1, 2]:
        raise ValueError("Resolution treatment changed")
    verify_files(release["bound_files"])
    return release


def memory():
    return {line.split(":")[0]: int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
            if line.split(":")[0] in ["MemAvailable", "MemTotal"]}


def gpu():
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True).strip()
    return {int(line.split(",")[0]): int(line.split(",")[1]) for line in text.splitlines() if line}


def materialize_config(release, release_path, name, output):
    import yaml
    from yaml.nodes import ScalarNode, SequenceNode
    arm = release["arms"][name]
    root = yaml.compose(Path(arm["config_template"]).read_text())
    def get(node, key):
        return next(value for field, value in node.value if field.value == key)
    def put(node, key, value):
        for index, (field, prior) in enumerate(node.value):
            if field.value == key:
                node.value[index] = (field, value)
                return
        raise KeyError(key)
    def scalar(value):
        return ScalarNode("tag:yaml.org,2002:str", str(value))
    def pathnode(value):
        return SequenceNode("tag:yaml.org,2002:python/object/apply:pathlib.PosixPath", [scalar(value)])
    def write_once(path, content):
        if path.exists():
            if path.read_text() != content:
                raise ValueError("Materialized configuration changed: " + str(path))
        else:
            path.write_text(content)
    directory = output / "configs" / sha(release_path)
    directory.mkdir(parents=True, exist_ok=True)
    gate = json.loads(Path(arm["RGB_gate_template"]).read_text())
    assert gate["approved"] is False
    gate.update(approved=True, paired_root_release_sha256=sha(release_path),
                production_authorized=release["production_launch_authorized"],
                restoration_only=not release["production_launch_authorized"])
    gate_path = directory / (name + "_RGB_gate.json")
    write_once(gate_path, json.dumps(gate, indent=2, allow_nan=False) + "\n")
    pipeline = get(root, "pipeline")
    put(root, "load_checkpoint", pathnode(release["source"]["checkpoint"]))
    put(root, "output_dir", pathnode(output / name / "run"))
    put(root, "experiment_name", scalar("native_resolution_pair_" + name))
    put(root, "timestamp", scalar("from100000"))
    put(pipeline, "final_grid_artifact_dir", pathnode(output / name / "grids"))
    put(pipeline, "final_color_gate", scalar(gate_path))
    put(pipeline, "final_color_gate_sha256", scalar(sha(gate_path)))
    config_path = directory / (name + ".yml")
    write_once(config_path, yaml.serialize(root))
    return config_path


def make_job(release, release_path, arm, source, config, output, restore_only):
    source_step = source["step"]
    target = source_step if restore_only else source_step + release["interval"]
    bindings = {str(config): sha(config), source["checkpoint"]: source["checkpoint_sha256"],
                source["runtime"]: source["runtime_sha256"], source["grid"]: source["grid_sha256"],
                str(Path(source["grid"]).with_suffix(".json")): sha(Path(source["grid"]).with_suffix(".json"))}
    return dict(root_release=str(release_path), root_release_sha256=sha(release_path), arm=arm,
                train_scale=release["arms"][arm]["train_scale"], config=str(config),
                source_checkpoint=source["checkpoint"], source_checkpoint_sha256=source["checkpoint_sha256"],
                source_runtime=source["runtime"], source_grid=source["grid"], source_step=source_step,
                target_step=target, output=str(output), restore_only=restore_only, bound_inputs=bindings)


def verify_chunk(job_path):
    job = json.loads(Path(job_path).read_text())
    path = Path(job["output"]) / "receipt.json"
    receipt = json.loads(path.read_text())
    if receipt["job_sha256"] != sha(job_path) or receipt["arm"] != job["arm"]:
        raise ValueError("Chunk receipt belongs to a different job")
    for key, expected in [("source_step", job["source_step"]), ("final_step", job["target_step"]),
                          ("source_checkpoint_sha256", job["source_checkpoint_sha256"]),
                          ("train_scale", job["train_scale"]), ("restore_only", job["restore_only"])]:
        if receipt[key] != expected:
            raise ValueError("Chunk lineage mismatch: " + key)
    assert all(receipt["restored_domain_equality"].values()) and all(receipt["serialization_roundtrip"].values())
    assert receipt["attempts"] == job["target_step"] - job["source_step"]
    assert receipt["actual_updates"] == receipt["final_adam"] - receipt["initial_adam"]
    pointer = dict(step=receipt["final_step"], checkpoint=receipt["output_checkpoint"],
                   checkpoint_sha256=receipt["output_checkpoint_sha256"], runtime=receipt["output_runtime"],
                   runtime_sha256=receipt["output_runtime_sha256"], grid=receipt["output_grid"],
                   grid_sha256=receipt["output_grid_sha256"])
    verify_files({pointer["checkpoint"]: pointer["checkpoint_sha256"], pointer["runtime"]: pointer["runtime_sha256"],
                  pointer["grid"]: pointer["grid_sha256"]})
    metadata = json.loads(Path(pointer["grid"]).with_suffix(".json").read_text())
    assert metadata["checkpoint_sha256"] == pointer["checkpoint_sha256"] and metadata["grid_archive_sha256"] == pointer["grid_sha256"]
    return receipt, pointer


def evaluation_row(path, pointer, release):
    value = json.loads(Path(path).read_text())
    assert value["step"] == pointer["step"] and value["checkpoint_sha256"] == pointer["checkpoint_sha256"]
    assert value["ready_sha256"] == sha(release["ready"]) and value["box_sha256"] == sha(release["box"])
    assert value["v2_manifest_sha256"] == sha(release["v2_manifest"])
    assert value["no_training_or_model_mutation"] is True and set(value["v1"]["by_view"]) == {"DC", "HC", "KA"}
    return dict(step=value["step"], **value["v1"], checkpoint=pointer["checkpoint"],
                checkpoint_sha256=pointer["checkpoint_sha256"], renders=str(Path(path).parent / "renders"),
                evaluation=str(path), evaluation_sha256=sha(path))


def main(args):
    release = validate_release(args.release, args.restore_only)
    for path in reversed(release["python_path"]):
        sys.path.insert(0, path)
    if socket.gethostname() != release["hostname"] or Path(sys.executable).resolve() != Path(release["python"]).resolve():
        raise ValueError("Use the reviewed dev3 historical runtime")
    output = Path(release["output"])
    output.mkdir(parents=True, exist_ok=True)
    with Path(release["GPU_lock"]).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / "state.json"
        if args.resume != state_path.exists() and not args.restore_only:
            raise ValueError("Existing state requires --resume; fresh runs cannot use --resume")
        verify_files({row["checkpoint"]: row["checkpoint_sha256"] for row in release["preserved_HD_history"]})
        atomic(output / "launches" / (str(time.time_ns()) + ".json"),
               dict(pid=os.getpid(), time=time.time(), root_release_sha256=sha(args.release),
                    restore_only=args.restore_only, resume=args.resume, GPU_lock=release["GPU_lock"]))
    
        def interrupt(signum, frame):
            raise KeyboardInterrupt("Controller signal " + str(signum))
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
    
        def wait_resources(label):
            started = time.monotonic()
            while True:
                available, processes = memory(), gpu()
                if not processes and available["MemAvailable"] >= release["minimum_available_RAM_GiB"] * 2**30:
                    return
                atomic(output / "status.json", dict(status="waiting_resources", phase=label, host=available, GPU=processes, time=time.time()))
                if time.monotonic() - started > release["resource_wait_timeout_seconds"]:
                    raise RuntimeError("Resources not released before next serial phase")
                time.sleep(10)
    
        def run_child(command, phase, log_path, progress_path=None):
            wait_resources(phase)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            child = None
            try:
                with log_path.open("a") as stream:
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                             start_new_session=True, env=environment)
                    while True:
                        code = child.poll()
                        host, processes = memory(), gpu()
                        with log_path.open("rb") as log:
                            log.seek(max(0, log_path.stat().st_size - 65536))
                            tail = log.read().decode(errors="replace")
                        health = dict(time=time.time(), controller_pid=os.getpid(), phase=phase,
                                      worker_pid=child.pid, worker_returncode=code, host=host, GPU=processes,
                                      OOM_evidence="out of memory" in tail.lower())
                        if progress_path is not None and progress_path.exists():
                            health["progress"] = json.loads(progress_path.read_text())
                        with (output / "supervision.jsonl").open("a") as log:
                            log.write(json.dumps(health) + "\n")
                        atomic(output / "status.json", health)
                        if code is not None:
                            if code:
                                raise RuntimeError("Child failed: " + phase + " code=" + str(code))
                            break
                        if host["MemAvailable"] < release["abort_below_available_RAM_GiB"] * 2**30:
                            raise RuntimeError("RAM reserve breached")
                        time.sleep(10)
            finally:
                if child is not None and child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait()
            wait_resources(phase + "_cache_exit")
    
        configs = {name: materialize_config(release, args.release, name, output) for name in release["arm_order"]}
        jobs = output / "jobs"
        jobs.mkdir(exist_ok=True)
        # The same worker must first prove exact source restoration for both arms.
        # These zero-update checkpoints never become the production source cursor.
        if not state_path.exists() or args.restore_only:
            preflight = {}
            for name in release["arm_order"]:
                attempt = str(time.time_ns())
                folder = output / "preflight" / (name + "_" + attempt)
                job = make_job(release, args.release, name, release["source"], configs[name], folder, True)
                path = jobs / ("restore_" + name + "_" + attempt + ".json")
                atomic(path, job)
                run_child([sys.executable, release["worker"], "--job", str(path)], "restore_" + name,
                          output / "logs" / (path.stem + ".log"))
                receipt, pointer = verify_chunk(path)
                assert receipt["actual_updates"] == receipt["attempts"] == 0
                preflight[name] = dict(receipt=str(folder / "receipt.json"), sha256=sha(folder / "receipt.json"))
            atomic(output / "preflight" / ("complete_" + sha(args.release) + ".json"),
                   dict(root_release_sha256=sha(args.release), arms=preflight,
                        output_checkpoints_excluded_from_ancestry=True))
            if args.restore_only:
                return
        state = json.loads(state_path.read_text()) if state_path.exists() else initial_state(release, sha(args.release))
        if state["release_sha256"] != sha(args.release):
            raise ValueError("Resume release changed")
        atomic(state_path, state)
        while next_arm(state, release["arm_order"]) is not None or state["pending"] is not None:
            pending = state["pending"]
            if pending is None:
                name = next_arm(state, release["arm_order"])
                source = state["arms"][name]["cursor"]
                target = source["step"] + release["interval"]
                attempt = str(time.time_ns())
                folder = output / name / "chunks" / ("step-%09d_attempt-%s" % (target, attempt))
                job = make_job(release, args.release, name, source, configs[name], folder, False)
                path = jobs / (name + "_" + str(target) + "_" + attempt + ".json")
                atomic(path, job)
                pending = dict(phase="train", arm=name, target_step=target, job=str(path), job_sha256=sha(path))
                state["pending"] = pending
                atomic(state_path, state)
            name, path = pending["arm"], Path(pending["job"])
            if sha(path) != pending["job_sha256"]:
                raise ValueError("Pending job changed")
            job = json.loads(path.read_text())
            source_cursor = state["arms"][name]["cursor"]
            assert job["arm"] == name and job["source_step"] == source_cursor["step"]
            assert job["source_checkpoint_sha256"] == source_cursor["checkpoint_sha256"]
            assert job["target_step"] == pending["target_step"] == source_cursor["step"] + release["interval"]
            assert job["root_release_sha256"] == sha(args.release) and not job["restore_only"]
            if pending["phase"] == "train":
                receipt_path = Path(job["output"]) / "receipt.json"
                if not receipt_path.exists():
                    # Preserve incomplete attempts; replay from the last committed arm cursor.
                    if Path(job["output"]).exists():
                        attempt = str(time.time_ns())
                        job["output"] = str(Path(job["output"]).with_name("step-%09d_attempt-%s" % (job["target_step"], attempt)))
                        path = jobs / (name + "_" + str(job["target_step"]) + "_" + attempt + ".json")
                        atomic(path, job)
                        pending["job"] = str(path)
                        pending["job_sha256"] = sha(path)
                        atomic(state_path, state)
                    run_child([sys.executable, release["worker"], "--job", str(path)], name + "_train_" + str(job["target_step"]),
                              output / "logs" / (path.stem + ".log"), Path(job["output"]) / "progress.json")
                receipt, pointer = verify_chunk(path)
                pending.update(phase="evaluate", cursor=pointer, receipt=str(Path(job["output"]) / "receipt.json"),
                               receipt_sha256=sha(Path(job["output"]) / "receipt.json"))
                atomic(state_path, state)
            elif pending["phase"] == "evaluate":
                if sha(pending["receipt"]) != pending["receipt_sha256"]:
                    raise ValueError("Pending chunk receipt changed")
                receipt, verified_pointer = verify_chunk(path)
                if verified_pointer != pending["cursor"]:
                    raise ValueError("Pending evaluation cursor changed")
            else:
                raise ValueError("Unknown pending phase")
            pointer = pending["cursor"]
            evaluation_path = Path(pending["evaluation"]) if pending.get("evaluation") else None
            if evaluation_path is None or not evaluation_path.exists():
                folder = output / name / "evaluations" / ("step-%09d_attempt-%d" % (pointer["step"], time.time_ns()))
                pending["evaluation"] = str(folder / "evaluation.json")
                atomic(state_path, state)
                command = [sys.executable, release["evaluator"], "--config", str(configs[name]),
                           "--checkpoint", pointer["checkpoint"], "--checkpoint-sha256", pointer["checkpoint_sha256"],
                           "--output", str(folder), "--ready", release["ready"], "--box", release["box"],
                           "--v2-manifest", release["v2_manifest"], "--renderer", release["renderer"]]
                for path_value in release["python_path"]:
                    command += ["--python-path", path_value]
                run_child(command, name + "_evaluate_" + str(pointer["step"]),
                          output / "logs" / (name + "_evaluation_" + folder.name + ".log"))
                evaluation_path = folder / "evaluation.json"
            row = evaluation_row(evaluation_path, pointer, release)
            state = commit_gate(state, name, row, pointer)
            atomic(state_path, state)
            atomic(output / "matched_metrics.json", matched_metrics(state))
            atomic(output / "gate_complete.json", dict(arm=name, step=row["step"], state_sha256=sha(state_path),
                                                        stopped=state["arms"][name]["plateau"]["stopped"], time=time.time()))
        verify_files(release["bound_files"])
        verify_files({row["checkpoint"]: row["checkpoint_sha256"] for row in release["preserved_HD_history"]})
        atomic(output / "complete.json", dict(time=time.time(), source=release["source"],
                                               selections={name: arm["selection"] for name, arm in state["arms"].items()},
                                               plateau={name: arm["plateau"] for name, arm in state["arms"].items()},
                                               matched=matched_metrics(state), reason="both_independent_arm_plateaus"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restore-only", action="store_true")
    arguments = parser.parse_args()
    try:
        main(arguments)
    except BaseException as error:
        # Do not convert an interrupted/incomplete experiment into successful completion.
        try:
            release = json.loads(arguments.release.read_text())
            directory = Path(release["output"])
            if directory.exists():
                atomic(directory / ("failure-%d.json" % time.time_ns()),
                       dict(time=time.time(), error=repr(error), root_release_sha256=sha(arguments.release)))
        finally:
            raise
