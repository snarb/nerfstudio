"""Actual completed-output collection parent/child/transport crash regressions."""
from pathlib import Path
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time


def until(predicate, seconds=120):
    end = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > end:
            raise TimeoutError("Private async-collection regression timed out")
        time.sleep(.05)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--action", default="test")
    p.add_argument("--config", type=Path)
    a = p.parse_args()
    sys.path.insert(0, str(a.code))
    import marshawn_native_async_collect as helper
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = [sys.executable, str(Path(__file__).resolve()), "--code", str(a.code),
              "--base-config", str(a.base_config), "--root", str(root)]
    if a.action != "test":
        manager = helper.Collectors(a.config)
        config = manager.config
        row = config["records"][0]
        if a.action == "parent":
            for value in config["records"]:
                manager.submit(value)
            (Path(config["root"]) / "test_parent_ready").write_text(str(os.getpid()))
            time.sleep(120)
            return
        base = helper.base_module(config)
        if a.action == "after_collect":
            original = base.assist.collect

            def observe(*args, **kwargs):
                value = original(*args, **kwargs)
                helper.write(Path(config["root"]) / "paused_after_assist.json", dict(pid=os.getpid()))
                os.kill(os.getpid(), signal.SIGSTOP)
                return value

            base.assist.collect = observe
        elif a.action == "during_rsync":
            original = base.subprocess.run
            first = [True]

            def observe(argv, *args, **kwargs):
                if not first[0]:
                    return original(argv, *args, **kwargs)
                first[0] = False
                check = kwargs.pop("check", False)
                child = subprocess.Popen(argv, **kwargs)
                os.kill(child.pid, signal.SIGSTOP)
                helper.write(Path(config["root"]) / "paused_transport.json",
                             dict(collector=helper.identity(), transport=helper.identity(child.pid)))
                os.kill(os.getpid(), signal.SIGSTOP)
                code = child.wait()
                if check and code:
                    raise subprocess.CalledProcessError(code, argv)
                return subprocess.CompletedProcess(argv, code)

            base.subprocess.run = observe
        else:
            raise ValueError("Unknown private observation window")
        raise SystemExit(helper.worker(a.config, row["camera"], 0))

    base_config = helper.read(a.base_config)
    campaign = Path(base_config["campaign"])
    processes = []
    transports = []

    def case(name, count=1):
        config = json.loads(json.dumps(base_config))
        config["records"] = config["records"][:count]
        output = campaign / "operations/sam3_assist/v5" / ("async_crash1695_" + name)
        config["root"] = str(output / "controller")
        config["isolated_campaign"] = str(output / "isolated_campaign")
        for row in config["records"]:
            for host, variant in row["variants"].items():
                old = Path(variant["task"]).parent
                new = output / ("tasks" if host == "silver" else "tasks_local") / row["camera"]
                new.mkdir(parents=True, exist_ok=True)
                for filename in ("task.json", "reference.npz", "source_original_receipt.json",
                                 "source_portable_receipt.json", "union_resolver.py", "seed_policy_v5.json"):
                    shutil.copy2(old / filename, new / filename)
                (new / "raw_native.png").symlink_to((old / "raw_native.png").readlink())
                variant["task"] = str(new / "task.json")
            row.update(row["variants"]["silver"])
        path = root / (name + ".json")
        helper.write(path, config)
        manager = helper.Collectors(path)
        for row in config["records"]:
            manager.intent(row)
        return path, manager

    def launch(action, config):
        with (root / (action + ".log")).open("ab") as stream:
            child = subprocess.Popen(prefix + ["--action", action, "--config", str(config)],
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
                                                                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
        processes.append(child)
        return child

    def finish(manager, row):
        until(lambda: not manager.active(row))
        manager.submit(row)
        until(lambda: manager.result(row) is not None)
        result = helper.read(manager.root / row["camera"] / "complete.json")
        if not result["proof"]["union_conserved"] or not result["proof"]["canonical_post_clear_replay_exact"]:
            raise AssertionError("Original source checks were lost during recovery")
        original = Path(base_config["isolated_campaign"]) / "frames/001695"
        actual = Path(manager.config["isolated_campaign"]) / "frames/001695"
        for folder in ("native_masks", "person_candidates"):
            if helper.sha(original / folder / (row["camera"] + ".npz")) != helper.sha(actual / folder / (row["camera"] + ".npz")):
                raise AssertionError("Recovered actual camera archive differs")
        return result

    try:
        path, manager = case("parent_death", 2)
        parent = launch("parent", path)
        until(lambda: all((manager.root / row["camera"] / "worker_identity.json").exists()
                          for row in manager.config["records"]))
        owners = {row["camera"]: helper.read(manager.root / row["camera"] / "worker_identity.json")
                  for row in manager.config["records"]}
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)
        adopted = helper.Collectors(path)
        for row in adopted.config["records"]:
            if not helper.alive(owners[row["camera"]]):
                raise AssertionError("Collector did not survive its controller parent")
            if adopted.submit(row):
                raise AssertionError("Restart spawned a duplicate surviving camera collector")
        until(lambda: all(adopted.result(row) is not None for row in adopted.config["records"]))
        for row in adopted.config["records"]:
            result = helper.read(adopted.root / row["camera"] / "complete.json")
            if result["worker"]["pid"] != owners[row["camera"]]["pid"]:
                raise AssertionError("Adoption changed the actual collector owner")

        path, manager = case("after_assist")
        child = launch("after_collect", path)
        marker = Path(manager.config["root"]) / "paused_after_assist.json"
        until(marker.exists)
        row = manager.config["records"][0]
        before = helper.inventory(manager.config, row["camera"])
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        recovered = finish(manager, row)
        if not recovered["retained_partial"]:
            raise AssertionError("Interrupted partially published outputs were not retained")
        for relative, value in before.items():
            if helper.sha(Path(recovered["retained_partial"]) / relative) != value:
                raise AssertionError("Interrupted source evidence was overwritten")

        path, manager = case("transport_FD")
        child = launch("during_rsync", path)
        marker = Path(manager.config["root"]) / "paused_transport.json"
        until(marker.exists)
        transport = helper.read(marker)["transport"]
        transports.append(transport)
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        row = manager.config["records"][0]
        with helper.lock(manager.root / "slots/0.lock") as held:
            if held is not None:
                raise AssertionError("Orphan transport dropped its inherited collector slot lease")
        with helper.lock(manager.root / row["camera"] / "collection.lock") as held:
            if held is not None:
                raise AssertionError("Orphan transport dropped its inherited camera lease")
        if manager.submit(row):
            raise AssertionError("Duplicate writer launched while orphan transport retained leases")
        if not helper.alive(transport):
            raise AssertionError("Stopped exact test rsync identity changed")
        os.kill(transport["pid"], signal.SIGCONT)
        until(lambda: not helper.alive(transport))
        finish(manager, row)
        report = dict(passed=True, source_configuration_sha256=helper.sha(a.base_config),
                      helper_sha256=helper.sha(a.code / "marshawn_native_async_collect.py"),
                      controller_parent_death_preserves_exact_two_collectors=True,
                      completed_before_parent_state_merge_adopted_without_rerun=True,
                      collector_death_after_assist_recovers_with_retained_original_evidence=True,
                      collector_death_during_rsync_preserves_both_leases_in_child=True,
                      all_recovered_native_and_person_archives_byte_identical=True,
                      no_GPU_or_NN_job_or_canonical_source_mutation=True)
        helper.write(root / "report.json", report)
        print(json.dumps(report))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for old in transports:
            if helper.alive(old):
                os.kill(old["pid"], signal.SIGCONT)


if __name__ == "__main__":
    main()
