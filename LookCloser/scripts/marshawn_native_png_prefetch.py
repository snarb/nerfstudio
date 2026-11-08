"""Optional chronological PNG-only prefetch; never publishes model/source readiness."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".%d.pending" % os.getpid())
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    try:
        stat = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(pid=pid, start=stat[19], state=stat[0],
                boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def alive(value):
    current = identity(value["pid"]) if value else None
    return bool(current and current["state"] not in ("Z", "X") and
                all(current[key] == value[key] for key in ("pid", "start", "boot")))


def source_binding(campaign, frame, camera, verify_png=False):
    campaign = Path(campaign)
    marker = campaign / "sources" / str(frame) / "source_ready.json"
    source = read(marker)
    if source["frame"] != frame or len(source["cameras"]) != 66 or not source["source_verified_all66"]:
        raise ValueError("Canonical ALL66 source-ready receipt required")
    row = next(row for row in source["records"] if row["camera"] == camera)
    if sha(row["receipt"]) != row["receipt_sha256"]:
        raise ValueError("Current source receipt changed")
    receipt = read(row["receipt"])
    image = Path(receipt["raw_native_image"])
    if receipt["frame"] != frame or receipt["camera"] != camera or receipt["raw_sha256"] != row["raw_sha256"]:
        raise ValueError("Wrong current-frame native source")
    if (str(image) != row["raw_native_image"] or
            receipt["raw_native_image_sha256"] != row["raw_native_image_sha256"]):
        raise ValueError("Native raster source differs from source-ready inventory")
    if verify_png and sha(image) != receipt["raw_native_image_sha256"]:
        raise ValueError("Actual native PNG changed")
    stat = image.stat()
    return dict(frame=frame, camera=camera, source_ready_sha256=sha(marker),
                source_receipt_sha256=row["receipt_sha256"], original_TIFF_sha256=row["raw_sha256"],
                native_png_sha256=receipt["raw_native_image_sha256"], source_native_png=str(image),
                size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def command(config, argv, held_fd=None, timeout=60):
    args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", config["host"], shlex.join(list(map(str, argv)))]
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout,
                          pass_fds=() if held_fd is None else (held_fd,))


def rpc(config, action, value, held_fd=None):
    request = dict(value, helper_sha256=config["remote_helper_sha256"])
    token = hashlib.sha256(json.dumps(dict(action=action, request=request), sort_keys=True).encode()).hexdigest()
    local = Path(config["operations"]) / "requests" / (token + ".json")
    if local.exists() and read(local) != request:
        raise ValueError("Native prefetch immutable request changed")
    if not local.exists():
        write(local, request)
    remote = Path(config["cache_root"]) / "requests" / (token + ".json")
    command(config, ["mkdir", "-p", str(remote.parent)], held_fd)
    subprocess.run(["rsync", "-t", str(local), config["host"] + ":" + str(remote)], check=True,
                   capture_output=True, timeout=60, pass_fds=() if held_fd is None else (held_fd,))
    result = command(config, [config["python"], config["remote_helper"], action, "--request", str(remote)], held_fd)
    return json.loads(result.stdout)


def active_consumer(config, frame, binding):
    path = Path(config["operations"]) / "consumers" / ("%06d.json" % frame)
    if not path.exists():
        return False
    value = read(path)
    return bool(value.get("kind") == "NATIVE_PNG_CACHE_CONSUMER_V1" and
                value.get("frame") == frame and value.get("source_ready_sha256") == binding["source_ready_sha256"] and
                value.get("status") == "active" and alive(value.get("owner")))


def one(config, frame, camera, held_fd=None):
    binding = source_binding(config["campaign"], frame, camera, verify_png=False)
    operations = Path(config["operations"])
    record = operations / "records" / ("%06d" % frame) / (camera + ".json")
    if record.exists():
        old = read(record)
        if old["binding"] != binding:
            raise ValueError("Cached current PNG belongs to another source")
        return "complete"
    result = rpc(config, "ensure", dict(root=config["cache_root"], binding=binding), held_fd)
    if not result["ready"]:
        if active_consumer(config, frame, binding):
            return "yield_active_consumer"
        capacity = rpc(config, "capacity", dict(root=config["cache_root"]), held_fd)
        if capacity["free_bytes"] < config["minimum_free_bytes"] + binding["size"] or capacity["cache_bytes"] + binding["size"] > config["maximum_cache_bytes"]:
            return "wait_capacity"
        if sha(binding["source_native_png"]) != binding["native_png_sha256"]:
            raise ValueError("Actual native source changed before transfer")
        incoming = Path(config["cache_root"]) / "incoming" / ("%06d" % frame) / camera / "raw_native.png"
        command(config, ["mkdir", "-p", str(incoming.parent)], held_fd)
        subprocess.run(["rsync", "-tL", "--partial", "--timeout=45", "--bwlimit=" + str(config["bandwidth_KiB_per_second"]),
                        binding["source_native_png"], config["host"] + ":" + str(incoming)],
                       check=True, capture_output=True, timeout=240,
                       pass_fds=() if held_fd is None else (held_fd,))
        result = rpc(config, "ensure", dict(root=config["cache_root"], binding=binding, incoming=str(incoming)), held_fd)
    if not result["ready"] or result["receipt"]["binding"] != binding:
        raise ValueError("Native PNG cache did not publish the exact source binding")
    write(record, dict(binding=binding, remote=result, no_NN_or_model_state=True))
    return "complete"


def guard(config, path):
    if sha(path) != config["configuration_sha256"]:
        raise ValueError("Native prefetch configuration changed")
    for filename, digest in config["files_sha256"].items():
        if sha(filename) != digest:
            raise ValueError("Native prefetch operational code changed")


def run(config, path):
    root = Path(config["operations"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a+") as held:
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        write(root / "controller_identity.json", identity())
        while True:
            guard(config, path)
            complete = 0
            yielded = 0
            errors = []
            for frame in config["frames"]:
                marker = Path(config["campaign"]) / "sources" / str(frame) / "source_ready.json"
                if not marker.exists():
                    continue
                source = read(marker)
                for camera in source["cameras"]:
                    try:
                        status = one(config, frame, camera, held.fileno())
                    except Exception as error:
                        errors.append(dict(frame=frame, camera=camera, error=str(error)))
                        status = "error"
                    complete += status == "complete"
                    yielded += status == "yield_active_consumer"
                    write(root / "progress.json", dict(frame=frame, camera=camera, status=status,
                          completed_this_pass=complete, yielded_this_pass=yielded, errors=errors[-8:], time=time.time(),
                          PNG_inputs_only=True))
            total = len(config["frames"]) * 66
            if complete == total:
                write(root / "complete.json", dict(PNGs=total, time=time.time(), PNG_inputs_only=True))
                return
            time.sleep(10)


def watchdog(config, path):
    root = Path(config["operations"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "watchdog.lock").open("a+") as held:
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        write(root / "watchdog_identity.json", identity())
        while not (root / "complete.json").exists():
            guard(config, path)
            old = read(root / "controller_identity.json") if (root / "controller_identity.json").exists() else None
            if not alive(old):
                with (root / "controller.log").open("ab") as stream:
                    subprocess.Popen([sys.executable, __file__, "run", "--config", str(path)], stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            time.sleep(10)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run", "watchdog", "one"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--frame", type=int)
    p.add_argument("--camera")
    a = p.parse_args()
    c = read(a.config)
    # The file's digest is an external runtime binding, never a self-hash field.
    c["configuration_sha256"] = sha(a.config)
    if a.action == "one":
        print(one(c, a.frame, a.camera))
    elif a.action == "run":
        run(c, a.config)
    else:
        watchdog(c, a.config)
