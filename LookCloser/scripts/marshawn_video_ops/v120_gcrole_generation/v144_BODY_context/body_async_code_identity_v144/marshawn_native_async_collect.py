"""Two owned CPU collectors around the unchanged sealed V62 gather function.

No GPU job, native priority, mask math, or source publication is implemented here.
The native controller remains the sole writer of its fleet/camera state.
"""
from pathlib import Path
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
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
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    try:
        fields = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(pid=pid, start=fields[19], state=fields[0],
                boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def alive(old):
    if not old:
        return False
    now = identity(old["pid"])
    return bool(now and now["state"] not in ("Z", "X") and
                all(now[key] == old[key] for key in ("pid", "start", "boot")))


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as held:
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
            return
        yield held


def isolated_root(config):
    campaign = Path(config["campaign"]).resolve()
    root = Path(config["isolated_campaign"]).absolute()
    if root == campaign or "/operations/sam3_assist/v5/" not in str(root):
        raise ValueError("Collectors may write only isolated V5 review outputs")
    root.relative_to(campaign / "operations/sam3_assist/v5")
    for path in (root, root / "frames", root / "frames" / ("%06d" % config["frame"])):
        if path.is_symlink():
            raise ValueError("Collector output hierarchy aliases a different campaign")
    return root


def initialize(config):
    root = Path(config["root"])
    with lock(root / "collectors/initialize.lock") as held:
        if held is None:
            return False
        output = isolated_root(config)
        output.mkdir(parents=True, exist_ok=True)
        for name in ("sources", "code"):
            target = output / name
            expected = Path(config["campaign"]) / name
            if not target.exists():
                target.symlink_to(expected, target_is_directory=True)
            if target.resolve() != expected.resolve():
                raise ValueError("Collector source/code alias changed")
    return True


def camera_paths(config, camera):
    frame = isolated_root(config) / "frames" / ("%06d" % config["frame"])
    # These exact paths belong only to this camera's unchanged gather result.
    return [frame / "native_masks" / (camera + suffix) for suffix in (".npz", ".json", ".assist.partial.npz")] + [
        frame / "native_masks/assist_provenance" / (camera + suffix)
        for suffix in (".remote_mask.json", ".operation.json", ".remote_candidates.json")] + [
        frame / "person_candidates" / (camera + suffix) for suffix in (".npz", ".json")] + [
        frame / "pre_clear" / camera, frame / "union_resolution_inputs" / camera]


def inventory(config, camera):
    root = isolated_root(config)
    paths = []
    for path in camera_paths(config, camera):
        if path.is_dir():
            paths.extend(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            paths.append(path)
    if not paths:
        raise ValueError("Collector has no output artifacts")
    result = {}
    for path in sorted(paths):
        if path.is_symlink():
            raise ValueError("Collected output may not alias source inputs")
        result[str(path.relative_to(root))] = sha(path)
    return result


def quarantine_partial(config, camera, unit):
    existing = [path for path in camera_paths(config, camera) if path.exists()]
    if not existing:
        return None
    source = isolated_root(config)
    target = unit / "interrupted" / str(time.time_ns())
    moved = []
    for path in existing:
        if path.is_symlink():
            raise ValueError("Refuse to move aliased source as partial collector output")
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
        moved.append(str(path.relative_to(source)))
    write(target / "retained.json", dict(frame=config["frame"], camera=camera,
                                         original_output=str(source), retained=moved,
                                         reason="Previous collector had no atomic completion receipt"))
    return target


def base_module(config):
    directory = (Path(config["campaign"]) / "operations/sam3_assist/code").resolve()
    sys.path[:0] = [str(directory), str(Path(config["campaign"]) / "code/LookCloser/scripts")]
    module = importlib.import_module("marshawn_sam3_union_batch_v62_runtime")
    if Path(module.__file__).resolve() != directory / "marshawn_sam3_union_batch_v62_runtime.py":
        raise ValueError("Collector escaped the qualified original gather source")
    return module


class Collectors:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = read(self.config_path)
        self.root = Path(self.config["root"]) / "collectors"
        self.configuration_sha256 = sha(self.config_path)
        self.base_sha256 = sha(Path(self.config["campaign"]) / "operations/sam3_assist/code/marshawn_sam3_union_batch_v62_runtime.py")
        self.helper_sha256 = sha(__file__)
        self.root.mkdir(parents=True, exist_ok=True)
        while not initialize(self.config):
            time.sleep(.05)

    def intent(self, row):
        unit = self.root / row["camera"]
        binding = dict(configuration_sha256=self.configuration_sha256,
                       helper_sha256=self.helper_sha256, base_gather_source_sha256=self.base_sha256,
                       frame=self.config["frame"], camera=row["camera"], row=row)
        path = unit / "intent.json"
        if path.exists():
            if read(path) != binding:
                raise ValueError("Existing collector belongs to another source/configuration")
        else:
            write(path, binding)
        return unit

    def result(self, row):
        unit = self.root / row["camera"]
        path = unit / "complete.json"
        if not path.exists():
            return None
        self.intent(row)
        result = read(path)
        if (result["intent_sha256"] != sha(unit / "intent.json") or
                result["files_sha256"] != inventory(self.config, row["camera"])):
            raise ValueError("Completed collector proof or output changed")
        return result["proof"]

    def active(self, row):
        unit = self.root / row["camera"]
        if not (unit / "intent.json").exists():
            return False
        self.intent(row)
        for name in ("worker_identity.json", "launch_identity.json"):
            path = unit / name
            if path.exists() and alive(read(path)):
                return True
        with lock(unit / "collection.lock") as held:
            return held is None

    def submit(self, row):
        if self.result(row) is not None or self.active(row):
            return False
        free = []
        for slot in range(2):
            with lock(self.root / "slots" / (str(slot) + ".lock")) as held:
                if held is not None:
                    free.append(slot)
        # Launch reservations also count before a just-spawned child takes its
        # slot lock. Actual slot locks enforce the two-collector bound even if
        # the controller dies between Popen and identity publication.
        launching = set()
        for path in self.root.glob("*/launch_identity.json"):
            value = read(path)
            if alive(value):
                launching.add(value["slot"])
        free = [slot for slot in free if slot not in launching]
        if not free:
            return False
        unit = self.intent(row)
        argv = [sys.executable, str(Path(__file__).resolve()), "worker", "--config", str(self.config_path),
                "--camera", row["camera"], "--slot", str(free[0])]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        with (unit / "stdout.log").open("ab") as stream:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, env=env, start_new_session=True)
        value = identity(child.pid)
        if value is not None:
            write(unit / "launch_identity.json", dict(value, slot=free[0], intent_sha256=sha(unit / "intent.json")))
        return True


def worker(config_path, camera, slot):
    config = read(config_path)
    manager = Collectors(config_path)
    unit = manager.root / camera
    intent = read(unit / "intent.json")
    row = intent["row"]
    if intent["configuration_sha256"] != sha(config_path) or intent["helper_sha256"] != sha(__file__):
        raise ValueError("Collector worker source/config changed")
    with lock(manager.root / "slots" / (str(slot) + ".lock")) as pool_held:
        if pool_held is None:
            return 75
        with lock(unit / "collection.lock") as camera_held:
            if camera_held is None:
                return 75
            if manager.result(row) is not None:
                return 0
            base = base_module(config)
            if sha(base.__file__) != intent["base_gather_source_sha256"]:
                raise ValueError("Original gather source changed")
            base.guard(config)
            write(unit / "worker_identity.json", dict(identity(), slot=slot, intent_sha256=sha(unit / "intent.json")))
            retained = quarantine_partial(config, camera, unit)
            started = time.time()
            original_run = base.subprocess.run
            events = []

            def inherited_transport(argv, *args, **kwargs):
                beginning = time.time()
                kwargs["pass_fds"] = tuple(set(kwargs.get("pass_fds", ())) |
                                           {pool_held.fileno(), camera_held.fileno()})
                try:
                    return original_run(argv, *args, **kwargs)
                finally:
                    events.append(dict(argv=argv, seconds=time.time() - beginning))

            base.subprocess.run = inherited_transport
            try:
                proof = base.gather(config, row)
            finally:
                base.subprocess.run = original_run
            result = dict(intent_sha256=sha(unit / "intent.json"), proof=proof,
                          files_sha256=inventory(config, camera), started_at=started, finished_at=time.time(),
                          worker=identity(), transport=events, source_and_validation_unchanged=True,
                          retained_partial=str(retained) if retained else None)
            write(unit / "complete.json", result)
            return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["worker"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--slot", type=int, choices=[0, 1], required=True)
    a = p.parse_args()
    raise SystemExit(worker(a.config, a.camera, a.slot))
