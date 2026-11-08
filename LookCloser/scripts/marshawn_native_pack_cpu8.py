"""Pack unchanged V5 native-SAM3 task inputs in ordered camera processes.

This is an inactive operational factory helper, not a producer or GPU launcher.
Each camera retains silver-then-olive packing and the sealed adapter's fsync,
source hashing, seed policy, and reference tensors. Process isolation is needed
because that adapter temporarily replaces its imported module's seed function.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _bounded_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"


def _pack_camera(item):
    _bounded_cpu()
    campaign = Path(item["campaign"])
    identity = Path(item["campaign_identity"])
    code = identity / "operations/sam3_assist/code"
    sys.path.insert(0, str(code))
    assist = importlib.import_module("marshawn_sam3_assist_v5")
    io = importlib.import_module("marshawn_sam3_assist")
    if Path(assist.__file__).resolve() != code / "marshawn_sam3_assist_v5.py":
        raise ValueError("Native packing import escaped the qualified operational adapter")
    for name, expected in item["adapter_sources_sha256"].items():
        if sha(code / name) != expected:
            raise ValueError("Native packing operational source changed: " + name)
    # These imports occur after one-thread environment limits, before any seed
    # arrays are loaded. No SAM3 model or checkpoint is loaded by pack().
    import cv2
    cv2.setNumThreads(1)
    from threadpoolctl import threadpool_limits
    started = time.time()
    frame, camera = item["frame"], item["camera"]
    remote_root = Path(item["release"]["remote_root"])
    remote_frame = remote_root / "operations/sam3_assist/v5" / ("frame%06d" % frame)
    variants = {}
    with threadpool_limits(limits=1):
        for host in ("silver", "olive"):
            folder = "tasks" if host == "silver" else "tasks_local"
            task = Path(item["task_root"]) / folder / camera
            remote_task = (remote_frame / "tasks" / camera if host == "silver"
                           else Path(item["logical_root"]) / folder / camera)
            assist.pack(campaign, frame, camera, task, str(remote_task),
                        str(remote_root / "code" if host == "silver" else identity / "code"),
                        str(remote_root / "tools/sam3_overlay") if host == "silver"
                        else item["release"]["local_overlay"],
                        identity / "operations/marshawn_union_resolution_v5.py")
            variants[host] = dict(task=str(task / "task.json"),
                                  task_sha256=io.sha(task / "task.json"))
    original = campaign / "frames" / ("%06d" % frame) / "native_masks" / (camera + ".npz")
    logical_original = identity / "frames" / ("%06d" % frame) / "native_masks" / (camera + ".npz")
    record = dict(camera=camera, host="silver", **variants["silver"], variants=variants,
                  original_native=str(logical_original) if original.exists() else None,
                  original_native_sha256=io.sha(original) if original.exists() else None)
    return record, dict(camera=camera, pid=os.getpid(), started_at=started,
                        finished_at=time.time(), host_order=["silver", "olive"])


def pack_records(campaign, frame, cameras, task_root, release, *, workers=8,
                 campaign_identity=None, logical_root=None):
    """Return original-order factory records; write only beneath task_root.

    The optional identity/logical roots support read-only immutable-input parity
    replay: output locators differ while portable task bytes retain production
    paths. Production callers leave both options unset. This function does not
    publish a configuration, source marker, drain request, or GPU intent.
    """
    campaign, task_root = Path(campaign).resolve(), Path(task_root).resolve()
    identity = Path(campaign_identity).resolve() if campaign_identity else campaign
    logical = Path(logical_root).resolve() if logical_root else task_root
    cameras = list(cameras)
    if not cameras or len(cameras) != len(set(cameras)):
        raise ValueError("An ordered, unique nonempty camera list is required")
    if not 1 <= workers <= 8:
        raise ValueError("Packing pool must contain one to eight CPU workers")
    code = identity / "operations/sam3_assist/code"
    source_names = ("marshawn_sam3_assist.py", "marshawn_sam3_assist_v5.py",
                    "marshawn_sam3_union_capture_v5.py")
    bindings = {name: sha(code / name) for name in source_names}
    if release.get("files_sha256"):
        for name, value in bindings.items():
            expected = release["files_sha256"].get(str(code / name))
            if expected is not None and expected != value:
                raise ValueError("Release differs from native packing source: " + name)
    started = time.time()
    _bounded_cpu()
    items = [dict(campaign=str(campaign), campaign_identity=str(identity), frame=int(frame),
                  camera=camera, task_root=str(task_root), logical_root=str(logical),
                  release=release, adapter_sources_sha256=bindings) for camera in cameras]
    # executor.map preserves input order even when cameras complete out of order.
    # Spawn isolates seed overrides and avoids inherited large array contexts.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        outputs = list(pool.map(_pack_camera, items, chunksize=1))
    records = [row[0] for row in outputs]
    execution = dict(kind="UNCHANGED_V5_ORDERED_CAMERA_PACK_CPU8_V1", frame=int(frame),
                     workers=workers, camera_order=cameras, host_order=["silver", "olive"],
                     adapter_sources_sha256=bindings, helper_sha256=sha(__file__),
                     started_at=started, finished_at=time.time(),
                     cameras=[row[1] for row in outputs], no_GPU_or_NeRF_state_loaded=True)
    return records, execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--campaign-identity", type=Path)
    parser.add_argument("--logical-root", type=Path)
    args = parser.parse_args()
    source = json.loads((args.campaign / "sources" / str(args.frame) / "source_ready.json").read_text())
    if source["frame"] != args.frame or len(source["cameras"]) != 66 or not source["source_verified_all66"]:
        raise ValueError("Packing requires verified same-frame ALL66 sources")
    records, execution = pack_records(args.campaign, args.frame, source["cameras"],
                                      args.task_root, json.loads(args.release.read_text()),
                                      workers=args.workers, campaign_identity=args.campaign_identity,
                                      logical_root=args.logical_root)
    sys.path.insert(0, str((args.campaign_identity or args.campaign) / "operations/sam3_assist/code"))
    from marshawn_sam3_assist import write
    write(args.task_root / "packing_records.json", records)
    write(args.task_root / "packing_execution.json", execution)
    print(json.dumps(dict(cameras=len(records), seconds=execution["finished_at"]-execution["started_at"])))


if __name__ == "__main__":
    main()
