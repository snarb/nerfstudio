"""Create-only SG NVMe storage of source-bound immutable native PNG inputs."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import uuid


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def paths(root, binding):
    root = Path(root).resolve()
    if root.name != "native_png_prefetch_v1" or root.parent.name != "operations":
        raise ValueError("Owned native-input cache root required")
    frame, camera = binding["frame"], binding["camera"]
    if type(frame) is not int or not 1705 <= frame <= 2015 or frame % 5 or len(camera) != 2 or not camera.isalpha() or not camera.isupper():
        raise ValueError("Current original frame and physical camera required")
    directory = root / "cache" / ("%06d" % frame) / camera
    normal = root.parents[1] / "operations/sam3_assist/v5" / ("frame%06d" % frame) / "tasks" / camera / "raw_native.png"
    return root, directory, normal


def verify(path, binding):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Native cache file must be an owned regular PNG")
    if path.stat().st_size != binding["size"] or sha(path) != binding["native_png_sha256"]:
        raise ValueError("Native PNG source hash/size differs")
    if path.stat().st_mtime_ns != binding["mtime_ns"]:
        raise ValueError("Native PNG source mtime differs")


def ready(root, binding):
    _, directory, _ = paths(root, binding)
    receipt = directory / "ready.json"
    image = directory / "raw_native.png"
    if not receipt.exists() or not image.exists():
        return None
    value = read(receipt)
    if value["binding"] != binding:
        raise ValueError("Existing cache belongs to another immutable source")
    verify(image, binding)
    return value


def ensure(root, binding, incoming=None):
    root, directory, normal = paths(root, binding)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "install.lock").open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        old = ready(root, binding)
        if old is not None:
            return dict(ready=True, receipt=old, receipt_sha256=sha(directory / "ready.json"))
        target = directory / "raw_native.png"
        mode = "reused_unsealed_verified_cache"
        if target.exists():
            verify(target, binding)
        elif normal.exists():
            verify(normal, binding)
            os.link(normal, target)
            mode = "hardlink_from_existing_normal_source_input"
        elif incoming is not None:
            incoming = Path(incoming).absolute()
            incoming.relative_to(root / "incoming")
            verify(incoming, binding)
            # A hardlink is an atomic create and never replaces another file.
            os.link(incoming, target)
            mode = "verified_prefetched_source_input"
        else:
            return dict(ready=False)
        verify(target, binding)
        value = dict(kind="SHA_VERIFIED_NATIVE_PNG_CACHE_V1", binding=binding,
                     path=str(target), inode=target.stat().st_ino,
                     mode=mode, no_model_or_NN_output=True)
        atomic(directory / "ready.json", value)
        return dict(ready=True, receipt=value, receipt_sha256=sha(directory / "ready.json"))


def link_frame(root, bindings):
    records = []
    for binding in bindings:
        _, directory, normal = paths(root, binding)
        receipt = ready(root, binding)
        if receipt is None:
            records.append(dict(camera=binding["camera"], linked=False, reason="cache_not_ready"))
            continue
        normal.parent.mkdir(parents=True, exist_ok=True)
        with (normal.parent / ".native_png_cache_link.lock").open("a+") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            if normal.exists():
                verify(normal, binding)
                created = False
            else:
                os.link(directory / "raw_native.png", normal)
                created = True
            verify(normal, binding)
            records.append(dict(camera=binding["camera"], linked=True, created=created,
                                target=str(normal), same_inode=normal.stat().st_ino == (directory / "raw_native.png").stat().st_ino,
                                cache_receipt_sha256=sha(directory / "ready.json")))
    return dict(kind="CREATE_ONLY_NATIVE_PNG_CACHE_CONSUMER_V1", records=records,
                cache_optional=True, original_rsync_and_stream_SHA_required=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["ensure", "link-frame", "capacity"])
    p.add_argument("--request", type=Path, required=True)
    a = p.parse_args()
    q = read(a.request)
    if q["helper_sha256"] != sha(__file__):
        raise ValueError("Native PNG cache helper changed")
    if a.action == "ensure":
        result = ensure(q["root"], q["binding"], q.get("incoming"))
    elif a.action == "link-frame":
        result = link_frame(q["root"], q["bindings"])
    else:
        root = Path(q["root"])
        root.mkdir(parents=True, exist_ok=True)
        result = dict(free_bytes=shutil.disk_usage(root).free,
                      # Native PNGs are hardlinked, so count unique owned inodes.
                      cache_bytes=sum(p.stat().st_size for p in (root / "cache").glob("*/*/raw_native.png")))
    print(json.dumps(result))
