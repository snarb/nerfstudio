"""Inactive create-only HEAD spec preparation; no staging or NN launch.

The original queue's exact AST prefix runs serially. Only its local output
directory is rebound. It can run during BODY inference because all input
pixels and previous-frame references are already published and immutable.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

QUEUE_SHA256 = "70f54e41bd39061e8750efc013a91cc08a2ce05843a4d6243694b39371f5c983"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    sync_directory(path.parent)


def input_binding(campaign, frame, previous):
    """Bind every published input read by the unchanged original prefix."""
    campaign = Path(campaign)
    prior = campaign / "frames" / ("%06d" % previous) / "data/admission_unknown_hair/manifest.json"
    manifest = read(prior)
    cameras = read(campaign / "source_inventory.json")["cameras"]
    if manifest["frame"] != previous or len(manifest["records"]) != len(cameras) or len(cameras) != 66:
        raise ValueError("Foreign/incomplete prior HEAD manifest")
    paths = [campaign / "source_inventory.json", prior,
             campaign / "operations/sam3_assist/v5/full_release.json",
             campaign / "operations/marshawn_head_queue_v90.py"]
    for number in (previous, frame):
        ready = campaign / "sources" / str(number) / "source_ready.json"
        if not ready.exists():
            raise ValueError("Immutable source-ready input missing")
        paths.append(ready)
    for camera in cameras:
        row = next(x for x in manifest["records"] if x["camera"] == camera)
        strong = [p for p in row["proofs"] if p["status"] == "CURRENT_NATIVE_DUAL_PROMPT_ACTOR_HAIR"]
        chosen = strong[0] if strong else row["proofs"][-1]
        directory = prior.parent / chosen["directory"]
        paths += [directory / "qualification_original.json", directory / "current_native_proposal.npz"]
        if not strong:
            paths.append(campaign / "frames" / ("%06d" % previous) / "native_masks" / (camera + ".npz"))
        for number in (previous, frame):
            receipt = campaign / "sources" / str(number) / camera / "receipt.json"
            value = read(receipt)
            if value["frame"] != number or value["camera"] != camera:
                raise ValueError("Foreign current/previous source receipt")
            native = Path(value["raw_native_image"])
            if sha(native) != value["raw_native_image_sha256"]:
                raise ValueError("Native source PNG changed")
            paths += [receipt, native]
    return {str(path): sha(path) for path in paths}


def original_prefix(campaign):
    """Extract original statements, stopping before the first SSH/stage call."""
    campaign = Path(campaign)
    source = campaign / "operations/marshawn_head_queue_v90.py"
    if sha(source) != QUEUE_SHA256:
        raise ValueError("Original HEAD queue source changed")
    tree = ast.parse(source.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make")
    cut = next(i for i, n in enumerate(original.body)
               if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Attribute)
               and isinstance(n.value.func.value, ast.Name)
               and n.value.func.value.id == "subprocess" and n.value.func.attr == "run")
    prefix = original.body[:cut]
    local_output = [n for n in prefix if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "O" for t in n.targets)]
    if len(local_output) != 1:
        raise ValueError("Unexpected original local output assignment")
    # Only the local output assignment is replaced by an explicit argument.
    # O.name remains the canonical frame name, preserving every remote locator.
    body = [n for n in prefix if n is not local_output[0]]
    loop = next(n for n in prefix if isinstance(n, ast.For))
    if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(loop)):
        raise ValueError("Unexpected import inside scientific camera loop")
    function = ast.FunctionDef(
        name="exact_spec_prefix",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=x) for x in ("frame", "previous", "O")],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body + [ast.Return(value=ast.Name(id="O", ctx=ast.Load()))], decorator_list=[])
    sys.path[:0] = [str(campaign / "operations"), str(campaign / "code/LookCloser/scripts")]
    spec = importlib.util.spec_from_file_location("_original_head_spec_queue", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.V = campaign
    scope = dict(module.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), "exec"), scope)
    proof = dict(original_queue_sha256=QUEUE_SHA256,
                 original_camera_loop_AST_sha256=hashlib.sha256(ast.dump(loop, include_attributes=False).encode()).hexdigest(),
                 prefix_AST_sha256=hashlib.sha256(ast.dump(ast.Module(body=prefix, type_ignores=[]), include_attributes=False).encode()).hexdigest(),
                 permitted_delta="Local O assignment supplied as private argument; same basename. All remaining prefix AST nodes unchanged. Stop before first SSH call.",
                 cv2_threads=module.cv2.getNumThreads())
    return scope["exact_spec_prefix"], proof


def validate(campaign, frame, previous, bundle):
    bundle = Path(bundle)
    manifest = read(bundle / "spec_ahead_complete.json")
    if (manifest["frame"], manifest["previous"], manifest["helper_sha256"]) != (frame, previous, sha(__file__)):
        raise ValueError("Foreign prepared HEAD bundle")
    if manifest["input_sha256"] != input_binding(campaign, frame, previous):
        raise ValueError("Prepared HEAD source/reference inputs changed")
    for name, digest in manifest["files_sha256"].items():
        if sha(bundle / name) != digest:
            raise ValueError("Prepared HEAD file changed: " + name)
    if (bundle / "launch.json").exists():
        raise ValueError("Spec-only bundle unexpectedly contains NN launch")
    return manifest


def prepare(campaign, frame, previous, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ("%06d.lock" % frame)).open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        bundle = output / ("%d_head_ALL66_ROI_v73" % frame)
        if bundle.exists():
            return validate(campaign, frame, previous, bundle)
        started = time.time()
        binding = input_binding(campaign, frame, previous)
        bound_at = time.time()
        original, proof = original_prefix(campaign)
        attempt = output / (".attempt." + uuid.uuid4().hex)
        attempt.mkdir()
        private = attempt / bundle.name
        original(frame, previous, private)
        prefix_finished = time.time()
        if input_binding(campaign, frame, previous) != binding:
            raise ValueError("HEAD immutable inputs changed during preparation")
        files = list(private.glob("*/spec.json")) + [private / "plan.json", private / "source_ROI_proofs.json"]
        if len(files) != 68:
            raise ValueError("Expected66 specs plus original plan/proofs")
        manifest = dict(frame=frame, previous=previous, started_at=started, input_bound_at=bound_at,
                        prefix_finished_at=prefix_finished, finished_at=time.time(),
                        helper_sha256=sha(__file__), input_sha256=binding, AST_proof=proof,
                        files_sha256={str(p.relative_to(private)): sha(p) for p in files},
                        no_NN_GPU_stage_or_source_mutation=True,
                        adoption_API="validate(campaign,frame,previous,bundle); sole source owner may then adopt exact specs/plan/proofs under its original stage ownership")
        atomic(private / "spec_ahead_complete.json", manifest)
        os.rename(private, bundle)
        sync_directory(output)
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--previous", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.campaign, args.frame, args.previous, args.output)
    print(json.dumps({k: result[k] for k in ("frame", "started_at", "input_bound_at", "prefix_finished_at", "finished_at")}))
