"""Opt-in, frame-configured DEC5 MHR local completion; no fitting or promotion.

The numerical producers remain frozen. Two candidate path literals and one
mask-parent path literal are adapted with exact-count source checks. Run stages
in separate processes: imported audit globals are deliberately configured first.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import importlib
import inspect
import json
import os
import re

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
from study_multiview_face_prior import read, save, sha

ARM = "silhouette100"
RECIPE_SHA256 = "29ed956bc89d846e11ee8fb9c18e5bdfcf37a489629ccc8911cc4e800ad28d28"
FIELDS = {"frame", "production_request", "mesh", "metadata", "prior",
          "inherited_fit", "measured_control", "mask_parent_request",
          "frozen_admission_request"}
RESERVED_PREFIXES = ("E004_B", "F004_D", "G004_B", "H004_D",
                     "I004_B", "K004_B", "L004_D", "M004_B")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_spec(spec, output):
    require(set(spec) == FIELDS, "Unexpected/missing config fields; no gate overrides accepted")
    require(isinstance(spec["frame"], str) and re.fullmatch(r"\d{6}", spec["frame"]),
            "Frame must be an explicit six-digit source time")
    output = Path(output).resolve()
    require(len(output.parts) >= 4, "Use a dedicated private output directory")
    for key in FIELDS - {"frame"}:
        p = Path(spec[key])
        require(p.is_absolute() and p.exists(), f"Missing absolute input: {key}")
        p = p.resolve()
        protected = p if p.is_dir() else p.parent
        require(output != protected and output not in protected.parents
                and protected not in output.parents, f"Output overlaps input tree: {key}")
    return output


def check_seal(root, checked):
    seal = read(root / "final_seal.json")
    require(seal["status"] == "passed", "Prior is not sealed")
    paths = [(root / p, h) for p, h in seal["inventory"].items()]
    paths += [(Path(p), h) for p, h in seal["checked_bindings"].items()]
    for path, digest in paths:
        require(sha(path) == digest, f"Changed sealed input: {path}")
        checked[str(path.resolve())] = digest
    checked[str((root / "final_seal.json").resolve())] = sha(root / "final_seal.json")


def inspect_inputs(spec, output):
    from joint_temporal_texture import cameras, HELD_CAMERAS, CALIBRATION, SOURCE
    import admit_mhr_local_patch_depth as admission
    output = validate_spec(spec, output)
    frame = spec["frame"]; prior = Path(spec["prior"]); checked = {}
    check_seal(prior, checked)
    protocol = read(prior / "protocol.json")
    require(protocol["frame"] == frame and not protocol["target_used"], "Wrong-time or target-conditioned prior")
    recipe_hash = hashlib.sha256(json.dumps(protocol["recipe"], sort_keys=True).encode()).hexdigest()
    require(recipe_hash == RECIPE_SHA256, "Prior recipe differs from frozen 2px/100-step control")
    require(protocol["continuation"]["same_base_reference"]
            and protocol["continuation"]["no_weight_or_mask_changes"], "Changed continuation reference")
    require(sha(spec["inherited_fit"]) == protocol["input_hashes"].get(spec["inherited_fit"]),
            "Inherited conformance does not belong to this prior")
    rows, raw, metadata = cameras(frame)
    names = {r["physical_camera"] for r in rows}
    fit, reserved = set(protocol["fit_cameras"]), set(protocol["validation_cameras"])
    require(len(rows) == 62 and len(names) == 62 and not names & HELD_CAMERAS, "Train camera leakage/duplicates")
    require(len(fit) == 54 and len(reserved) == 8 and not fit & reserved and fit | reserved == names,
            "Prior 54/8 split mismatch")
    require(reserved == {n for n in names if n.startswith(RESERVED_PREFIXES)}, "Changed reserved camera cohort")
    require(Path(protocol["original_mesh"]).resolve() == Path(raw).resolve()
            and sha(raw) == protocol["original_mesh_sha256"], "Prior normalization/raw frame mismatch")
    require(Path(metadata).resolve() == Path(spec["metadata"]).resolve(), "Wrong normalization metadata")
    entries = [r for r in read(spec["production_request"])["inventory"] if r["frame_id"] == frame]
    require(len(entries) == 1, "Production request must contain exactly one source time")
    entry = entries[0]
    for key in ("mesh", "metadata"):
        require(Path(entry[key]).resolve() == Path(spec[key]).resolve()
                and sha(spec[key]) == entry[key + "_sha256"], f"Production {key} mismatch")
    measured = read(Path(spec["measured_control"]) / "request.json")
    require(measured["frame"] == frame and measured["source_mesh_sha256"] == sha(spec["mesh"]),
            "Measured-control source/frame mismatch; no implicit geometry rebind")
    require(sha(spec["mask_parent_request"]) == measured["parent_request_sha256"], "Mask parent mismatch")
    frozen = read(spec["frozen_admission_request"])
    require(sha(admission.__file__) == frozen["script_sha256"], "Admission producer changed")
    for name, digest in frozen["helpers"].items():
        p = Path(admission.__file__).with_name(name)
        require(sha(p) == digest, f"Frozen guard changed: {name}")
        checked[str(p.resolve())] = digest
    for key in FIELDS - {"frame"}:
        p = Path(spec[key])
        if p.is_file(): checked[str(p.resolve())] = sha(p)
    for p in [Path(spec["measured_control"]) / "request.json", CALIBRATION,
              SOURCE / frame / "transforms.json", Path(__file__),
              Path(__file__).with_name("build_mhr_silhouette_patch_candidates.py"),
              Path(__file__).with_name("build_mhr_local_patch_candidates.py"),
              Path(__file__).with_name("audit_mhr_local_patch_depth.py")]:
        checked[str(p.resolve())] = sha(p)
    return dict(schema=1, spec=spec, output=str(output), input_hashes=checked,
                frame=frame, training_cameras=sorted(names), reserved_fitting_cameras=sorted(reserved),
                normalization_metadata_identical=True, frozen_numerics=True,
                fresh_fit_performed=False, heldout_used=False, production_modified=False,
                dataset_scope="fixed calibrated DEC5 62-train-camera rig only",
                inferred_geometry=True, visual_acceptance=False)


def adapt(module, function, replacements, namespace):
    source = inspect.getsource(getattr(module, function)); generated = source
    for before, after, count in replacements:
        require(generated.count(before) == count, f"Frozen adapter mismatch: {function}: {before}")
        generated = generated.replace(before, after)
    state = dict(module.__dict__, **namespace)
    exec(compile(generated, "<explicit_local_mhr_path_adapter>", "exec"), state)
    return state[function], dict(frozen_path=str(Path(module.__file__).resolve()),
        frozen_sha256=sha(module.__file__), original_source=source, generated_source=generated,
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        replacements=[dict(before=a, after=b, expected_count=c) for a, b, c in replacements])


def configure(config):
    import admit_mhr_local_patch_depth as admission
    root = Path(config["output"]); spec = config["spec"]
    admission.FRAME = spec["frame"]
    admission.SOURCE = Path(spec["measured_control"])
    admission.CANDIDATES = root / "candidates"
    admission.OUT = root / "admission"
    admission.PRIOR = admission.CANDIDATES / "prior"
    admission.ARMS = [ARM]
    fn, proof = adapt(admission, "inputs", [
        ("Path('/mnt/data/dec5_phase30_early_texture_dynamic_150/request.json')", "MASK_PARENT", 1)],
        dict(MASK_PARENT=Path(spec["mask_parent_request"])))
    admission.inputs = fn
    return admission, proof


def build(config, destination):
    import build_mhr_silhouette_patch_candidates as builder
    spec = config["spec"]; prior = Path(spec["prior"])
    def rebound_read(path):
        value = read(path)
        if Path(path) == prior / "protocol.json":
            value = deepcopy(value)
            value.update(original_mesh=spec["mesh"], original_mesh_sha256=sha(spec["mesh"]))
        return value
    fn, proof = adapt(builder, "build", [
        ("inherited=Path('/mnt/data/dec5_mhr_measured_conformance/smooth100/fit.npz')",
         "inherited=INHERITED", 1), ("frame='001193'", "frame=FRAME", 1)],
        dict(PRIOR=prior, INHERITED=Path(spec["inherited_fit"]), FRAME=spec["frame"],
             ARM=ARM, read=rebound_read))
    receipt = Path(config["output"]) / (destination.name + "_adapter.json")
    require(not receipt.exists(), "Stage receipt already exists; retain failed workspace")
    save(receipt, dict(proof, config_sha256=sha(Path(config["output"]) / "config.json")))
    fn(destination)


def prefix_check(config):
    import open3d as o3d
    root = Path(config["output"]); old = o3d.io.read_triangle_mesh(config["spec"]["mesh"])
    ov, ot = np.asarray(old.vertices), np.asarray(old.triangles); result = {}
    for branch in ("strict", "interpolated"):
        folder = root / "admission" / ARM / branch
        r = read(folder / "result.json")
        mesh = o3d.io.read_triangle_mesh(str(folder / "mesh.ply"))
        v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        np.testing.assert_array_equal(v[:len(ov)], ov)
        np.testing.assert_array_equal(t[:len(ot)], ot)
        require(np.isfinite(v).all() and t.min() >= 0 and t.max() < len(v), "Invalid geometry")
        checks = r["rounds"][-1]["checks"]
        require(r["rounds"][-1]["removed_triangles"] == 0
                and len(checks) == len({(c["camera"], c["offset"]) for c in checks}) == 124
                and all(c["trusted_free_pixels"] == 0 for c in checks), "Incomplete final native guard")
        result[branch] = dict(added=len(t)-len(ot), mesh_sha256=sha(folder / "mesh.ply"),
                              original_vertices=len(ov), original_triangles=len(ot))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["init", "check", "build", "admit", "audit"])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--spec", type=Path, help="Required only for init; exact documented JSON fields")
    args = p.parse_args()
    if args.stage == "init":
        require(args.spec is not None and not args.output.exists(), "Init needs spec and a new output")
        config = inspect_inputs(read(args.spec), args.output)
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "config.json", config)
        print("initialized", config["frame"], len(config["input_hashes"]), "bindings", flush=True)
        return
    require(args.spec is None, "After init, use the pinned output config only")
    config = read(args.output / "config.json")
    require(config == inspect_inputs(config["spec"], args.output), "Inputs/config changed")
    admission, adapter = configure(config)
    if args.stage == "build":
        build(config, args.output / "candidates")
        return
    if args.stage == "check":
        _, rows, depths, masks, names, receipt = admission.inputs()
        require(receipt["depth_receipt"] == read(Path(config["spec"]["prior"]) / "protocol.json")["evidence"]["depth_receipt"],
                "Prior/admission measured depths differ")
        path = args.output / "input_check.json"
        require(not path.exists(), "Input-check receipt exists")
        save(path, dict(config_sha256=sha(args.output / "config.json"), input_adapter=adapter,
                        receipt=receipt, cameras=len(rows), native_depth_shapes=[list(d.shape) for d in depths],
                        mask_cameras=names, status="passed"))
        print("inputs checked", len(rows), flush=True)
        return
    check = read(args.output / "input_check.json")
    require(check["config_sha256"] == sha(args.output / "config.json"), "Missing/stale input check")
    if args.stage == "admit":
        save(args.output / "admission_adapter.json", dict(adapter, config_sha256=sha(args.output / "config.json")))
        admission.run()
        save(args.output / "geometry.json", prefix_check(config))
        return
    target = args.output / "audit.json"
    require(not target.exists(), "Audit exists; use a new output for reruns")
    build(config, args.output / "candidate_replay")
    for name in ("domain_evidence.npz", "proposal_evidence.npz"):
        a = np.load(args.output / "candidates" / ARM / name)
        b = np.load(args.output / "candidate_replay" / ARM / name)
        require(a.files == b.files, "Candidate array keys differ")
        for key in a.files: np.testing.assert_array_equal(a[key], b[key])
    require(sha(args.output / "candidates" / ARM / "local_raw.ply")
            == sha(args.output / "candidate_replay" / ARM / "local_raw.ply"), "Candidate mesh replay differs")
    # Import only after globals/inputs have been bound; one CLI stage per process.
    importlib.import_module("audit_mhr_local_patch_depth").main()
    geometry = prefix_check(config)
    save(target, dict(status="passed", config_sha256=sha(args.output / "config.json"),
         candidate_arrays_and_ply_exact=True, admission_audit_sha256=sha(admission.OUT / "audit.json"),
         geometry=geometry, input_adapter=adapter, visual_acceptance=False, production_modified=False,
         inventory={str(f.relative_to(args.output)): sha(f) for f in sorted(args.output.rglob("*")) if f.is_file()}))
    print("audit passed; visual acceptance still required", flush=True)


if __name__ == "__main__":
    main()

