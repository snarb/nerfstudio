#!/usr/bin/env python3
"""Resumable controller for the 50-frame DEC5 fixed-pose PatchMatch-TSDF campaign.

This is an opt-in orchestration layer.  It does not change any Nerfstudio or
single-frame model defaults.  Reconstruction, manual-GT-only face scoring,
and visual verdict publication are separate resumable states so no partial
frame can be mistaken for a complete result.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
from typing import Sequence

from colmap_patchmatch_tsdf_campaign_common import (
    CSV_FIELDS,
    EVAL_PHYSICAL_CAMERA,
    FRAME_COUNT,
    RECIPE,
    append_jsonl,
    atomic_csv,
    atomic_json,
    canonical_sha256,
    copy_or_validate_immutable,
    discover_frames,
    load_json,
    robust_initial_thresholds,
    sha256,
    stage_fixed_calibration_dataset,
    validate_hash_manifest,
    validate_source_frame,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CAMPAIGN_SCRIPTS = (
    "colmap_patchmatch_tsdf_campaign_common.py",
    "convert_exr_nerfstudio_to_jpeg.py",
    "run_colmap_patchmatch_tsdf_campaign.py",
    "run_colmap_patchmatch_tsdf.py",
    "run_colmap_patchmatch_tsdf_remote_worker.py",
    "score_colmap_patchmatch_tsdf_face.py",
    "audit_colmap_patchmatch_tsdf_campaign.py",
    "export_nerfstudio_colmap_model.py",
    "seed_colmap_from_nerfstudio.py",
    "build_colmap_patch_match_config.py",
    "import_colmap_mvs_depth_dataset.py",
    "fuse_depth_tsdf_mesh.py",
    "build_angular_camera_subset.py",
    "render_tsdf_mesh_depth.py",
    "render_mesh_image_blend.py",
)
EXPECTED_CALIBRATION_SHA256 = "79a91edfd8b441df1ff229839e2cc5f0b861ebe3fd626f40b04280d76a5f3900"
EXPECTED_COLMAP_MARKERS = ("COLMAP 3.13.0.dev0", "Commit 5509fffe", "with CUDA")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str], *, capture: bool = False, log: Path | None = None) -> subprocess.CompletedProcess:
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            return subprocess.run(command, check=True, text=True, stdout=stream, stderr=subprocess.STDOUT)
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def ssh_command(host: str, command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    return run(["ssh", host, shlex.join(command)], capture=capture)


def script_inventory() -> list[dict[str, object]]:
    result = []
    for name in CAMPAIGN_SCRIPTS:
        path = SCRIPT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append({"name": name, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    return result


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR.parent, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def create_request(args: argparse.Namespace) -> dict:
    frames = discover_frames(args.source_root)
    source_inventory = [validate_source_frame(path) | {"frame_id": path.name} for path in frames]
    request = {
        "schema_version": 1,
        "campaign": "dec5_5a3_first50_fixed_pose_colmap_patchmatch_tsdf",
        "created_at": now(),
        "created_on_host": socket.gethostname(),
        "source_root": str(args.source_root),
        "frame_count": len(frames),
        "ordered_frame_ids": [path.name for path in frames],
        "source_inventory": source_inventory,
        "calibration_template_source": str(args.calibration_template),
        "calibration_template_sha256": sha256(args.calibration_template),
        "calibration_mapping_key": "physical_camera",
        "eval_physical_camera": EVAL_PHYSICAL_CAMERA,
        "recipe": RECIPE,
        "jpeg_ingest": {
            "curve": "global_exposure_then_reinhard_then_srgb",
            "middle_gray": 0.18,
            "exposure_mode": "per-image",
            "exposure_percentile": 70.0,
            "jpeg_quality": 98,
            "jpeg_subsampling": "4:4:4",
            "temporary_one_frame_at_a_time": True,
        },
        "face_metric_protocol": {
            "domain": "display",
            "roi": "manual_polygon_on_heldout_gt_only",
            "candidate_surface_mask": False,
            "prediction_used_for_roi_selection": False,
            "person_or_face_segmentation": False,
            "face_psnr_definition": "exact selected RGB pixels",
            "face_ssim_lpips_definition": "tight face bbox; both images zero outside GT-defined mask",
            "lpips_network": "alex",
            "lpips_normalize": True,
            "old_000899_surface_mask_metrics_numerically_comparable": False,
        },
        "initial_visual_gate_frames": [path.name for path in frames[:3]],
        "visual_batch_boundaries": [3, 10, 20, 30, 40, 50],
        "initial_regression_signal_floors": {"face_psnr_drop": 1.0, "face_ssim_drop": 0.03, "face_lpips_rise": 0.05},
        "remote": {
            "host": args.remote_host,
            "scratch_root": str(args.remote_scratch_root),
            "python": str(args.remote_python),
            "colmap": str(args.remote_colmap),
            "gpu_index": args.gpu_index,
        },
        "controller_git_head": git_head(),
        "scripts": script_inventory(),
    }
    if request["calibration_template_sha256"] != EXPECTED_CALIBRATION_SHA256:
        raise ValueError("Primary calibration template does not match the expected SHA-256")
    request["request_sha256"] = canonical_sha256(request)
    return request


def initialize_campaign(args: argparse.Namespace) -> dict:
    request_path = args.output_root / "campaign_request.json"
    if request_path.is_file():
        request = load_json(request_path)
        if request.get("request_sha256") != canonical_sha256({key: value for key, value in request.items() if key != "request_sha256"}):
            raise ValueError("Existing campaign_request.json self-hash is invalid")
        for directory in (
            args.output_root / "frames", args.output_root / "contact_sheets",
            args.output_root / "config" / "face_polygons", args.output_root / ".work" / "frames",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return request
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"Refusing to initialize a non-empty output root without a request: {args.output_root}")
    request = create_request(args)
    if args.output_root.exists():
        args.output_root.rmdir()
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = args.output_root.with_name(f".{args.output_root.name}.init.tmp-{os.getpid()}")
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir()
    config = stage / "config"
    copy_or_validate_immutable(
        args.calibration_template,
        config / "calibration" / "transforms.json",
        EXPECTED_CALIBRATION_SHA256,
    )
    for row in request["scripts"]:
        copy_or_validate_immutable(Path(row["path"]), config / "code" / row["name"], row["sha256"])
    atomic_json(stage / "campaign_request.json", request)
    manifest = {
        "schema_version": 1,
        "request_sha256": request["request_sha256"],
        "status": "initialized",
        "ordered_frame_ids": request["ordered_frame_ids"],
        "frame_states": {frame_id: {"state": "pending"} for frame_id in request["ordered_frame_ids"]},
        "initial_baseline": None,
        "regression_thresholds": None,
        "visual_batches_completed": [],
        "updated_at": now(),
    }
    atomic_json(stage / "campaign_manifest.json", manifest)
    atomic_csv(stage / "metrics.csv", [])
    (stage / "frames").mkdir()
    (stage / "contact_sheets").mkdir()
    (stage / "config" / "face_polygons").mkdir()
    (stage / ".work" / "frames").mkdir(parents=True)
    os.replace(stage, args.output_root)
    print(f"campaign initialized request={request['request_sha256']} frames={len(request['ordered_frame_ids'])}")
    return request


def assert_request_matches_runtime(args: argparse.Namespace, request: dict) -> None:
    if str(args.source_root) != request["source_root"]:
        raise ValueError("Runtime source root differs from immutable campaign request")
    if str(args.remote_host) != request["remote"]["host"]:
        raise ValueError("Runtime remote host differs from immutable campaign request")
    runtime_remote = {
        "scratch_root": str(args.remote_scratch_root),
        "python": str(args.remote_python),
        "colmap": str(args.remote_colmap),
        "gpu_index": args.gpu_index,
    }
    for key, value in runtime_remote.items():
        if value != request["remote"][key]:
            raise ValueError(f"Runtime remote {key} differs from immutable campaign request")
    for row in request["scripts"]:
        path = SCRIPT_DIR / row["name"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Runtime script differs from immutable request: {path}")
    config_calibration = args.output_root / "config" / "calibration" / "transforms.json"
    if sha256(config_calibration) != request["calibration_template_sha256"]:
        raise ValueError("Permanent calibration config hash changed")


def update_frame_state(output_root: Path, frame_id: str, state: str, **extra: object) -> None:
    path = output_root / "campaign_manifest.json"
    manifest = load_json(path)
    manifest["frame_states"][frame_id] = {"state": state, "updated_at": now(), **extra}
    states = [row["state"] for row in manifest["frame_states"].values()]
    manifest["status"] = "complete" if states and all(value in {"pass", "fail"} for value in states) else "running"
    manifest["updated_at"] = now()
    atomic_json(path, manifest)


def remote_preflight(args: argparse.Namespace) -> dict:
    probe = ssh_command(args.remote_host, [str(args.remote_colmap), "-h"], capture=True)
    output = probe.stdout + probe.stderr
    missing = [marker for marker in EXPECTED_COLMAP_MARKERS if marker not in output]
    if missing:
        raise RuntimeError(f"Remote COLMAP build mismatch; missing {missing}")
    python_probe = ssh_command(args.remote_host, [str(args.remote_python), "-c", "import cv2,numpy,open3d,torch; print('python-ok')"], capture=True)
    gpu = ssh_command(args.remote_host, ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture=True)
    if gpu.stdout.strip():
        raise RuntimeError(f"GPU already has compute processes; refusing concurrent PatchMatch: {gpu.stdout.strip()}")
    disk = ssh_command(args.remote_host, ["df", "-Pk", str(args.remote_scratch_root.parent)], capture=True)
    return {"colmap_probe": "\n".join(output.splitlines()[:2]), "python_probe": python_probe.stdout.strip(), "gpu": gpu.stdout.strip(), "disk": disk.stdout.strip()}


def rsync(source: str, destination: str, *, delete: bool = False) -> None:
    # The shared /mnt/data mount allows payload writes but rejects rsync's
    # metadata-setting and dot-file rename path.  In-place content transfer is
    # safe here because every received file is subsequently checked against
    # the remote retained manifest before atomic publication.
    command = [
        "rsync", "-r", "--inplace", "--no-times", "--no-perms",
        "--omit-dir-times", "--partial", "--human-readable",
    ]
    if delete:
        command.append("--delete")
    command += [source, destination]
    run(command)


def copy_tree_content(source: Path, destination: Path) -> None:
    """Copy bytes and directory shape without unsupported shared-mount metadata."""

    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        else:
            raise ValueError(f"Adoption source contains unsupported non-file entry: {path}")


@contextlib.contextmanager
def controller_lock(output_root: Path):
    path = output_root / ".campaign_controller.lock"
    if path.is_file():
        try:
            token = path.read_text(encoding="utf-8").split()[0]
            pid = int(token.removeprefix("pid="))
            os.kill(pid, 0)
        except (OSError, ValueError, IndexError):
            path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Another campaign controller is active with pid={pid}: {path}")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"Another campaign controller may be active: {path}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()} started_at={now()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def compact_stage(log: Path) -> str:
    if not log.is_file():
        return "starting"
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if "stage=" in line or "frame=" in line:
            return line[-1000:]
    return lines[-1][-1000:] if lines else "starting"


def record_check(args: argparse.Namespace, frame_id: str, remote_frame: Path, process: subprocess.Popen, log: Path) -> None:
    try:
        command = [
            str(args.remote_python), "-c",
            (
                "import json,pathlib,subprocess,sys; p=pathlib.Path(sys.argv[1]); "
                "print(json.dumps({'geometric_maps':len(list(p.glob('pipeline/dense/stereo/depth_maps/**/*.geometric.bin'))),"
                "'photometric_maps':len(list(p.glob('pipeline/dense/stereo/depth_maps/**/*.photometric.bin'))),"
                "'pipeline_manifest':(p/'pipeline/pipeline_manifest.json').is_file(),"
                "'retained':(p/'retained/retained_manifest.json').is_file()}))"
            ),
            str(remote_frame),
        ]
        inventory = ssh_command(args.remote_host, command, capture=True).stdout.strip()
        gpu = ssh_command(
            args.remote_host,
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture=True,
        ).stdout.strip()
        disk = ssh_command(args.remote_host, ["df", "-Pk", str(args.remote_scratch_root)], capture=True).stdout.strip()
        error_scan = ssh_command(
            args.remote_host,
            ["bash", "-lc", f"grep -R -E -i 'OOM|out of memory|CUDA error' {shlex.quote(str(remote_frame / 'pipeline' / 'logs'))} 2>/dev/null | tail -20 || true"],
            capture=True,
        ).stdout.strip()
        status = "ok"
    except Exception as error:  # monitoring failure must be visible but does not kill an otherwise live worker
        inventory, gpu, disk, error_scan, status = "", "", "", repr(error), "check_error"
    append_jsonl(
        args.output_root / "campaign_checks.jsonl",
        {
            "timestamp": now(), "frame_id": frame_id, "controller_pid": os.getpid(),
            "ssh_worker_pid": process.pid, "ssh_worker_alive": process.poll() is None,
            "compact_stage": compact_stage(log), "remote_inventory": inventory,
            "gpu_processes": gpu, "remote_disk": disk, "oom_cuda_evidence": error_scan,
            "check_status": status,
        },
    )


def validate_staged_dataset(path: Path, source: Path, request: dict) -> bool:
    manifest = path / "staging_manifest.json"
    transforms = path / "transforms.json"
    if not manifest.is_file() or not transforms.is_file():
        return False
    audit = load_json(manifest)
    return (
        audit.get("source_dataset") == str(source.resolve())
        and audit.get("source_transforms_sha256") == sha256(source / "transforms.json")
        and audit.get("calibration_template_sha256") == request["calibration_template_sha256"]
        and audit.get("train_camera_count") == 62
        and audit.get("eval_camera_count") == 1
        and len(list((path / "images").glob("*.jpg"))) == 63
    )


def reconstruct_frame(args: argparse.Namespace, request: dict, frame_id: str) -> None:
    final = args.output_root / "frames" / frame_id
    work = args.output_root / ".work" / "frames" / frame_id
    retained = work / "retained"
    if final.is_dir():
        print(f"frame={frame_id} status=skipped-final")
        return
    if retained.is_dir() and (retained / "retained_manifest.json").is_file():
        validate_hash_manifest(retained, load_json(retained / "retained_manifest.json"))
        update_frame_state(args.output_root, frame_id, "reconstructed", retained=str(retained))
        print(f"frame={frame_id} status=skipped-reconstructed")
        return
    source = args.source_root / frame_id
    validate_source_frame(source)
    scratch = work / "scratch"
    converted = scratch / "jpeg65"
    staged = scratch / "staged63"
    work.mkdir(parents=True, exist_ok=True)
    update_frame_state(args.output_root, frame_id, "converting")
    converter = args.output_root / "config" / "code" / "convert_exr_nerfstudio_to_jpeg.py"
    run(
        [
            sys.executable, str(converter), "--input", str(source), "--output", str(converted),
            "--middle-gray", "0.18", "--exposure-mode", "per-image",
            "--exposure-percentile", "70", "--quality", "98", "--resume",
        ],
        log=work / "conversion.log",
    )
    if not validate_staged_dataset(staged, source, request):
        if staged.exists():
            quarantine = work / f"quarantine_staged_{int(time.time())}"
            os.replace(staged, quarantine)
        stage_fixed_calibration_dataset(
            source, converted, args.output_root / "config" / "calibration" / "transforms.json", staged
        )
    if not validate_staged_dataset(staged, source, request):
        raise RuntimeError("Staged 62/1 fixed-calibration dataset failed validation")

    preflight = remote_preflight(args)
    append_jsonl(args.output_root / "campaign_checks.jsonl", {"timestamp": now(), "frame_id": frame_id, "check_status": "preflight", **preflight})
    remote_base = args.remote_scratch_root / request["request_sha256"]
    remote_code = remote_base / "code"
    remote_frame = remote_base / frame_id / "attempt_0"
    ssh_command(args.remote_host, ["mkdir", "-p", str(remote_code), str(remote_frame)])
    rsync(str(args.output_root / "config" / "code") + "/", f"{args.remote_host}:{remote_code}/", delete=True)
    rsync(str(staged) + "/", f"{args.remote_host}:{remote_frame / 'data'}/", delete=True)
    remote_log = work / "remote_worker.log"
    command = [
        "ssh", args.remote_host,
        shlex.join(
            [
                "env", f"PYTHONPATH=/home/ubuntu/repos/nerfstudio:{remote_code}",
                f"CUDA_VISIBLE_DEVICES={args.gpu_index}", str(args.remote_python),
                str(remote_code / "run_colmap_patchmatch_tsdf_remote_worker.py"),
                "--frame-id", frame_id, "--data", str(remote_frame / "data"),
                "--workspace", str(remote_frame), "--colmap-bin", str(args.remote_colmap),
                "--gpu-index", "0",
            ]
        ),
    ]
    update_frame_state(args.output_root, frame_id, "remote_running", remote_workspace=str(remote_frame))
    with remote_log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, text=True, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            record_check(args, frame_id, remote_frame, process, remote_log)
            deadline = time.monotonic() + args.check_interval_seconds
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(min(30.0, deadline - time.monotonic()))
        return_code = process.wait()
    record_check(args, frame_id, remote_frame, process, remote_log)
    if return_code != 0:
        update_frame_state(args.output_root, frame_id, "failed_remote", remote_workspace=str(remote_frame), worker_log=str(remote_log))
        raise RuntimeError(f"Remote worker failed for {frame_id}; retained in quarantine workspace {remote_frame}")

    incoming = work / f".incoming.tmp-{os.getpid()}"
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir()
    rsync(f"{args.remote_host}:{remote_frame / 'retained'}/", str(incoming) + "/")
    retained_manifest = load_json(incoming / "retained_manifest.json")
    validate_hash_manifest(incoming, retained_manifest)
    remote_result = load_json(incoming / "remote_result.json")
    if frame_id == request["ordered_frame_ids"][0]:
        coverage = float(remote_result["depth_coverage_mean"])
        if abs(coverage - 0.38515681781411787) > 0.02 or int(remote_result["mesh_components"]) != 1:
            update_frame_state(args.output_root, frame_id, "failed_canary", depth_coverage_mean=coverage)
            raise RuntimeError(f"000899 depth/mesh canary failed: coverage={coverage}, components={remote_result['mesh_components']}")
    os.replace(incoming, retained)
    validate_hash_manifest(retained, retained_manifest)
    update_frame_state(args.output_root, frame_id, "reconstructed", retained=str(retained), remote_result=remote_result)
    shutil.rmtree(scratch)
    ssh_command(
        args.remote_host,
        [
            str(args.remote_python), "-c",
            "import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); assert 'lookcloser_dec5_5a3_patchmatch_tsdf_50_scratch' in str(p); shutil.rmtree(p)",
            str(remote_frame),
        ],
    )
    print(f"frame={frame_id} status=reconstructed")


def prior_accepted_rows(output_root: Path, ordered: list[str], frame_id: str) -> list[dict]:
    result = []
    for candidate in ordered[: ordered.index(frame_id)]:
        path = output_root / "frames" / candidate / "result.json"
        if path.is_file():
            row = load_json(path)
            if row.get("visual_status") == "pass" and row.get("metric_status") == "pass":
                result.append(row)
    return result[-5:]


def metric_status(metrics: dict, manifest: dict, previous: list[dict]) -> tuple[str, list[str]]:
    values = {key: float(metrics[key]) for key in ("face_psnr", "face_ssim", "face_lpips")}
    if not all(math.isfinite(value) for value in values.values()):
        return "fail_nonfinite", ["nonfinite"]
    thresholds = manifest.get("regression_thresholds")
    if thresholds is None or not previous:
        return "pass", []
    medians = {key: float(statistics.median(float(row[key]) for row in previous)) for key in values}
    reasons = []
    if values["face_psnr"] < medians["face_psnr"] - float(thresholds["face_psnr"]):
        reasons.append("face_psnr_drop")
    if values["face_ssim"] < medians["face_ssim"] - float(thresholds["face_ssim"]):
        reasons.append("face_ssim_drop")
    if values["face_lpips"] > medians["face_lpips"] + float(thresholds["face_lpips"]):
        reasons.append("face_lpips_rise")
    return ("regression_flag" if reasons else "pass"), reasons


def score_frame(args: argparse.Namespace, request: dict, frame_id: str) -> None:
    final = args.output_root / "frames" / frame_id
    work = args.output_root / ".work" / "frames" / frame_id
    retained = work / "retained"
    if final.is_dir():
        print(f"frame={frame_id} status=skipped-final")
        return
    validate_hash_manifest(retained, load_json(retained / "retained_manifest.json"))
    polygon = args.output_root / "config" / "face_polygons" / f"{frame_id}.json"
    if not polygon.is_file():
        raise FileNotFoundError(f"Manual held-out-GT polygon is required: {polygon}")
    if (retained / "metrics.json").is_file():
        print(f"frame={frame_id} status=skipped-scored")
        return
    temporary = work / f".score.tmp-{os.getpid()}"
    scorer = args.output_root / "config" / "code" / "score_colmap_patchmatch_tsdf_face.py"
    run(
        [
            sys.executable, str(scorer), "--frame-id", frame_id,
            "--prediction", str(retained / "render" / "eval_pred_0000.exr"),
            "--ground-truth", str(retained / "render" / "eval_gt_0000.exr"),
            "--face-polygons", str(polygon), "--output-dir", str(temporary),
            "--device", args.metric_device,
        ],
        log=work / "score.log",
    )
    metrics = load_json(temporary / "metrics.json")
    manifest_path = args.output_root / "campaign_manifest.json"
    manifest = load_json(manifest_path)
    previous = prior_accepted_rows(args.output_root, request["ordered_frame_ids"], frame_id)
    status, reasons = metric_status(metrics, manifest, previous)
    metrics["metric_status"] = status
    metrics["regression_reasons"] = reasons
    metrics["regression_reference_frame_ids"] = [row["frame_id"] for row in previous]
    final = args.output_root / "frames" / frame_id
    metrics["prediction"] = str(final / "render" / "eval_pred_0000.exr")
    metrics["ground_truth"] = str(final / "render" / "eval_gt_0000.exr")
    metrics["face_polygons"] = str(final / "metrics" / "face_polygons.json")
    metrics["review_crops"] = {
        key: str(final / "visual" / Path(value).name)
        for key, value in metrics["review_crops"].items()
    }
    atomic_json(temporary / "metrics.json", metrics)
    shutil.copyfile(polygon, temporary / "face_polygons.json")
    for path in temporary.iterdir():
        if path.name == "metrics.json":
            os.replace(path, retained / "metrics.json")
        elif path.name == "face_polygons.json":
            destination = retained / "metrics" / path.name
            destination.parent.mkdir()
            os.replace(path, destination)
        else:
            destination = retained / "visual" / path.name
            destination.parent.mkdir(exist_ok=True)
            os.replace(path, destination)
    temporary.rmdir()
    update_frame_state(args.output_root, frame_id, "scored", metric_status=status, regression_reasons=reasons)
    print(f"frame={frame_id} status=scored metric_status={status}")


def parse_bool(value: str) -> bool:
    if value.lower() in {"true", "yes", "1"}:
        return True
    if value.lower() in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def record_visual(args: argparse.Namespace, request: dict) -> None:
    frame_id = args.frame_id
    work = args.output_root / ".work" / "frames" / frame_id / "retained"
    if not (work / "metrics.json").is_file():
        raise RuntimeError("Frame must be reconstructed and scored before visual review")
    if args.visual_status == "pass" and (args.ear_artifact or args.lipstick_artifact):
        raise ValueError("A visual pass cannot declare an ear or lipstick artifact")
    receipt = {
        "schema_version": 1, "frame_id": frame_id, "reviewed_at": now(),
        "comparison": "heldout_gt_vs_prediction",
        "reviewed_crops": ["face_ear_hair", "ear_native", "lipstick_lips_hand", "actor_overview"],
        "background_ignored": True, "silhouette_holes_ignored": False,
        "visual_status": args.visual_status, "ear_artifact": args.ear_artifact,
        "lipstick_artifact": args.lipstick_artifact, "visual_notes": args.visual_notes,
    }
    atomic_json(work / "visual_review.json", receipt)
    update_frame_state(args.output_root, frame_id, "reviewed", visual_status=args.visual_status)
    print(f"frame={frame_id} status=reviewed verdict={args.visual_status}")


def rebuild_csv(output_root: Path, ordered: list[str]) -> None:
    rows = []
    for frame_id in ordered:
        path = output_root / "frames" / frame_id / "result.json"
        if path.is_file():
            payload = load_json(path)
            rows.append({key: payload[key] for key in CSV_FIELDS})
    atomic_csv(output_root / "metrics.csv", rows)


def finalize_frame(args: argparse.Namespace, request: dict, frame_id: str) -> None:
    final = args.output_root / "frames" / frame_id
    if final.is_dir():
        rebuild_csv(args.output_root, request["ordered_frame_ids"])
        print(f"frame={frame_id} status=skipped-final")
        return
    retained = args.output_root / ".work" / "frames" / frame_id / "retained"
    metrics = load_json(retained / "metrics.json")
    visual = load_json(retained / "visual_review.json")
    if visual["visual_status"] == "uncertain":
        raise RuntimeError("Uncertain visual review cannot be finalized; inspect the native crops again")
    remote = load_json(retained / "remote_result.json")
    status = "pass" if visual["visual_status"] == "pass" and metrics["metric_status"] != "fail_nonfinite" else "fail"
    result = {
        "schema_version": 1, "frame_id": frame_id,
        "source_dataset": str(args.source_root / frame_id),
        "eval_physical_camera": EVAL_PHYSICAL_CAMERA,
        "train_camera_count": 62, "texture_camera_count": remote["texture_camera_count"],
        "face_psnr": metrics["face_psnr"], "face_ssim": metrics["face_ssim"], "face_lpips": metrics["face_lpips"],
        "depth_coverage_mean": remote["depth_coverage_mean"], "depth_coverage_min": remote["depth_coverage_min"],
        "mesh_vertices": remote["mesh_vertices"], "mesh_triangles": remote["mesh_triangles"],
        "mesh_components": remote["mesh_components"],
        "render_path": str(final / "render" / "eval_pred_0000.png"),
        "mesh_path": str(final / "mesh" / "colmap_patchmatch_tsdf.ply"),
        "render_sha256": remote["render_sha256"], "mesh_sha256": remote["mesh_sha256"],
        "metric_status": metrics["metric_status"], "visual_status": visual["visual_status"],
        "ear_artifact": visual["ear_artifact"], "lipstick_artifact": visual["lipstick_artifact"],
        "visual_notes": visual["visual_notes"], "status": status,
        "request_sha256": request["request_sha256"], "remote_validation": remote,
        "retained_manifest_sha256": sha256(retained / "retained_manifest.json"),
    }
    atomic_json(retained / "result.json", result)
    os.replace(retained, final)
    work_parent = args.output_root / ".work" / "frames" / frame_id
    for leftover in list(work_parent.iterdir()):
        if leftover.is_file():
            leftover.unlink()
    if not any(work_parent.iterdir()):
        work_parent.rmdir()
    update_frame_state(args.output_root, frame_id, status, metric_status=metrics["metric_status"], visual_status=visual["visual_status"])
    manifest_path = args.output_root / "campaign_manifest.json"
    manifest = load_json(manifest_path)
    first_three = request["ordered_frame_ids"][:3]
    if frame_id == first_three[-1] and manifest.get("regression_thresholds") is None:
        baseline_rows = [load_json(args.output_root / "frames" / item / "result.json") for item in first_three]
        if not all(row["visual_status"] == "pass" for row in baseline_rows):
            raise RuntimeError("Initial baseline cannot be fixed until all first three frames visually pass")
        thresholds = robust_initial_thresholds(baseline_rows)
        manifest["initial_baseline"] = {
            "frame_ids": first_three,
            "metrics": [{key: row[key] for key in ("frame_id", "face_psnr", "face_ssim", "face_lpips")} for row in baseline_rows],
        }
        manifest["regression_thresholds"] = thresholds
        manifest["updated_at"] = now()
        atomic_json(manifest_path, manifest)
    rebuild_csv(args.output_root, request["ordered_frame_ids"])
    print(f"frame={frame_id} status={status} published={final}")


def selected_frame_ids(request: dict, values: list[str] | None, limit: int | None) -> list[str]:
    ordered = request["ordered_frame_ids"]
    selected = ordered if not values else values
    unknown = sorted(set(selected) - set(ordered))
    if unknown or len(selected) != len(set(selected)):
        raise ValueError(f"Invalid frame selection; unknown={unknown}")
    selected = [frame_id for frame_id in ordered if frame_id in set(selected)]
    return selected if limit is None else selected[:limit]


def adopt_verified_frame(args: argparse.Namespace, request: dict, frame_id: str) -> None:
    """Adopt a verified retained result when only local transport code changed.

    This is intentionally stricter than ordinary resume: every request field
    that can affect pixels or geometry must match, and the sole permitted
    script difference is this campaign controller itself.
    """

    previous_root = args.from_output_root.expanduser().resolve()
    previous_request = load_json(previous_root / "campaign_request.json")
    old_core = {
        key: value for key, value in previous_request.items()
        if key not in {"created_at", "created_on_host", "controller_git_head", "request_sha256", "scripts"}
    }
    new_core = {
        key: value for key, value in request.items()
        if key not in {"created_at", "created_on_host", "controller_git_head", "request_sha256", "scripts"}
    }
    if old_core != new_core:
        raise ValueError("Cannot adopt: a reconstruction-affecting campaign request field changed")
    old_scripts = {row["name"]: row["sha256"] for row in previous_request["scripts"]}
    new_scripts = {row["name"]: row["sha256"] for row in request["scripts"]}
    differences = sorted(name for name in set(old_scripts) | set(new_scripts) if old_scripts.get(name) != new_scripts.get(name))
    if differences != ["run_colmap_patchmatch_tsdf_campaign.py"]:
        raise ValueError(f"Cannot adopt: non-controller script hashes changed: {differences}")
    source = previous_root / ".work" / "frames" / frame_id / "retained"
    if not source.is_dir():
        source = previous_root / "frames" / frame_id
    validate_hash_manifest(source, load_json(source / "retained_manifest.json"))
    remote = load_json(source / "remote_result.json")
    if remote.get("frame_id") != frame_id or remote.get("validation_status") != "pass":
        raise ValueError("Cannot adopt an unvalidated or mismatched remote frame")
    destination = args.output_root / ".work" / "frames" / frame_id / "retained"
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy_tree_content(source, destination)
    validate_hash_manifest(destination, load_json(destination / "retained_manifest.json"))
    receipt = {
        "schema_version": 1,
        "frame_id": frame_id,
        "adopted_at": now(),
        "previous_request_sha256": previous_request["request_sha256"],
        "current_request_sha256": request["request_sha256"],
        "identical_reconstruction_request": True,
        "differing_script_hashes": differences,
        "reason": "local rsync metadata mode only; all reconstruction/scoring/audit code hashes identical",
    }
    atomic_json(destination / "adoption.json", receipt)
    update_frame_state(args.output_root, frame_id, "reconstructed", adopted=receipt, remote_result=remote)
    print(f"frame={frame_id} status=adopted previous_request={previous_request['request_sha256']}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/data/dec5_5a3_nerfstudio_exr_1920x1080"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50"))
    parser.add_argument("--calibration-template", type=Path, default=Path("/home/brans/lookcloser_temp/dec5_000899_full65_glomap_pose_intrinsics_jpeg/transforms.json"))
    parser.add_argument("--remote-host", default="ubuntu@dev3")
    parser.add_argument("--remote-scratch-root", type=Path, default=Path("/fsx/oregon/lookcloser_dec5_5a3_patchmatch_tsdf_50_scratch"))
    parser.add_argument("--remote-python", type=Path, default=Path("/home/ubuntu/anaconda3/envs/nerfstudio/bin/python"))
    parser.add_argument("--remote-colmap", type=Path, default=Path("/usr/local/bin/colmap"))
    parser.add_argument("--gpu-index", default="0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    init = subparsers.add_parser("init")
    add_common(init)
    init.add_argument("--preflight", action="store_true")
    reconstruct = subparsers.add_parser("reconstruct")
    add_common(reconstruct)
    reconstruct.add_argument("--frames", nargs="*")
    reconstruct.add_argument("--limit", type=int)
    reconstruct.add_argument("--check-interval-seconds", type=int, default=900)
    score = subparsers.add_parser("score")
    add_common(score)
    score.add_argument("--frames", nargs="*")
    score.add_argument("--limit", type=int)
    score.add_argument("--metric-device", choices=("auto", "cpu", "cuda"), default="auto")
    review = subparsers.add_parser("review")
    add_common(review)
    review.add_argument("--frame-id", required=True)
    review.add_argument("--visual-status", choices=("pass", "fail", "uncertain"), required=True)
    review.add_argument("--ear-artifact", type=parse_bool, required=True)
    review.add_argument("--lipstick-artifact", type=parse_bool, required=True)
    review.add_argument("--visual-notes", required=True)
    finalize = subparsers.add_parser("finalize")
    add_common(finalize)
    finalize.add_argument("--frames", nargs="*")
    finalize.add_argument("--limit", type=int)
    adopt = subparsers.add_parser("adopt")
    add_common(adopt)
    adopt.add_argument("--from-output-root", type=Path, required=True)
    adopt.add_argument("--frames", nargs="+", required=True)
    adopt.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    for name in ("source_root", "output_root", "calibration_template"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if hasattr(args, "check_interval_seconds") and not 60 <= args.check_interval_seconds <= 3600:
        parser.error("--check-interval-seconds must be in [60, 3600]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    request = initialize_campaign(args)
    assert_request_matches_runtime(args, request)
    if args.action == "init":
        if args.preflight:
            print(json.dumps(remote_preflight(args), indent=2, sort_keys=True))
        return 0
    if args.action == "review":
        record_visual(args, request)
        return 0
    frames = selected_frame_ids(request, args.frames, args.limit)
    with controller_lock(args.output_root):
        for frame_id in frames:
            if args.action == "reconstruct":
                reconstruct_frame(args, request, frame_id)
            elif args.action == "score":
                score_frame(args, request, frame_id)
            elif args.action == "finalize":
                finalize_frame(args, request, frame_id)
            elif args.action == "adopt":
                adopt_verified_frame(args, request, frame_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
