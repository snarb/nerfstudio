"""One reviewed native HD/4K chunk; exits and releases its cache before evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
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


def validate_job(job):
    release_path = Path(job["root_release"])
    release = json.loads(release_path.read_text())
    restore_only = job["restore_only"]
    if not (release.get("production_launch_authorized") is True or
            (restore_only and release.get("restoration_probe_authorized") is True)):
        raise ValueError("Reviewed production or restoration-only release required")
    if sha(release_path) != job["root_release_sha256"]:
        raise ValueError("Root release changed")
    if job["arm"] not in release["arms"]:
        raise ValueError("Unreviewed resolution arm")
    arm = release["arms"][job["arm"]]
    if job["train_scale"] != arm["train_scale"]:
        raise ValueError("TRAIN scale changed")
    expected = 0 if restore_only else release["interval"]
    if job["target_step"] - job["source_step"] != expected:
        raise ValueError("A chunk must contain exactly one reviewed interval")
    if job["source_step"] < release["anchor_step"] or (job["source_step"] - release["anchor_step"]) % release["interval"]:
        raise ValueError("Source is outside the paired gate cadence")
    output = Path(job["output"]).resolve()
    if Path(release["output"]).resolve() not in output.parents:
        raise ValueError("Chunk output outside reviewed root")
    source = Path(job["source_checkpoint"]).resolve()
    if job["source_step"] == release["anchor_step"]:
        if str(source) != str(Path(release["source"]["checkpoint"]).resolve()) or job["source_checkpoint_sha256"] != release["source"]["checkpoint_sha256"]:
            raise ValueError("Initial ancestry must be the original selected HD checkpoint")
    elif (Path(release["output"]) / job["arm"] / "chunks").resolve() not in source.parents:
        raise ValueError("Continuation source belongs to another arm or technical probe")
    if job["bound_inputs"].get(job["source_checkpoint"]) != job["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint hash binding is inconsistent")
    if Path(sys.executable).resolve() != Path(release["python"]).resolve():
        raise ValueError("Historical Python runtime required")
    for bindings in [release["bound_files"], job["bound_inputs"]]:
        for path, expected_hash in bindings.items():
            if sha(path) != expected_hash:
                raise ValueError("Bound file changed: " + path)
    return release


def main(args):
    job = json.loads(args.job.read_text())
    release = validate_job(job)
    for path in reversed(release["python_path"]):
        sys.path.insert(0, path)
    import numpy as np
    import torch
    import yaml
    from nerfstudio.engine.callbacks import TrainingCallbackLocation
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    from nerfstudio.scripts.train import _set_random_seed
    from nerfstudio.utils import writer
    if torch.__version__ != release["torch_version"] or torch.version.cuda != release["cuda_version"]:
        raise ValueError("Historical Torch/CUDA runtime changed")
    torch.set_num_threads(4)
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=False)
    source = Path(job["source_checkpoint"])
    saved = torch.load(source, map_location="cpu", weights_only=False)
    assert int(saved["step"]) == job["source_step"]
    runtime = json.loads(Path(job["source_runtime"]).read_text())
    config = yaml.load(Path(job["config"]).read_text(), Loader=yaml.Loader)
    config.load_checkpoint = source
    config.output_dir = output / "run"
    config.pipeline.final_grid_artifact_dir = output / "grids"
    config.save_only_latest_checkpoint = False
    _set_random_seed(config.machine.seed)
    trainer = config.setup(local_rank=0, world_size=1)
    trainer.setup()
    pipeline = trainer.pipeline

    def equal(left, right):
        if torch.is_tensor(left):
            return torch.equal(left.detach().cpu(), right.detach().cpu())
        if isinstance(left, np.ndarray):
            return np.array_equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right

    def domains():
        return dict(pipeline=pipeline.state_dict(),
                    optimizers={key: value.state_dict() for key, value in trainer.optimizers.optimizers.items()},
                    schedulers={key: value.state_dict() for key, value in trainer.optimizers.schedulers.items()},
                    scalers=trainer.grad_scaler.state_dict())

    restored = {key: equal(value, saved[key]) for key, value in domains().items()}
    assert all(restored.values()), restored
    for split, key in [("train", "cached_train_image_idx"), ("eval", "eval_cached_image_idx")]:
        cache = getattr(pipeline.datamanager, split + "_image_dataloader").cached_collated_batch
        actual = cache["image_idx"].detach().cpu().tolist()
        wanted = runtime[key]
        lookup = {int(value): index for index, value in enumerate(actual)}
        order = torch.tensor([lookup[value] for value in wanted])
        for key, value in list(cache.items()):
            if torch.is_tensor(value) and value.ndim and value.shape[0] == len(actual):
                cache[key] = value[order.to(value.device)]
        assert cache["image_idx"].detach().cpu().tolist() == wanted
    pipeline.dynamic_num_rays_per_batch = runtime["dynamic_num_rays_per_batch"]
    pipeline._update_pixel_samplers()
    grid_path = Path(job["source_grid"])
    with np.load(grid_path) as grid:
        object.__setattr__(pipeline, "_actual_core", torch.from_numpy(grid["core"]).to(pipeline.device))
        object.__setattr__(pipeline, "_actual_active", torch.from_numpy(grid["active"]).to(pipeline.device))
    metadata = json.loads(grid_path.with_suffix(".json").read_text())
    assert metadata["checkpoint_sha256"] == job["source_checkpoint_sha256"]
    assert metadata["grid_archive_sha256"] == sha(grid_path)
    object.__setattr__(pipeline, "_actual_threshold", metadata["gpu_native_threshold"])
    object.__setattr__(pipeline, "_last_core_step", metadata["core_update_step"])
    assert torch.equal(pipeline.model.occupancy_grid.binaries, pipeline._actual_active)
    cache = pipeline.datamanager.train_image_dataloader.cached_collated_batch
    assert tuple(cache["image"].shape) == (63, 1012 * job["train_scale"], 1920 * job["train_scale"], 3)
    assert cache["image"].dtype == torch.float32 and not cache["image"].is_cuda
    _restore_rng_state(saved["rng_state"])
    assert equal(_capture_rng_state(), saved["rng_state"])
    trainer._loaded_rng_state = None
    assert trainer._start_step == job["source_step"] + 1

    def adam():
        counts = [int(value["step"]) for value in trainer.optimizers.optimizers["fields"].state.values()]
        assert counts and min(counts) == max(counts)
        return max(counts)

    initial_adam = adam()
    atomic(output / "restoration.json", dict(time=time.time(), pid=os.getpid(), domains_equal=restored,
           RNG_exact=True, runtime_exact=True, source_step=job["source_step"], initial_adam=initial_adam,
           dynamic_rays=pipeline.dynamic_num_rays_per_batch, train_scale=job["train_scale"]))
    if not job["restore_only"]:
        del saved
        for step in range(trainer._start_step, job["target_step"] + 1):
            trainer.step = step
            pipeline.train()
            for callback in trainer.callbacks:
                callback.run_callback_at_location(step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
            value, terms, metrics = trainer.train_iteration(step)
            if not bool(torch.isfinite(value)):
                raise RuntimeError("Nonfinite training scalar; no further updates")
            for callback in trainer.callbacks:
                callback.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
            trainer.eval_iteration(step)
            writer.write_out_storage()
            if step % 100 == 0 or step == job["target_step"]:
                progress = dict(time=time.time(), pid=os.getpid(), arm=job["arm"], step=step,
                                attempts=step-job["source_step"], actual_updates=adam()-initial_adam,
                                adam_updates=adam(), dynamic_next_rays=pipeline.dynamic_num_rays_per_batch,
                                lr=trainer.optimizers.optimizers["fields"].param_groups[0]["lr"],
                                scale=trainer.grad_scaler.get_scale(),
                                GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                atomic(output / "progress.json", progress)
                with (output / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(progress) + "\n")
    else:
        assert all(equal(value, saved[key]) for key, value in domains().items())
        assert equal(_capture_rng_state(), saved["rng_state"])
        del saved
    trainer._train_batch_prefetch_barrier()
    checkpoint = output / ("step-%09d.ckpt" % job["target_step"])
    state = dict(step=job["target_step"], **domains(), rng_state=_capture_rng_state(),
                 checkpoint_load_audit=trainer.checkpoint_load_audit)
    temporary = checkpoint.with_suffix(".ckpt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    roundtrip = {key: equal(state[key], loaded[key]) for key in ["pipeline", "optimizers", "schedulers", "scalers", "rng_state"]}
    assert all(roundtrip.values()), roundtrip
    del loaded
    exported = pipeline.export_grid_for_checkpoint(checkpoint, job["target_step"])
    assert equal(_capture_rng_state(), state["rng_state"])
    runtime_new = dict(dynamic_num_rays_per_batch=pipeline.dynamic_num_rays_per_batch,
                       cached_train_image_idx=cache["image_idx"].detach().cpu().tolist(),
                       eval_cached_image_idx=pipeline.datamanager.eval_image_dataloader.cached_collated_batch["image_idx"].detach().cpu().tolist(),
                       native_core_update_step=pipeline._last_core_step)
    atomic(output / "runtime.json", runtime_new)
    attempts = job["target_step"] - job["source_step"]
    receipt = dict(kind="PAIRED_NATIVE_RESOLUTION_CHUNK", arm=job["arm"], train_scale=job["train_scale"],
                   restore_only=job["restore_only"], source_step=job["source_step"], final_step=job["target_step"],
                   source_checkpoint=str(source), source_checkpoint_sha256=sha(source),
                   output_checkpoint=str(checkpoint), output_checkpoint_sha256=sha(checkpoint),
                   output_runtime=str(output / "runtime.json"), output_runtime_sha256=sha(output / "runtime.json"),
                   output_grid=exported["grid_archive"], output_grid_sha256=exported["grid_archive_sha256"],
                   initial_adam=initial_adam, final_adam=adam(), attempts=attempts,
                   actual_updates=adam()-initial_adam, AMP_skips=attempts-(adam()-initial_adam),
                   restored_domain_equality=restored, serialization_roundtrip=roundtrip,
                   job_sha256=sha(args.job), release_sha256=sha(job["root_release"]),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   evaluator_not_run_in_worker=True, time=time.time())
    assert 0 <= receipt["actual_updates"] <= attempts
    atomic(output / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ["arm", "final_step", "attempts", "actual_updates"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    main(parser.parse_args())
