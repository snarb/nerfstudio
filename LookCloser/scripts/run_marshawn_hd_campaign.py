#!/usr/bin/env python3
"""Manifest-driven HD AABB/occupancy comparison; each cold run is capped at 22000 updates.

init --data DATA --boxes BOXES.json --root ROOT builds 12 immutable specifications.
run --root ROOT --variant scene_tight_halo0 --gate REVIEW.json runs one specification.
Gate JSON: {"approved": true, "transforms_sha256": "...", "boxes_sha256": "..."}.
Approval records a completed internal calibration/bounds visual review, not a quality claim.

Boxes use prepared camera coordinates, with one normalization for all variants:
{"coordinate_system":"prepared_world", "scopes":{"scene":{"tight":[[min],[max]],
"medium":[[min],[max]],"loose":[[min],[max]]},"subject":{...}}}.
Images remain unmasked RGB (except valid undistortion-border masks). A tighter box
alone cannot certify empty background or hidden geometry. Use the companion grid audit.

The serialized parser target is imported from this module during ns-eval. Keep this
script's directory on PYTHONPATH (the controller sets it for every child process).
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

MODULE = "run_marshawn_hd_campaign"
SCRIPT = Path(__file__).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_boxes(value):
    if value.get("coordinate_system") != "prepared_world":
        raise ValueError("boxes must explicitly use prepared_world coordinates")
    for scope in ("scene", "subject"):
        previous = None
        for size in ("tight", "medium", "loose"):
            box = value["scopes"][scope][size]
            if len(box) != 2 or any(len(v) != 3 for v in box):
                raise ValueError("each AABB must contain two XYZ corners")
            if any(not math.isfinite(x) for v in box for x in v):
                raise ValueError("AABB contains nonfinite values")
            if any(box[0][k] >= box[1][k] for k in range(3)):
                raise ValueError("AABB must have positive extent")
            if previous and any(box[0][k] > previous[0][k] or box[1][k] < previous[1][k] for k in range(3)):
                raise ValueError("tight/medium/loose AABBs must be nested")
            previous = box
    return value


def validate_seed_policy(policy):
    """Check bound inputs before loading any frequency values into the live grid."""
    if not isinstance(policy, dict):
        raise ValueError('frequency initialization policy must be explicitly configured')
    if policy.get('policy') == 'zero_explicit':
        return None
    if policy.get('policy') != 'paper_train_sfm':
        raise ValueError('unknown frequency initialization policy')
    manifest_path = Path(policy['manifest'])
    if sha(manifest_path) != policy['manifest_sha256']:
        raise ValueError('frequency seed manifest hash changed')
    manifest = read(manifest_path)
    if (manifest.get('kind') != 'frequency_only_train_observed_sfm_seed'
            or manifest.get('uses_eval_rgb') is not False
            or manifest.get('occupancy_admission') is not False
            or manifest.get('frequency_halo_radius') != 0):
        raise ValueError('seed must use train-only frequency evidence without occupancy admission or halo')
    for key in ('transforms_sha256', 'normalization_sha256', 'boxes_sha256'):
        if manifest.get(key) != policy[key]:
            raise ValueError(f'frequency seed source binding mismatch: {key}')
    data = Path(policy['data'])
    if (sha(data/'transforms.json') != policy['transforms_sha256']
            or sha(data/'preparation.json') != policy['normalization_sha256']
            or sha(policy['boxes']) != policy['boxes_sha256']):
        raise ValueError('frequency seed dataset, normalization or bounds changed')
    entry = manifest['variants'][policy['variant']]
    seed_path = Path(policy['seed'])
    if (seed_path.resolve() != (manifest_path.parent/entry['file']).resolve()
            or entry['sha256'] != policy['seed_sha256'] or sha(seed_path) != policy['seed_sha256']):
        raise ValueError('frequency seed tensor hash or variant file mismatch')
    if entry['aabb'] != policy['aabb']:
        raise ValueError('frequency seed variant AABB differs from the experiment')
    metadata = read(data/'transforms.json')
    training = set(metadata['train_filenames'])
    frames = [frame for frame in metadata['frames'] if frame['file_path'] in training]
    bindings = manifest.get('map_bindings', {})
    if (len(frames) != manifest['train_cameras']
            or set(bindings) != {frame['physical_camera'] for frame in frames}):
        raise ValueError('frequency seed camera identities differ from the train split')
    for frame in frames:
        image = data/frame['file_path']
        path = data/'lookcloser_frequencies'/(image.stem+'.pt')
        record = bindings[frame['physical_camera']]
        if (sha(image) != record['source_sha256'] or sha(path) != record['map_sha256']
                or sha(path.with_suffix('.json')) != record['metadata_sha256']):
            raise ValueError(f'frequency seed map/source bytes changed: {image.name}')
    projection = SCRIPT.parents[2]/'nerfstudio/utils/lookcloser_projection.py'
    if manifest.get('projection_source_sha256') != sha(projection):
        raise ValueError('frequency projection implementation differs from the seed generator')
    return manifest


def seed_policy(manifest_path, variant, aabb, data, boxes):
    """Bind one reviewed sparse seed to its exact dataset and candidate bounds."""
    manifest_path = Path(manifest_path).resolve()
    manifest = read(manifest_path)
    entry = manifest['variants'][variant]
    policy = dict(policy='paper_train_sfm', manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
        seed=str((manifest_path.parent/entry['file']).resolve()), seed_sha256=entry['sha256'],
        variant=variant, aabb=aabb, data=str(Path(data).resolve()), boxes=str(Path(boxes).resolve()),
        transforms_sha256=sha(Path(data)/'transforms.json'),
        normalization_sha256=sha(Path(data)/'preparation.json'), boxes_sha256=sha(boxes))
    validate_seed_policy(policy)
    return policy


def load_frequency_seed(policy):
    """Validate the sparse NPZ entirely on CPU; never change its native cell mapping."""
    import numpy as np
    validate_seed_policy(policy)
    with np.load(policy['seed'], allow_pickle=False) as source:
        for key in ('grid_resolution', 'num_levels', 'min_res', 'max_res'):
            if source[key].shape != () or not np.isfinite(source[key]):
                raise ValueError('frequency seed schedule metadata must contain finite scalar values')
        if any(not np.issubdtype(source[key].dtype, np.integer) for key in ('grid_resolution', 'num_levels')):
            raise ValueError('frequency seed dimensions must be integer scalars')
        indices = source['indices_xyz'].copy()
        levels = source['levels'].copy()
        aabb = source['aabb'].copy()
        resolution = int(source['grid_resolution'])
        num_levels = int(source['num_levels'])
        minimum, maximum = float(source['min_res']), float(source['max_res'])
    if (resolution != 128 or num_levels != 16 or minimum != 16 or maximum != 8192
            or aabb.shape != (2, 3) or not np.issubdtype(aabb.dtype, np.floating) or not np.isfinite(aabb).all()
            or not np.array_equal(aabb, np.asarray(policy['aabb'], dtype=aabb.dtype))):
        raise ValueError('seed dimensions, frequency schedule or AABB mismatch')
    if (indices.dtype != np.int64 or indices.ndim != 2 or indices.shape[1] != 3 or len(indices) == 0
            or levels.dtype != np.float32 or levels.shape != (len(indices),)):
        raise ValueError('invalid sparse frequency seed index/level shape or dtype')
    # Native FP32 mapping clamps to resolution-1.001 before integer conversion;
    # its last reachable plane is126 for128 cells. Do not seed unreachable127.
    if (np.any(indices < 0) or np.any(indices > resolution-2)
            or len(np.unique(indices, axis=0)) != len(indices)
            or not np.isfinite(levels).all() or np.any(levels < 0) or np.any(levels >= num_levels)
            or np.any(levels != np.rint(levels))):
        raise ValueError('duplicate/unreachable indices or invalid discrete frequency levels')
    return dict(indices=indices, levels=levels, aabb=aabb, resolution=resolution,
                num_levels=num_levels, min_res=minimum, max_res=maximum)


def initialize_frequency_grid(grid, policy):
    """Cold initialization only; checkpoint state loading subsequently overrides it."""
    import torch
    if torch.count_nonzero(grid.grid):
        raise ValueError('cold frequency grid must initialize to zeros before applying its policy')
    if not isinstance(policy, dict):
        raise ValueError('an explicit frequency initialization policy is required')
    if policy.get('policy') == 'zero_explicit':
        return dict(policy='zero_explicit', cells=0)
    seed = load_frequency_seed(policy)
    if (grid.grid.dtype != torch.float32 or tuple(grid.grid.shape) != (seed['resolution'],)*3 or grid.num_levels != seed['num_levels']
            or grid.min_res != seed['min_res'] or grid.max_res != seed['max_res']):
        raise ValueError('live frequency grid dimensions or schedule differ from the seed')
    live_aabb = torch.stack((grid.aabb_min_buf, grid.aabb_max_buf))
    if live_aabb.dtype != torch.float32:
        raise ValueError('frequency seed requires the validated native FP32 grid mapping')
    expected = torch.as_tensor(seed['aabb'], device=live_aabb.device, dtype=live_aabb.dtype)
    if not torch.equal(live_aabb, expected):
        raise ValueError('live frequency grid AABB differs from the seed')
    indices = torch.as_tensor(seed['indices'], device=grid.grid.device, dtype=torch.long)
    levels = torch.as_tensor(seed['levels'], device=grid.grid.device, dtype=grid.grid.dtype)
    with torch.no_grad():
        grid.grid[indices[:, 0], indices[:, 1], indices[:, 2]] = levels
    return dict(policy='paper_train_sfm', cells=len(indices), manifest_sha256=policy['manifest_sha256'],
                seed_sha256=policy['seed_sha256'], occupancy_modified=False, frequency_halo_radius=0)


def select_checkpoint(metrics_path, model_dir):
    """Only exact saved, finite all-view metric rows are eligible; never use loss."""
    with Path(metrics_path).open() as stream:
        rows = list(csv.DictReader(stream))
    candidates = []
    for row in rows:
        try:
            step = int(row["step"])
            metrics = tuple(float(row[k]) for k in ("eval_all_psnr", "eval_all_ssim", "eval_all_lpips"))
        except (ValueError, KeyError, TypeError):
            continue
        checkpoint = Path(model_dir) / f"step-{step:09d}.ckpt"
        if 0 <= step < 22000 and all(map(math.isfinite, metrics)) and checkpoint.is_file():
            candidates.append((checkpoint, metrics))
    if not candidates:
        raise RuntimeError("no exact checkpoint with finite full-eval PSNR/SSIM/LPIPS; selection fails closed")
    maximum = max(v[1][0] for v in candidates)
    winner = min((v for v in candidates if maximum - v[1][0] <= .07 + 1e-12),
                 key=lambda v: (v[1][2], -v[1][0], int(v[0].stem.split('-')[-1])))
    return winner[0], "all_eval_psnr_lpips_tie_0.07_exact_checkpoint"


def audit_inputs(spec):
    """Fail before CUDA training if a camera/map is absent or its raster contract differs."""
    import torch
    from PIL import Image
    data = Path(spec["data"])
    metadata = read(data / "transforms.json")
    training = set(metadata["train_filenames"])
    evaluation = set(metadata["val_filenames"])
    if training & evaluation or not training or not evaluation:
        raise ValueError("invalid explicit train/eval split")
    frames = {f["file_path"]: f for f in metadata["frames"]}
    if set(frames) != training | evaluation:
        raise ValueError("split does not cover exactly the prepared images")
    expected_maps = {Path(name).stem for name in training}
    if {path.stem for path in (data / 'lookcloser_frequencies').glob('*.pt')} != expected_maps:
        raise ValueError("frequency inventory must match exactly the complete train split")
    records = []
    for name in sorted(frames):
        frame = frames[name]
        image = data / name
        with Image.open(image) as pixels:
            if pixels.size != (frame["w"], frame["h"]):
                raise ValueError(f"image/intrinsics raster mismatch: {name}")
        record = dict(image=name, image_sha256=sha(image))
        if "mask_path" in frame:
            mask = data / frame["mask_path"]
            with Image.open(mask) as pixels:
                if pixels.size != (frame["w"], frame["h"]):
                    raise ValueError(f"mask raster mismatch: {name}")
            record["mask_sha256"] = sha(mask)
        frequency = data / "lookcloser_frequencies" / (image.stem + ".pt")
        if name in training:
            fit = read(frequency.with_suffix(".json"))
            expected = dict(min_res=16, max_res=8192, n_levels=16,
                            patch_size=8, stride=8, value_type="scalar_resolution")
            if any(fit.get(k) != v for k, v in expected.items()):
                raise ValueError(f"frequency schedule mismatch: {frequency}")
            if fit.get("image_shape") != [frame["h"], frame["w"]] or fit.get("crop_coords_xywh") is not None:
                raise ValueError(f"frequency raster/crop mismatch: {frequency}")
            if fit.get('image') != image.name:
                raise ValueError(f"frequency camera identity mismatch: {frequency}")
            values = torch.load(frequency, map_location="cpu", weights_only=True)
            shape = ((frame["h"] - 8) // 8 + 1, (frame["w"] - 8) // 8 + 1)
            if tuple(values.shape) != shape or not torch.isfinite(values).all() or not ((values >= 16 - 1e-4) & (values <= 8192 + .01)).all():
                raise ValueError(f"invalid frequency tensor: {frequency}")
            record.update(frequency_sha256=sha(frequency), frequency_metadata_sha256=sha(frequency.with_suffix(".json")))
        elif frequency.exists():
            raise ValueError(f"held-out image has a training frequency map: {name}")
        records.append(record)
    return dict(train_count=len(training), eval_count=len(evaluation), images=records)


def __getattr__(name):
    # Lazy so manifest and selector utilities work without torch/CUDA installed.
    if name == "MarshawnPipeline":
        import torch
        from PIL import Image
        import numpy as np
        from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline

        class MarshawnPipeline(LookCloserPipeline):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.marshawn_frequency_seed_receipt = initialize_frequency_grid(
                    self.model.freq_grid, self.config.marshawn_frequency_initialization)
                self._marshawn_border_masks = None
                self.dec5_frequency_sampler = self._frequency_pixels

            def _frequency_pixels(self, indices):
                dataset = self.datamanager.train_dataset
                if self._marshawn_border_masks is None:
                    self._marshawn_border_masks = []
                    for path in dataset._dataparser_outputs.mask_filenames or []:
                        with Image.open(path) as mask:
                            self._marshawn_border_masks.append(torch.from_numpy(np.asarray(mask.convert('L')).copy() > 0))
                cameras = dataset.cameras
                height = cameras.height[indices].reshape(-1).cpu()
                width = cameras.width[indices].reshape(-1).cpu()
                y = (torch.rand(len(indices)) * height).long()
                x = (torch.rand(len(indices)) * width).long()
                valid = torch.ones(len(indices), dtype=torch.bool)
                if self._marshawn_border_masks:
                    for camera in indices.unique().tolist():
                        chosen = indices == camera
                        valid[chosen] &= self._marshawn_border_masks[camera][y[chosen], x[chosen]]
                # Existing normalized-Jacobian projection uses opacity>=.9 whenever
                # a support sampler is supplied. Here support means valid image border,
                # not a semantic foreground mask or independent geometric truth.
                return y, x, valid

            def get_train_loss_dict(self, step):
                start, end, lower, upper = self.config.marshawn_distortion_schedule
                fraction = min(1., max(0., (step - start) / (end - start)))
                self.model.config.distortion_loss_mult = lower + (upper - lower) * fraction
                return super().get_train_loss_dict(step)

        MarshawnPipeline.__module__ = MODULE
        MarshawnPipeline.__qualname__ = name
        globals()[name] = MarshawnPipeline
        return MarshawnPipeline
    if name != "MarshawnDataParser":
        raise AttributeError(name)
    import torch
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio
    from nerfstudio.data.scene_box import SceneBox

    class MarshawnDataParser(Nerfstudio):
        def _generate_dataparser_outputs(self, split="train"):
            if self.config.auto_scale_poses or self.config.center_method != "none" or self.config.orientation_method != "none":
                raise RuntimeError("prepared_world requires identity parser normalization")
            outputs = super()._generate_dataparser_outputs(split)
            outputs.scene_box = SceneBox(aabb=torch.tensor(self.config.marshawn_aabb, dtype=torch.float32))
            outputs.metadata["marshawn_explicit_aabb"] = self.config.marshawn_aabb
            return outputs

    MarshawnDataParser.__module__ = MODULE
    MarshawnDataParser.__qualname__ = name
    globals()[name] = MarshawnDataParser
    return MarshawnDataParser


def quiet_arguments(spec):
    return ["--data", spec["data"], "--output-dir", spec["output_dir"],
        "--experiment-name", spec["name"], "--timestamp", "cold_seed42",
        "--center-method", "none", "--orientation-method", "none", "--scale-factor", "1.0",
        "--scene-scale", "1.0", "--eval-mode", "filename", "--seed", "42",
        "--max-num-iterations", str(spec["max_updates"]), "--step-interval", "4000",
        "--train-num-rays-per-batch", "4096", "--max-res", "8192",
        "--grid-resolution", "128", "--stable-occupancy-reduction",
        "--adaptive-warmup-steps", "4096", "--occupancy-warmup-steps", "4096",
        "--occupancy-binary-warmup-steps", "4096", "--max-steps-per-ray", "4096",
        "--adaptive-coarse-step-size", str(spec["coarse_step"]),
        "--occupancy-dilation-radius", str(spec["radius"]),
        "--supervision-interval-seconds", "1800", "--poll-seconds", "30",
        "--keep-all-checkpoints", "--eval-checkpoint", "best", "--stop-on-no-improve",
        "--no-update-summary", "--no-artifact-score", "--no-artifact-roi-score"]


def initialize(args):
    seed_manifest = getattr(args, 'frequency_seed_manifest', None)
    zero_init = bool(getattr(args, 'zero_frequency_init', False))
    if bool(seed_manifest) == zero_init:
        raise ValueError('choose exactly one of --frequency-seed-manifest or --zero-frequency-init')
    boxes = validate_boxes(read(args.boxes))
    data = args.data.resolve()
    transforms = data / "transforms.json"
    metadata = read(transforms)
    if metadata.get("orientation_override", "none") != "none":
        raise ValueError("prepared transforms must not override identity orientation")
    normalization_sha256 = sha(data / "preparation.json")
    if boxes.get("normalization_sha256") != normalization_sha256:
        raise ValueError("bounds and prepared dataset must share the exact normalization receipt")
    root = args.root.resolve()
    manifest_path = root / "campaign.json"
    if manifest_path.exists():
        raise FileExistsError(f"campaign already exists: {manifest_path}")
    if not 4097 <= args.max_updates <= 22000:
        raise ValueError("max_updates must be in [4097,22000]")
    variants = []
    common_span = max(boxes['scopes']['scene']['loose'][1][k] - boxes['scopes']['scene']['loose'][0][k] for k in range(3))
    if not math.isfinite(args.coarse_step) or args.coarse_step <= 0:
        raise ValueError("coarse step must be finite and positive")
    runtime_sources = source_hashes()
    for scope in ("scene", "subject"):
        for size in ("tight", "medium", "loose"):
            box = boxes["scopes"][scope][size]
            for radius in (0, 1):
                name = f"{scope}_{size}_halo{radius}"
                spec = dict(name=name, scope=scope, size=size, radius=radius,
                    aabb=box, data=str(data), output_dir=str(root / "runs"),
                    max_updates=args.max_updates, coarse_step=args.coarse_step,
                    transforms_sha256=sha(transforms), boxes_sha256=sha(args.boxes),
                    source_sha256=sha(SCRIPT), status="prepared")
                spec["normalization_sha256"] = normalization_sha256
                spec["common_scene_span"] = common_span
                spec["runtime_sources"] = runtime_sources
                spec['frequency_initialization'] = (seed_policy(seed_manifest, name, box, data, args.boxes)
                    if seed_manifest else dict(policy='zero_explicit'))
                if seed_manifest:
                    load_frequency_seed(spec['frequency_initialization'])
                path = root / "specs" / f"{name}.json"
                write(path, spec)
                variants.append(str(path))
    manifest = dict(schema_version=1, coordinate_system="prepared_world", data=str(data),
        transforms_sha256=sha(transforms), boxes_sha256=sha(args.boxes), source_sha256=sha(SCRIPT),
        boxes=str(args.boxes.resolve()), variants=variants,
        selection="exact all_eval PSNR; LPIPS tie-break within 0.07dB",
        review_status="requires calibration, bounds and occupancy visual review",
        max_updates_per_cold_run=args.max_updates, supervision_interval_seconds=1800,
        frequency_initialization_policy='paper_train_sfm' if seed_manifest else 'zero_explicit',
        frequency_seed_manifest=str(Path(seed_manifest).resolve()) if seed_manifest else None,
        frequency_seed_manifest_sha256=sha(seed_manifest) if seed_manifest else None,
        limitations=["unobserved geometry is not certified empty", "subject AABB keeps full uncomposited RGB targets",
                      "coarse step requires frozen-render convergence audit", "canonical 75940/106316 schedule is not used"])
    write(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


def verify_spec(spec):
    if not 4097 <= spec["max_updates"] <= 22000 or spec["radius"] not in (0, 1):
        raise ValueError("invalid experiment budget or radius")
    if sha(Path(spec["data"]) / "transforms.json") != spec["transforms_sha256"]:
        raise RuntimeError("transforms changed since manifest initialization")
    if sha(SCRIPT) != spec["source_sha256"]:
        raise RuntimeError("controller changed since manifest initialization")
    if sha(Path(spec["data"]) / "preparation.json") != spec["normalization_sha256"]:
        raise RuntimeError("normalization receipt changed since initialization")
    if source_hashes() != spec["runtime_sources"]:
        raise RuntimeError("runtime implementation changed since initialization")
    policy = spec.get('frequency_initialization')
    validate_seed_policy(policy)
    if policy['policy'] == 'paper_train_sfm':
        load_frequency_seed(policy)


def source_hashes():
    repository = SCRIPT.parents[2]
    files = [SCRIPT, SCRIPT.parent / 'run_lookcloser_quiet.py']
    files += [repository / 'nerfstudio' / path for path in (
        'fields/lookcloser_field.py', 'models/lookcloser.py',
        'pipelines/lookcloser_pipeline.py', 'model_components/lookcloser_grid.py',
        'model_components/lookcloser_occupancy.py', 'model_components/lookcloser_samplers.py',
        'model_components/lookcloser_colliders.py', 'model_components/losses.py',
        'lookcloser_pixel_sampler.py', 'engine/trainer.py',
        'configs/method_configs.py', 'data/dataparsers/nerfstudio_dataparser.py',
        'utils/lookcloser_rng.py', 'utils/lookcloser_projection.py')]
    return {str(path): sha(path) for path in files}


def configure_native(config, spec):
    import run_marshawn_hd_campaign as module
    parser = config.pipeline.datamanager.dataparser
    parser._target = module.MarshawnDataParser
    parser.marshawn_aabb = spec["aabb"]
    parser.auto_scale_poses = False
    parser.center_method = "none"
    parser.orientation_method = "none"
    parser.scale_factor = 1.0
    parser.load_3D_points = False
    pipe = config.pipeline
    pipe._target = module.MarshawnPipeline
    pipe.frequency_projection_mode = 'normalized_jacobian'
    pipe.marshawn_distortion_schedule = [4096, 12288, .01, 1.]
    if 'frequency_initialization' not in spec:
        raise ValueError('spec must explicitly bind its frequency initialization policy')
    pipe.marshawn_frequency_initialization = spec['frequency_initialization']
    pipe.independent_rng_streams = True
    pipe.training_seed = 42
    model = pipe.model
    # Keep density-to-opacity admission independent of the candidate box diagonal.
    model.render_step_size = float(spec["coarse_step"])
    model.occupancy_update_step_size = float(spec["coarse_step"])
    model.independent_rng_streams = True
    model.training_seed = 42
    model.correct_sh_directions = True
    model.density_activation = 'trunc_exp'
    model.eval_use_training_near_plane = True
    model.corrected_arm_allocator = True
    model.depth_loss_mult = 0.
    model.fixed_num_samples_per_ray = 256
    model.max_steps_per_ray = 4096
    model.adaptive_min_step_size = min(1e-4, spec['common_scene_span'] / (4 * 8192))
    model.adaptive_max_step_size = min(.1, spec['common_scene_span'] / 32)
    if model.adaptive_coarse_step_size > model.adaptive_max_step_size:
        raise ValueError("coarse step exceeds common physical maximum step")
    sampler = pipe.datamanager.pixel_sampler
    sampler.fas_warmup_steps = 4096
    sampler.fas_ramp_steps = 4096
    sampler.fas_strength = .5
    return config


def worker(path):
    import run_lookcloser_quiet as quiet
    spec = read(path)
    verify_spec(spec)
    audit_inputs(spec)
    original_command = quiet.train_command

    def command(args):
        argv = original_command(args)
        command_path = Path(path).with_suffix(".train_argv.json")
        write(command_path, argv[1:])
        return [sys.executable, str(SCRIPT), "train", "--spec", str(path), "--argv", str(command_path)]

    quiet.train_command = command
    quiet.best_eval_checkpoint = select_checkpoint
    sys.argv = ["run_lookcloser_quiet.py", *quiet_arguments(spec)]
    return quiet.main()


def install_checkpoint_protection(trainer_class):
    """Retain trained state before scheduled/final eval without changing any updates."""
    if getattr(trainer_class, '_marshawn_checkpoint_protection', False):
        return
    original_eval = trainer_class.eval_iteration
    original_after_train = trainer_class._after_train

    @functools.wraps(original_eval)
    def eval_iteration(self, step):
        save_interval = self.config.steps_per_save
        eval_intervals = (self.config.steps_per_eval_batch, self.config.steps_per_eval_image,
                          self.config.steps_per_eval_all_images)
        if (step > 0 and save_interval > 0 and step % save_interval == 0
                and any(interval > 0 and step % interval == 0 for interval in eval_intervals)):
            self.save_checkpoint(step)
        return original_eval(self, step)

    def after_train(self):
        # Preserve final state even if the extra final all-image eval raises.
        if self.step % self.config.steps_per_eval_all_images:
            self.save_checkpoint(self.step)
            previous = self.config.steps_per_eval_all_images
            try:
                self.config.steps_per_eval_all_images = 1
                self.eval_iteration(self.step)
            finally:
                self.config.steps_per_eval_all_images = previous
        original_after_train(self)

    trainer_class.eval_iteration = eval_iteration
    trainer_class._after_train = after_train
    trainer_class._marshawn_checkpoint_protection = True


def train_worker(path, argv_path):
    spec = read(path)
    verify_spec(spec)
    import tyro
    from nerfstudio.scripts import train
    from nerfstudio.engine.trainer import Trainer
    # Use Nerfstudio's exact CLI config, then persist the custom parser target and AABB.
    sys.argv = ["ns-train", *read(argv_path)]
    config = tyro.cli(train.AnnotatedBaseConfigUnion, args=read(argv_path))
    configure_native(config, spec)
    install_checkpoint_protection(Trainer)
    train.main(config)


def run_variant(args):
    root = args.root.resolve()
    manifest = read(root / "campaign.json")
    gate = read(args.gate)
    for key in ("transforms_sha256", "boxes_sha256"):
        if gate.get(key) != manifest[key]:
            raise RuntimeError(f"preflight review gate has mismatched {key}")
    if gate.get("approved") is not True:
        raise RuntimeError("preflight calibration/bounds review has not passed")
    path = root / "specs" / f"{args.variant}.json"
    if str(path) not in manifest["variants"]:
        raise ValueError("unknown variant")
    spec = read(path)
    verify_spec(spec)
    run_dir = Path(spec["output_dir"]) / spec["name"] / "lookcloser" / "cold_seed42"
    if run_dir.exists():
        raise FileExistsError("cold variant output exists; preserve it and initialize a new campaign")
    write(root / "input_audits" / f"{args.variant}.json", audit_inputs(spec))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SCRIPT.parent) + os.pathsep + environment.get("PYTHONPATH", "")
    log_path = root / "logs" / f"{args.variant}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write(root / "status" / f"{args.variant}.json", dict(status="running", started=time.time(), gate=str(args.gate.resolve())))
    with log_path.open("w") as stream:
        process = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--spec", str(path)],
            env=environment, stdout=stream, stderr=subprocess.STDOUT)
        try:
            returncode = process.wait()
        except BaseException:
            process.send_signal(2)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.terminate()
            raise
    summary_path = run_dir / "run_summary.json"
    status = dict(status="awaiting_visual_review" if returncode == 0 and summary_path.is_file() else "failed",
                  worker_returncode=returncode, finished=time.time(), log=str(log_path), summary=str(summary_path))
    config_path = run_dir / 'config.yml'
    if config_path.is_file():
        status['config_sha256'] = sha(config_path)
    if summary_path.is_file():
        summary = read(summary_path)
        selected = summary.get('selected_checkpoint')
        if selected and Path(selected).is_file():
            status['selected_checkpoint_sha256'] = sha(selected)
        evaluation = summary.get('eval') or {}
        render_dir = evaluation.get('render_dir')
        if render_dir and Path(render_dir).is_dir():
            status['render_sha256'] = {str(p):sha(p) for p in sorted(Path(render_dir).rglob('*.png'))}
    write(root / "status" / f"{args.variant}.json", status)
    print(json.dumps(status))
    return returncode if returncode else (0 if summary_path.is_file() else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    for name in ("data", "boxes", "root"):
        init.add_argument("--" + name, type=Path, required=True)
    init.add_argument("--max-updates", type=int, default=22000)
    init.add_argument("--coarse-step", type=float, default=.001)
    policy = init.add_mutually_exclusive_group(required=True)
    policy.add_argument('--frequency-seed-manifest', type=Path)
    policy.add_argument('--zero-frequency-init', action='store_true')
    run = commands.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--variant", required=True)
    run.add_argument("--gate", type=Path, required=True)
    for name in ("worker", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--spec", type=Path, required=True)
        if name == "train":
            sub.add_argument("--argv", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args)
        return 0
    if args.command == "run":
        return run_variant(args)
    if args.command == "worker":
        return worker(args.spec)
    train_worker(args.spec, args.argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
