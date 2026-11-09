#!/usr/bin/env python3
"""Read-only native weight-profile export for the selected LookCloser control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from dec5_hair import cv_pose, ray_to_z, volume_depths
from run_dec5_hair_campaign import atomic_json, sha


def fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=4096)
    p.add_argument('--validation-only', action='store_true')
    p.add_argument('--camera-manifest', type=Path)
    p.add_argument('--rgb-downscale', type=int, default=1)
    p.add_argument('--target-roi', type=Path)
    p.add_argument('--rgb-samples', type=int)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--common-dense-probe',action='store_true',
                   help='Probe an adaptive-trained field with a common uniform marcher; retain learned grid/FR weights.')
    p.add_argument('--profiles-only', action='store_true',
                   help='Save 64 complete weight traces in each of 8 train cameras; no rendering/refitting.')
    p.add_argument('--probe-rays-per-chunk', type=int, default=512,
                   help='Chunk size for pose-manifest depth probes; weights/model stay fixed.')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configs = list(args.training.rglob('config.yml'))
    if len(configs) != 1:
        raise ValueError('Expected one pinned training configuration')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(24*(1 << 30)/torch.cuda.get_device_properties(0).total_memory)
    config = yaml.load(configs[0].read_text(), Loader=yaml.Loader)
    # Archived configs contain their original host's absolute data path.
    # Relocate only an exactly hash-matched campaign dataset, never silently
    # normalize a checkpoint against another set of cameras.
    training_request = json.loads((args.training/'request.json').read_text())
    if sha(args.root/'dataset/transforms.json') != training_request['data_sha256']:
        raise ValueError('Checkpoint dataset differs from the requested campaign')
    config.pipeline.datamanager.data = args.root/'dataset'
    config.pipeline.datamanager.dataparser.data = args.root/'dataset'
    pipeline = config.pipeline.setup(device='cuda', test_mode='val')
    checkpoint_path = args.checkpoint if args.checkpoint is not None else args.training/'best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pipeline.load_state_dict(checkpoint['pipeline'], strict=True)
    pipeline.eval()
    model = pipeline.model
    model.current_train_step = checkpoint['step']
    native_mode=model.config.ray_sampling_mode
    adaptive=(native_mode=='adaptive' or (native_mode=='auto' and model.config.enable_adaptive_ray_marching))
    if adaptive and not args.common_dense_probe:
        raise ValueError('Adaptive sample locations are not uniform: use --common-dense-probe for valid weight-profile depths')
    if args.common_dense_probe:
        model.config.ray_sampling_mode='fixed'
        model.config.enable_adaptive_ray_marching=False
    native_sampling = model.config.fixed_num_samples_per_ray
    model.config.fixed_num_samples_per_ray = args.samples
    before = fingerprint(model)
    dataset = pipeline.datamanager.train_dataset
    payload = json.loads((args.root/'dataset/transforms.json').read_text())
    by_name = {Path(f['file_path']).name:f for f in payload['frames']}
    original = [by_name[Path(p).name] for p in dataset.image_filenames]
    normalized = dataset.cameras.camera_to_worlds.detach().cpu().numpy()
    raw = np.array([f['transform_matrix'] for f in original])
    rotation = normalized[0, :3, :3] @ raw[0, :3, :3].T
    scale = np.linalg.norm(normalized[1, :3, 3]-normalized[0, :3, 3])/np.linalg.norm(raw[1, :3, 3]-raw[0, :3, 3])
    translation = normalized[0, :3, 3]-scale*rotation @ raw[0, :3, 3]
    error = np.linalg.norm(scale*raw[:, :3, 3] @ rotation.T+translation-normalized[:, :3, 3], axis=-1).max()
    if error > 1e-4:
        raise ValueError('Camera normalization is not a single verified similarity')
    atomic_json(args.output/'request.json', dict(checkpoint_sha256=sha(checkpoint_path), step=checkpoint['step'],
        samples=args.samples, native_sampling=native_sampling, fixed_field_sha256=before,
        normalized_per_calibration_unit=float(scale), pose_alignment_max_error=float(error),
        rotation=rotation.tolist(), translation=translation.tolist(), depth_convention='OpenCV camera-z',
        units='calibration units', opacity_threshold=.1, peak_separation_samples=2,
        extra_peak_not_certified_geometry=True,native_ray_sampling_mode=native_mode,
        probe_ray_sampling_mode=model.config.ray_sampling_mode,common_dense_probe=args.common_dense_probe))
    if args.profiles_only:
        records = []
        for index in np.linspace(0, len(original)-1, 8, dtype=int):
            frame = original[index]
            name = frame['physical_camera']
            semantic = np.load(args.root/'semantics'/(name+'.npz'))['probability'][..., 0]
            yy, xx = np.nonzero(semantic > 32767)
            selected = np.linspace(0, len(xx)-1, min(64, len(xx)), dtype=int)
            uv = np.stack((xx[selected]+.5, yy[selected]+.5), -1)
            camera = dataset.cameras[index:index+1].to('cuda')
            rays = camera.generate_rays(0, coords=torch.as_tensor(uv[:, ::-1].copy(), device='cuda', dtype=torch.float32))
            with torch.inference_mode():
                result = model(rays)
            weights = result['loss_weights'].float()[..., 0].cpu().numpy()
            near, far = [v[:, 0].float().cpu().numpy() for v in (rays.nears, rays.fars)]
            t = (near[:, None]+(far-near)[:, None]*(np.arange(weights.shape[-1])+.5)/weights.shape[-1])/scale
            z = ray_to_z(t, uv[:, None, :], frame)
            path = args.output/(name+'.npz')
            np.savez_compressed(path, native_uv=uv, ray_distance=t, camera_z=z, weights=weights,
                                **volume_depths(z, weights))
            records.append(dict(camera=name, rays=len(uv), sha256=sha(path)))
        if before != fingerprint(model):
            raise ValueError('Frozen model changed during profile export')
        atomic_json(args.output/'complete.json', dict(records=records, samples=args.samples,
                    checkpoint_sha256=sha(checkpoint_path), frozen_state_unchanged=True,
                    uses_eval_rgb=False, profile_selection='64 scan-order quantiles of train semantic hair per camera'))
        return
    if args.camera_manifest is not None:
        from PIL import Image
        from nerfstudio.cameras.cameras import Cameras, CameraType
        frames = json.loads(args.camera_manifest.read_text())['frames']
        records = []
        depth_probe = args.target_roi is not None
        if depth_probe and (args.rgb_downscale != 1 or not 1 <= args.probe_rays_per_chunk <= 1024):
            raise ValueError('Depth probes require native ROI and chunk size 1..1024')
        model.config.fixed_num_samples_per_ray = args.samples if depth_probe else (args.rgb_samples or native_sampling)
        for index, frame in enumerate(frames):
            raw_pose = np.array(frame['transform_matrix'])
            normalized_pose = raw_pose.copy()
            normalized_pose[:3, :3] = rotation @ raw_pose[:3, :3]
            normalized_pose[:3, 3] = scale*rotation @ raw_pose[:3, 3]+translation
            stride = args.rgb_downscale
            if stride < 1 or frame['w'] % stride or frame['h'] % stride:
                raise ValueError('Invalid neural preview downscale')
            if any(abs(float(frame.get(k, 0))) > 1e-10 for k in ('k1','k2','p1','p2')):
                raise ValueError('Novel cameras must be undistorted')
            camera = Cameras(camera_to_worlds=torch.as_tensor(normalized_pose[:3, :4], dtype=torch.float32)[None],
                fx=frame['fl_x']/stride, fy=frame['fl_y']/stride, cx=frame['cx']/stride, cy=frame['cy']/stride,
                width=frame['w']//stride, height=frame['h']//stride, camera_type=CameraType.PERSPECTIVE).to('cuda')
            name = frame['physical_camera']
            if depth_probe:
                mask = np.load(args.target_roi/(name+'.npz'), allow_pickle=False)['roi']
                if mask.shape != (frame['h'], frame['w']):
                    raise ValueError('Probe ROI/calibration shape mismatch')
                yy, xx = np.nonzero(mask)
                fields = ('expected','unconditioned','opacity','front','peak1','peak2','peak1_subpixel')
                output = {key:np.zeros(mask.shape, np.float32) for key in fields}
                for begin in range(0, len(xx), args.probe_rays_per_chunk):
                    x, y = xx[begin:begin+args.probe_rays_per_chunk], yy[begin:begin+args.probe_rays_per_chunk]
                    uv = np.stack((x+.5, y+.5), -1)
                    rays = camera.generate_rays(0, coords=torch.as_tensor(uv[:, ::-1].copy(), device='cuda', dtype=torch.float32))
                    with torch.inference_mode():
                        result = model(rays)
                    weights = result['loss_weights'].float()[..., 0].cpu().numpy()
                    near, far = [v[:, 0].float().cpu().numpy() for v in (rays.nears, rays.fars)]
                    t = (near[:, None]+(far-near)[:, None]*(np.arange(weights.shape[-1])+.5)/weights.shape[-1])/scale
                    for key, value in volume_depths(t, weights).items():
                        output[key][y, x] = value if key == 'opacity' else ray_to_z(value, uv, frame)
                # Atomic publication permits a checksum-verified external drain
                # to the artifact host when the compute host has little disk.
                temporary = args.output/(name+'.tmp.npz')
                np.savez_compressed(temporary, **output)
                digest = sha(temporary)
                temporary.replace(args.output/(name+'.npz'))
                records.append(dict(camera=name, sha256=digest))
            else:
                with torch.inference_mode():
                    result = model.get_outputs_for_camera_ray_bundle(camera.generate_rays(0))
                rgb = result['rgb'].float().cpu().numpy()
                np.savez_compressed(args.output/(name+'.npz'), rgb_display=rgb)
                records.append(dict(camera=name, sha256=sha(args.output/(name+'.npz'))))
            atomic_json(args.output/'progress.json', dict(status='rendering', completed=index+1, total=len(frames)))
        if before != fingerprint(model):
            raise ValueError('Frozen field changed during camera-manifest export')
        roi_receipt = json.loads((args.target_roi/'complete.json').read_text()) if depth_probe else {}
        atomic_json(args.output/'complete.json', dict(records=records, checkpoint_sha256=sha(checkpoint_path),
            camera_manifest_sha256=sha(args.camera_manifest), uses_eval_rgb=False, rgb_downscale=stride,
            target_roi_receipt_sha256=sha(args.target_roi/'complete.json') if depth_probe else None,
            target_roi_parent_receipt_sha256=roi_receipt.get('parent_prediction_receipt_sha256'),
            output_kind='depth_profiles' if depth_probe else 'rgb',
            frozen_state_unchanged=before == fingerprint(model)))
        atomic_json(args.output/'progress.json', dict(status='complete', completed=len(frames), total=len(frames)))
        return
    if args.target_roi is not None:
        dataset_eval = pipeline.datamanager.eval_dataset
        for index in range(len(dataset_eval)):
            frame = by_name[Path(dataset_eval.image_filenames[index]).name]
            name = frame['physical_camera']
            mask = np.load(args.target_roi/(name+'.npz'))['roi']
            yy, xx = np.nonzero(mask)
            output = {k:np.zeros(mask.shape, np.float32) for k in ('expected','unconditioned','opacity','front','peak1','peak2','peak1_subpixel')}
            camera = dataset_eval.cameras[index:index+1].to('cuda')
            for begin in range(0, len(xx), 128):
                x, y = xx[begin:begin+128], yy[begin:begin+128]
                coords = torch.as_tensor(np.stack((y+.5, x+.5), -1), device='cuda', dtype=torch.float32)
                rays = camera.generate_rays(0, coords=coords)
                with torch.inference_mode():
                    result = model(rays)
                weights = result['loss_weights'].float()[..., 0].cpu().numpy()
                near, far = [v[:, 0].float().cpu().numpy() for v in (rays.nears, rays.fars)]
                t = near[:, None]+(far-near)[:, None]*(np.arange(weights.shape[-1])+.5)/weights.shape[-1]
                values = volume_depths(t/scale, weights)
                for key, value in values.items():
                    output[key][y, x] = value if key == 'opacity' else ray_to_z(value, np.stack((x+.5, y+.5), -1), frame)
            np.savez_compressed(args.output/(name+'.npz'), **output)
            atomic_json(args.output/'progress.json', dict(status='exporting', completed=index+1, total=3))
        atomic_json(args.output/'complete.json', dict(cameras=3, target_roi_receipt_sha256=sha(args.target_roi/'complete.json'),
                    uses_eval_rgb=False, checkpoint_sha256=sha(checkpoint_path), unchanged=before == fingerprint(model)))
        atomic_json(args.output/'progress.json', dict(status='complete', completed=3, total=3))
        return
    if args.validation_only:
        from PIL import Image
        records = []
        model.config.fixed_num_samples_per_ray = args.rgb_samples or native_sampling
        evaluation = pipeline.datamanager.eval_dataset
        for index in range(len(evaluation)):
            frame = by_name[Path(evaluation.image_filenames[index]).name]
            camera = evaluation.cameras[index:index+1].to('cuda')
            with torch.inference_mode():
                result = model.get_outputs_for_camera_ray_bundle(camera.generate_rays(0))
            rgb = result['rgb'].float().cpu().numpy()
            name = frame['physical_camera']
            np.savez_compressed(args.output/(name+'.npz'), rgb_display=rgb,
                                ray_distance_calibration=result['depth'].float().cpu().numpy()/scale)
            Image.fromarray(np.rint(rgb.clip(0, 1)*255).astype(np.uint8)).save(args.output/(name+'.png'))
            records.append(dict(camera=name, sha256=sha(args.output/(name+'.npz'))))
            atomic_json(args.output/'progress.json', dict(status='rendering', completed=index+1, total=3))
        atomic_json(args.output/'complete.json', dict(records=records, checkpoint_sha256=sha(checkpoint_path),
                    uses_eval_rgb=False, resolution=[1920, 1080], sampling=model.config.fixed_num_samples_per_ray,
                    frozen_state_unchanged=before == fingerprint(model)))
        atomic_json(args.output/'progress.json', dict(status='complete', completed=3, total=3))
        return
    for index, frame in enumerate(original):
        name = frame['physical_camera']
        semantic = np.load(args.root/'semantics'/(name+'.npz'))['probability'][..., 0].astype(np.float32)/65535
        semantic = cv2.resize(semantic, (960, 540), interpolation=cv2.INTER_AREA)
        domain = cv2.dilate((semantic > .1).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        yy, xx = np.nonzero(domain)
        output = {k:np.zeros((540, 960), np.float32) for k in ('expected','unconditioned','opacity','front','peak1','peak2','peak1_subpixel')}
        camera = dataset.cameras[index:index+1].to('cuda')
        for begin in range(0, len(xx), 128):
            x, y = xx[begin:begin+128], yy[begin:begin+128]
            # The half-resolution MVS pixel center corresponds to 2*x+1 in native coordinates.
            coordinates = torch.as_tensor(np.stack((2*y+1, 2*x+1), -1), device='cuda', dtype=torch.float32)
            rays = camera.generate_rays(0, coords=coordinates)
            with torch.inference_mode():
                result = model(rays)
            weights = result['loss_weights'].float()[..., 0].cpu().numpy()
            near, far = [x[:, 0].float().cpu().numpy() for x in (rays.nears, rays.fars)]
            t = near[:, None]+(far-near)[:, None]*(np.arange(weights.shape[-1])+.5)/weights.shape[-1]
            values = volume_depths(t/scale, weights)
            native_uv = np.stack((2*x+1, 2*y+1), -1)
            for key, value in values.items():
                output[key][y, x] = value if key == 'opacity' else ray_to_z(value, native_uv, frame)
        np.savez_compressed(args.output/(name+'.npz'), **output)
        atomic_json(args.output/'progress.json', dict(status='exporting', completed=index+1, total=59))
    after = fingerprint(model)
    if before != after:
        raise ValueError('Frozen model state changed during diagnostic export')
    atomic_json(args.output/'complete.json', dict(before=before, after=after, unchanged=True, cameras=59,
                checkpoint_sha256=sha(checkpoint_path)))
    atomic_json(args.output/'progress.json', dict(status='complete', completed=59, total=59))


if __name__ == '__main__':
    main()
