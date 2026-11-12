"""Inference-only depth cleanup experiments for the frozen sparse24 models.

Never modifies weights, source masks, admission grids, or training services.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def camera_for(pipe, spec, root, name):
    ready = json.loads((Path(spec['data']) / 'ready.json').read_text())
    row = next(r for r in ready['records'] if r['camera'] == name)
    dataset = pipe.datamanager.eval_dataset
    idx = next(i for i, p in enumerate(dataset.image_filenames) if Path(p).stem == name)
    cam = copy.deepcopy(dataset.cameras[idx:idx+1].to(pipe.model.device))
    x, y, _, _ = row['native_crop_xyxy']
    width, height = json.loads((root/'original_camera_sizes.json').read_text())[name]
    cam.cx += x; cam.cy += y
    cam.width.fill_(width); cam.height.fill_(height)
    cam.rescale_output_resolution(min(1920/width, 1080/height))
    return cam


def save_rgb(path, array):
    pixels = np.uint8(np.clip(array * 255 + .5, 0, 255))
    h, w = pixels.shape[:2]
    canvas = Image.new('RGB', (1920, 1080))
    canvas.paste(Image.fromarray(pixels), ((1920-w)//2, (1080-h)//2))
    canvas.save(path)


def depth_support(raw, minimum_component=2000, preserve_lower_border=False, minimum_height=0):
    """Opaque connected field regions anchor depth; no image/SAM masks are read."""
    from scipy import ndimage as ndi
    alpha = raw['accumulation'][..., 0]
    depth = raw['depth'][..., 0]
    labels, _ = ndi.label(alpha > .98)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_component
    keep[0] = False
    if minimum_height:
        for label, box in enumerate(ndi.find_objects(labels), start=1):
            if box is not None and box[0].stop-box[0].start < minimum_height:
                keep[label] = False
    if preserve_lower_border:
        # The fourth actor exits at the lower-right canvas edge. Keep small
        # partial components there; detached overhead/background islands do not
        # qualify. This exception is specific to the calibrated GC delivery.
        h, w = alpha.shape
        for label in np.unique(labels[int(.70*h):, -1]):
            if label and sizes[label] >= 200:
                ys, _ = np.nonzero(labels == label)
                if ys.min() >= .70*h:
                    keep[label] = True
    core = keep[labels]
    if not core.any():
        raise ValueError('No trustworthy connected opaque surface; do not export empty result')
    distance, indices = ndi.distance_transform_edt(~core, return_indices=True)
    reference = depth[tuple(indices)]
    return core, distance.astype(np.float32), reference.astype(np.float32)


def install_depth_gate(model, camera, reference, distance, band_m, fade_m, radius, feather):
    """Gate sample density BEFORE transmittance/RGB integration, for this camera.

    Uses the existing calibrated projection and soft image-space support around
    connected opaque surface anchors. This is a view-conditioned cleanup, not a
    recovered semantic 3D segmentation. Original semantic admission stays active.
    """
    from nerfstudio.field_components.field_heads import FieldHeadNames
    device = model.device
    ref = torch.as_tensor(reference, device=device)
    dist = torch.as_tensor(distance, device=device)
    pose = camera.camera_to_worlds[0]
    origin, rotation = pose[:, 3], pose[:, :3]
    fx, fy, cx, cy = (getattr(camera, k).item() for k in ('fx', 'fy', 'cx', 'cy'))
    h, w = ref.shape
    unit = .2217203609534452 * 1.0805

    def smooth(value):
        value = value.clamp(0, 1)
        return value.square() * (3-2*value)

    def gate(module, args, outputs):
        samples = args[0]
        world = samples.frustums.get_positions()
        delta = world-origin
        local = delta @ rotation
        z = -local[..., 2]
        x = torch.floor(fx * local[..., 0]/z.clamp_min(1e-9)+cx).long()
        y = torch.floor(-fy * local[..., 1]/z.clamp_min(1e-9)+cy).long()
        valid = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x = x.clamp(0, w-1); y = y.clamp(0, h-1)
        t = torch.linalg.vector_norm(delta, dim=-1)
        depth_weight = smooth((band_m+fade_m-(t-ref[y, x]).abs()/unit)/fade_m)
        support_weight = smooth((radius+feather-dist[y, x])/feather)
        weight = valid * depth_weight * support_weight
        outputs[FieldHeadNames.DENSITY] = outputs[FieldHeadNames.DENSITY] * weight[..., None]
        return outputs

    return model.field.register_forward_hook(gate)


def run(args):
    root = Path(args.root)
    sys.path.insert(0, str(root/'tools'))
    from inference import load_pipeline
    from common import sha, atomic
    rows = json.loads((root/'sparse24_storage_audit.json').read_text())['results']
    row = next(r for r in rows if r['frame'] == args.frame)
    checkpoint = Path(row['path'])
    assert sha(checkpoint) == row['model_sha256']
    spec_path = checkpoint.parent.parent/'train/spec.json'
    out = Path(args.output)/f'{args.frame:06}'
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'raw' and (out/'raw_receipt.json').is_file():
        receipt = json.loads((out/'raw_receipt.json').read_text())
        assert receipt['checkpoint_sha256'] == row['model_sha256']
        print('RAW_CACHED', args.frame, flush=True)
        return
    start = time.time()
    pipe, spec, step = load_pipeline(root, spec_path, checkpoint, out)
    pipe.model.config.adaptive_max_frequency_level = 10
    camera = camera_for(pipe, spec, root, args.camera)
    if args.stage == 'raw':
        with torch.inference_mode():
            result = pipe.model.get_outputs_for_camera(camera)
        arrays = {k: result[k].float().cpu().numpy() for k in ('rgb', 'depth', 'accumulation')}
        np.savez_compressed(out/'raw.npz', **arrays)
        save_rgb(out/'raw.png', arrays['rgb'])
    else:
        raw = np.load(out/'raw.npz')
        if args.temporal:
            guide = np.load(out/(args.support_name+'.npz'))
            distance, reference = guide['distance'], guide['reference']
            minimum_component = int(guide['minimum_component']) if 'minimum_component' in guide else 2000
            minimum_height = int(guide['minimum_height']) if 'minimum_height' in guide else 0
        else:
            core, distance, reference = depth_support(raw)
            minimum_component = 2000
            minimum_height = 0
        handle = install_depth_gate(pipe.model, camera, reference, distance,
            args.band, args.fade, args.radius, args.feather)
        try:
            with torch.inference_mode():
                result = pipe.model.get_outputs_for_camera(camera)
        finally:
            handle.remove()
        arrays = {k: result[k].float().cpu().numpy() for k in ('rgb', 'depth', 'accumulation')}
        np.savez_compressed(out/(args.name+'.npz'), **arrays)
        save_rgb(out/(args.name+'.png'), arrays['rgb'])
    atomic(out/(('raw' if args.stage=='raw' else args.name)+'_receipt.json'), dict(frame=args.frame, camera=args.camera,
        checkpoint=str(checkpoint), checkpoint_sha256=row['model_sha256'], step=step,
        seconds=time.time()-start, masks_applied_to_output=False,
        cleanup=None if args.stage=='raw' else dict(band_m=args.band, fade_m=args.fade,
            radius_pixels=args.radius, feather_pixels=args.feather, core_alpha=.98,
            minimum_component=minimum_component, sample_density_gate=True, view_conditioned=True,
            minimum_anchor_height_pixels=minimum_height,
            temporal_support=args.temporal, support_name=args.support_name,
            support_sha256=sha(out/(args.support_name+'.npz')) if args.temporal else None,
            RGB_temporal_blending=False),
        camera_to_worlds=camera.camera_to_worlds.cpu().tolist(),
        model_units_per_metre=.2217203609534452*1.0805))
    print(args.stage.upper()+'_DONE', args.frame, round(time.time()-start, 2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--frame', type=int, required=True)
    p.add_argument('--output', required=True); p.add_argument('--camera', default='GC')
    p.add_argument('--stage', choices=['raw', 'clean'], default='raw')
    p.add_argument('--name', default='depth_clean')
    p.add_argument('--temporal', action='store_true')
    p.add_argument('--support-name', default='temporal_support')
    p.add_argument('--band', type=float, default=.15); p.add_argument('--fade', type=float, default=.10)
    p.add_argument('--radius', type=float, default=8); p.add_argument('--feather', type=float, default=6)
    run(p.parse_args())
