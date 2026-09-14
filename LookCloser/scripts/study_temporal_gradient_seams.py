"""CPU-only matched gradient-color control on an existing dynamic movie crop.

Reuses the previously tested solver on the newer fixed-profile hard texture.
No geometry/labels/visibility edits, no held-out RGB, no video replacement.
Explicit display-domain correction, NOT unchanged train-RGB reprojection.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import torch

from joint_temporal_texture import ROOT, SOURCE, HELD_CAMERAS, cameras, read, sha, atomic_json, exr, project, sample, bounded_warp, display
from render_smooth_temporal_mesh_video import verify_request
from bake_joint_temporal_mesh import camera_depth
from diffusion_mesh_repair import scene_for
from hard_source_gradient_leveling import level_source_gradients


def run(parent, output, frame, crop):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    request = verify_request(parent)
    record = next(r for r in request['inventory'] if r['frame_id'] == frame)
    saved = parent / 'frames' / frame
    complete = read(saved / 'complete.json')
    if complete['request_sha256'] != sha(parent / 'request.json'):
        raise ValueError('Parent render request mismatch')
    bindings = {str(parent / 'request.json'): sha(parent / 'request.json')}
    for name in ['frame.png', 'source_ids.png', 'target_depth.npz']:
        path = saved / name
        if sha(path) != complete['hashes'][name]:
            raise ValueError('Changed parent render')
        bindings[str(path)] = sha(path)
    if sha(record['mesh']) != record['mesh_sha256']:
        raise ValueError('Changed mesh')
    mesh = o3d.io.read_triangle_mesh(record['mesh'])
    v, t = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.uint32)
    tv = v[t]
    scene = scene_for(v, t)
    depth, ids, bary = camera_depth(scene, record['camera'])
    if not np.allclose(np.where(np.isfinite(depth), depth, 0), np.load(saved / 'target_depth.npz')['depth'], atol=1e-6):
        raise ValueError('Fresh camera raycast differs')
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= 1080 and 0 <= y0 < y1 <= 1920):
        raise ValueError('Invalid native portrait crop')
    sl = (slice(y0, y1), slice(x0, x1))
    pd = np.rot90(depth)[sl].copy()
    pi, pb = np.rot90(ids)[sl], np.rot90(bary)[sl]
    hit = np.isfinite(pd)
    weights = np.column_stack((1-pb[hit].sum(1), pb[hit]))
    points = (tv[pi[hit]] * weights[:, :, None]).sum(1)
    baseline = np.array(Image.open(saved / 'frame.png').convert('RGB'))[sl].copy()
    selection = np.rot90(np.asarray(Image.open(saved / 'source_ids.png')))[sl].astype(np.int64)
    selection[selection == 255] = -1
    rows, _, _ = cameras(frame)
    if [r['physical_camera'] for r in rows] != read(saved / 'result.json')['source_cameras']:
        raise ValueError('Camera label order changed')
    if len(rows) != 62 or set(r['physical_camera'] for r in rows) & HELD_CAMERAS:
        raise ValueError('Train camera inventory invalid')
    source = next(r for r in request['source_rows'] if Path(r['source_dataset']).name == frame)
    expected = {r['physical_camera']: r['sha256'] for r in source['source_images']}
    profiles = np.load(ROOT / 'parameters.npz')
    gain = torch.from_numpy(profiles['log_gain'])
    gain = (gain-gain.mean(0, keepdim=True)).exp()
    static = torch.from_numpy(profiles['static_warp'])
    exposure = read(ROOT / 'exposure.json')['fixed_exposure_gain']
    masks_root = Path(record['source_masks']['root'])
    if sha(masks_root / 'masks.npz') != record['source_masks']['masks_sha256']:
        raise ValueError('Changed source foreground masks')
    masks = dict(zip(read(masks_root / 'cameras.json'), np.load(masks_root / 'masks.npz')['masks']))
    spec = dict(frame=frame, crop=crop, parent_request_sha256=sha(parent / 'request.json'),
                mesh_sha256=record['mesh_sha256'], uses_heldout_rgb=False, geometry_changed=False,
                source_labels_changed=False, source_rgb_averaging=False, scope='local color diagnostic only',
                fixed_profiles_sha256=sha(ROOT / 'parameters.npz'), fixed_exposure=exposure,
                script_sha256=sha(__file__), solver_sha256=sha(Path(__file__).with_name('hard_source_gradient_leveling.py')))
    atomic_json(output / 'request.json', spec)
    warped, valid_masks = [], []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            path = Path(row['file_path'])
            if sha(path) != expected[row['physical_camera']]:
                raise ValueError('Source EXR changed')
            bindings[str(path)] = expected[row['physical_camera']]
            rgb = torch.from_numpy(exr(path).transpose(2, 0, 1).copy())[None]
            sd = camera_depth(scene, row)[0]
            sd[(~np.isfinite(sd)) | (masks[row['physical_camera']] == 0)] = 0
            sd = torch.from_numpy(sd)[None, None]
            uv, z = project(points, [row])
            q, z = torch.from_numpy(uv[:, None]), torch.from_numpy(z)
            d = sample(sd, q)[:, 0, 0]
            valid = (z > 0) & (d > 0) & ((d-z).abs() < .0015*z)
            valid &= (q[:,0,:,0]>2)&(q[:,0,:,0]<1917)&(q[:,0,:,1]>2)&(q[:,0,:,1]<1077)
            shifted = q + bounded_warp(static[i:i+1], torch.zeros_like(static[i:i+1]), q)
            d = sample(sd, shifted)[:, 0, 0]
            safe = (d > 0) & ((d-z).abs() < .0025*z)
            for dx, dy in [(0,0),(1,0),(0,1),(1,1)]:
                tap = sample(sd, q.floor()+q.new_tensor([dx,dy]))[:,0,0]
                valid &= (tap>0)&((tap-z).abs()<.003*z)
                tap = sample(sd, shifted.floor()+q.new_tensor([dx,dy]))[:,0,0]
                safe &= (tap>0)&((tap-z).abs()<.003*z)
            shifted = torch.where(safe[:,None,:,None], shifted, q)
            c = sample(rgb, shifted)[0,:,0].T * gain[i]
            colors = display(c.numpy().clip(0), exposure)
            image = np.zeros((*hit.shape, 3), np.float32)
            image[hit] = np.rint(colors*255).clip(0,255)/255
            vm = np.zeros(hit.shape, bool)
            vm[hit] = valid[0].numpy()
            warped.append(torch.from_numpy(image.transpose(2,0,1).copy()))
            valid_masks.append(torch.from_numpy(vm))
            if i % 8 == 0:
                atomic_json(output / 'progress.json', dict(stage='source_warps_cpu', camera=i, elapsed_seconds=time.monotonic()-started))
                print('source', i, flush=True)
        selected = np.zeros_like(baseline)
        invalid_selected = 0
        for i, rgb in enumerate(warped):
            mask = selection == i
            selected[mask] = np.rint(rgb.permute(1,2,0).numpy()[mask]*255).astype(np.uint8)
            invalid_selected += int((mask & ~valid_masks[i].numpy()).sum())
        error = np.abs(selected.astype(np.int16)-baseline.astype(np.int16))
        if error.max() > 1 or invalid_selected:
            raise ValueError(f'CPU native control mismatch: max_rgb8={error.max()}, invalid_selected={invalid_selected}')
        Image.fromarray(baseline).save(output / 'baseline.png')
        prediction = torch.from_numpy(baseline.transpose(2,0,1).copy()).float()/255
        atomic_json(output / 'progress.json', dict(stage='gradient_solver_cpu', elapsed_seconds=time.monotonic()-started))
        print('matched warps; solver started', flush=True)
        corrected, offset, stats = level_source_gradients(prediction, torch.from_numpy(selection), warped, valid_masks,
                                                          torch.from_numpy(np.where(hit,pd,0)))
    result = np.rint(corrected.permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
    Image.fromarray(result).save(output / 'corrected.png')
    np.savez_compressed(output / 'offset.npz', offset=offset.numpy(), selection=selection, depth=np.where(hit,pd,0))
    panel = Image.new('RGB', (baseline.shape[1]*2, baseline.shape[0]+25))
    draw = ImageDraw.Draw(panel)
    for i, (name, rgb) in enumerate([('published hard RGB', baseline), ('same labels + gradient color correction', result)]):
        panel.paste(Image.fromarray(rgb),(i*baseline.shape[1],25));draw.text((i*baseline.shape[1]+3,4),name,fill='white')
    panel.save(output / 'comparison.png')
    atomic_json(output / 'result.json', dict(stats=stats, elapsed_seconds=time.monotonic()-started,
                max_matched_rgb8_error=int(error.max()), invalid_selected_pixels=invalid_selected,
                input_hashes=bindings, visual_status='pending', image_quality_metrics_computed=False,
                hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('completed color diagnostic', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=Path('/mnt/data/dec5_elevated_camera_dynamic_150'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frame',default='001193')
    parser.add_argument('--crop',type=int,nargs=4,default=[200,740,700,1370])
    args=parser.parse_args()
    run(args.parent,args.output,args.frame,args.crop)
