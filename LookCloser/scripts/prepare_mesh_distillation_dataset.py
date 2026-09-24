"""Prepare a frozen single-time mesh teacher; does not launch NeRF training.

Reuses the selected renderer verbatim. Caches only immutable source images and
source raycasts across views. Unknown/background pixels are NOT negative evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import gzip
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

from joint_temporal_texture import (
    ROOT, SOURCE, CALIBRATION, HELD_CAMERAS, CAMERA_FIELDS,
    cameras, read, sha, atomic_json, exr, display,
)

OUTPUT = Path('/mnt/data/dec5_000973_mesh_distillation_v1')
PARENT = Path('/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free')
FRAME = '000973'
SEED = 20260924


def camera_plan(rows, seed=SEED):
    """Local barycentric positions: no extrapolation or held-out pose anchors."""
    xy = np.array([[ord(r['physical_camera'][0])-65,
                    ord(r['physical_camera'].split('_')[1][0])-65] for r in rows], float)
    tri = Delaunay(xy)
    # Central D..K, B..D. All 62 original poses are included separately.
    cells = [s for s in tri.simplices if
             (xy[s, 0] >= 3).all() and (xy[s, 0] <= 10).all() and
             (xy[s, 1] >= 1).all() and (xy[s, 1] <= 3).all() and
             np.ptp(xy[s], axis=0).max() <= 2]
    if not cells:
        raise ValueError('No local train-camera cells')
    rng = np.random.default_rng(seed)
    plan = []
    for i, row in enumerate(rows):
        plan.append(dict(id=f'train_{i:04d}', split='train', kind='real_train_pose',
                         camera=deepcopy(row), parents=[row['physical_camera']], weights=[1.]))
    accepted = []
    for i in range(262):
        ids = cells[i % len(cells)]
        for attempt in range(10000):
            weights = .1 + .7*rng.dirichlet(np.ones(3))
            pos2 = weights @ xy[ids]
            # Independent validation poses, not duplicated train rays.
            if all(np.linalg.norm(pos2-p) >= .035 for p in accepted):
                break
        else:
            raise ValueError('Could not separate synthetic poses')
        accepted.append(pos2)
        poses = np.asarray([rows[j]['transform_matrix'] for j in ids])
        pose = np.eye(4)
        pose[:3, 3] = weights @ poses[:, :3, 3]
        pose[:3, :3] = Rotation.from_matrix(poses[:, :3, :3]).mean(weights).as_matrix()
        row = deepcopy(rows[ids[0]])
        row['transform_matrix'] = pose.tolist()
        for key in ['fl_x', 'fl_y', 'cx', 'cy']:
            row[key] = float(sum(weights[k]*rows[j][key] for k, j in enumerate(ids)))
        split = 'train' if i < 238 else 'val'
        name = f'train_{i+62:04d}' if split == 'train' else f'val_{i-238:04d}'
        row.update(physical_camera=f'synthetic_{name}')
        row.pop('file_path', None)
        plan.append(dict(id=name, split=split, kind='local_interpolation', camera=row,
                         parents=[rows[j]['physical_camera'] for j in ids],
                         weights=weights.tolist(), rig_xy=pos2.tolist()))
    return plan


def gz_depth(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with gzip.open(temp, 'wb', compresslevel=3) as f:
        np.save(f, value.astype(np.float32), allow_pickle=False)
    os.replace(temp, path)


def verified_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    if sha(source) != sha(target):
        raise ValueError(f'Copy mismatch: {target}')


def initialize(out):
    rows, original_mesh, metadata = cameras(FRAME)
    parent = read(PARENT/'request.json')
    rec = deepcopy(next(r for r in parent['inventory'] if r['frame_id'] == FRAME))
    for path, key in [('mesh', 'mesh_sha256'), ('metadata', 'metadata_sha256')]:
        if sha(rec[path]) != rec[key]:
            raise ValueError('Teacher input changed')
    plan = camera_plan(rows)
    scripts = [Path(__file__).name, 'render_smooth_temporal_mesh_video.py',
               'run_view_consistent_dynamic_video.py', 'study_view_consistent_head_texture.py',
               'study_native_texture_footprint.py', 'study_early_texture_prior.py',
               'study_unwarped_head_texture.py', 'native_texture_footprint.py',
               'wide_dynamic_camera_flight.py', 'view_consistent_source_quality.py',
               'hard_surface_texture.py', 'joint_temporal_texture.py',
               'bake_joint_temporal_mesh.py', 'diffusion_mesh_repair.py',
               'temporal_texture_view_prior.py', 'render_patchmatch_camera_path.py']
    request = dict(schema_version=1, frame=FRAME, seed=SEED, plan=plan, record=rec,
                   recipe=parent['recipe'], original_mesh=str(original_mesh),
                   original_mesh_sha256=sha(original_mesh),
                   source_transforms_sha256=sha(SOURCE/FRAME/'transforms.json'),
                   calibration_sha256=sha(CALIBRATION),
                   profile_hashes={n: sha(ROOT/n) for n in ['parameters.npz', 'exposure.json', 'camera_profiles.json']},
                   source_hashes={r['physical_camera']: sha(r['file_path']) for r in rows},
                   scripts={n: sha(Path(__file__).with_name(n)) for n in scripts},
                   synthetic_train_count=300, synthetic_val_count=24,
                   teacher_reads_heldout_rgb=False, training_launched=False,
                   depth_convention='camera_z; normalized mesh coordinate units, NOT metric metres',
                   confidence_semantics='heuristic same-mesh visibility, NOT independent MVS confidence',
                   known_limitations=['teacher mesh has prior repairs and remaining defects',
                                      'outside coverage is unknown, not black supervision',
                                      '300 views are not 300 independent real observations',
                                      'single time cannot measure temporal reconstruction flicker'])
    out.mkdir(parents=True, exist_ok=True)
    if (out/'request.json').exists() and read(out/'request.json') != request:
        raise ValueError('Immutable request mismatch; use a new root')
    atomic_json(out/'request.json', request)
    verified_copy(rec['mesh'], out/'mesh/teacher.ply')
    verified_copy(original_mesh, out/'mesh/original_tsdf.ply')
    verified_copy(metadata, out/'mesh/normalization.json')
    verified_copy(CALIBRATION, out/'config/calibration.json')
    for n in request['profile_hashes']:
        verified_copy(ROOT/n, out/'config'/n)
    for n in scripts:
        verified_copy(Path(__file__).with_name(n), out/'config/scripts'/n)
    for name in ['complete.json', 'operations.json']:
        verified_copy(Path(rec['mesh']).parent/name, out/'mesh/provenance'/name)
    for n in ['complete.json', 'masks.npz', 'cameras.json']:
        verified_copy(Path(rec['source_masks']['root'])/n, out/'config/source_masks'/n)
    print('initialized train=300 val=24 frame=000973', flush=True)


def verify(out):
    r = read(out/'request.json')
    for n, digest in r['scripts'].items():
        if sha(Path(__file__).with_name(n)) != digest:
            raise ValueError('Producer changed: '+n)
    for n, digest in r['profile_hashes'].items():
        if sha(ROOT/n) != digest:
            raise ValueError('Color profile changed')
    if sha(CALIBRATION) != r['calibration_sha256']:
        raise ValueError('Calibration changed')
    return r


def triangle_keys(v, t):
    # Exact float32 positions, independent of vertex/face ordering and winding.
    p = np.asarray(v, np.float32)[t]
    order = np.lexsort((p[..., 2], p[..., 1], p[..., 0]), axis=1)
    return np.ascontiguousarray(np.take_along_axis(p, order[..., None], axis=1)).reshape(-1, 9).view('V36').ravel()


def support_products(depth, hit, count, inferred, cv2):
    rgb_valid = hit & (count > 0)
    interior = cv2.erode(rgb_valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    lo = cv2.erode(np.where(hit, depth, 1e6), np.ones((3, 3), np.uint8))
    hi = cv2.dilate(np.where(hit, depth, 0), np.ones((3, 3), np.uint8))
    smooth = (hi-lo) <= .003*np.maximum(depth, 1e-8)
    trusted = interior & smooth & (count >= 2) & ~inferred
    weight = np.minimum(count.astype(np.float32)/3., 1.)*interior
    weight[inferred] *= .25
    weight[~smooth] *= .25
    return rgb_valid, trusted, weight


def render(out, selected=None):
    import cv2
    import torch
    import open3d as o3d
    import render_smooth_temporal_mesh_video as renderer
    from run_view_consistent_dynamic_video import install
    request = verify(out)
    torch.set_num_threads(2)
    install()  # The exact previously tested hard renderer, no new RGB algorithm.
    rows, _, _ = cameras(FRAME)
    original_load, original_depth = renderer.load_sources, renderer.camera_depth
    source_cache, depth_cache = {}, {}
    source_names = {r['physical_camera'] for r in rows}
    def load(rs, manifest):
        if not source_cache:
            source_cache['images'] = original_load(rs, manifest)
        return source_cache['images']
    def cached_depth(scene, row):
        name = row['physical_camera']
        if name not in source_names:
            return original_depth(scene, row)
        key = (name, np.asarray(row['transform_matrix']).tobytes(), row['fl_x'], row['fl_y'], row['cx'], row['cy'])
        if key not in depth_cache:
            depth_cache[key] = original_depth(scene, row)
        return depth_cache[key]
    renderer.load_sources = load
    # install_source_masks captured the original function: rebind that closure's
    # referenced original_depth, not merely the public attribute.
    closure = dict(zip(renderer.render_one.__code__.co_freevars, renderer.render_one.__closure__))
    closure['original_depth'].cell_contents = cached_depth
    gather = renderer.gather_hard_rgb
    counts = []
    def capture(colors, weights, preferred):
        counts.append((weights > 0).sum(0).cpu().numpy().astype(np.uint8))
        return gather(colors, weights, preferred)
    renderer.gather_hard_rgb = capture
    mesh = o3d.io.read_triangle_mesh(str(out/'mesh/teacher.ply'))
    raw = o3d.io.read_triangle_mesh(str(out/'mesh/original_tsdf.ply'))
    original_faces = np.isin(triangle_keys(mesh.vertices, mesh.triangles), triangle_keys(raw.vertices, raw.triangles))
    np.save(out/'mesh/face_matches_original_tsdf.npy', original_faces)
    scene = renderer.scene_for(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.uint32))
    source_manifest = dict(source_images=[dict(physical_camera=r['physical_camera'], sha256=request['source_hashes'][r['physical_camera']]) for r in rows])
    for p in request['plan']:
        if selected is not None and p['id'] not in selected:
            continue
        dest = out/'synthetic/views'/p['id']
        if (dest/'complete.json').exists():
            c = read(dest/'complete.json')
            if c['request_sha256'] != sha(out/'request.json'):
                raise ValueError('View request mismatch')
            for n, digest in c['hashes'].items():
                if sha(dest/n) != digest:
                    raise ValueError('View hash mismatch')
            continue
        start = time.monotonic()
        rawroot = out/'teacher_renders'/p['id']
        (rawroot/'frames').mkdir(parents=True, exist_ok=True)
        atomic_json(rawroot/'request.json', dict(recipe=request['recipe'], dataset_request_sha256=sha(out/'request.json')))
        record = deepcopy(request['record']); record['camera'] = p['camera']
        counts.clear()
        atomic_json(out/'progress.json', dict(pid=os.getpid(), stage='render', view=p['id'], utc=time.time()))
        # An interrupted auxiliary export may reuse a finished RGB receipt. Its
        # support needs a fresh render; retain that receipt as explicit ancestry.
        rawdest = rawroot/'frames'/FRAME
        if (rawdest/'complete.json').exists():
            os.replace(rawdest/'complete.json', rawdest/'previous_complete.json')
        result = renderer.render_one(rawroot, record, source_manifest)
        d, ids, _ = original_depth(scene, p['camera'])
        saved = np.load(rawdest/'target_depth.npz')['depth']
        # Native train poses additionally have the verified source silhouette.
        hit = saved > 0
        if not np.allclose(d[hit], saved[hit], atol=1e-6):
            raise ValueError('Independent camera-depth check failed')
        count = np.zeros(hit.shape, np.uint8)
        count[hit] = np.concatenate(counts)
        inferred = np.zeros(hit.shape, bool)
        inferred[hit] = ~original_faces[ids[hit]]
        valid, trusted, weight = support_products(saved, hit, count, inferred, cv2)
        dest.mkdir(parents=True, exist_ok=True)
        verified_copy(rawdest/'prediction_native.png', dest/'rgb.png')
        verified_copy(rawdest/'source_ids.png', dest/'source_ids.png')
        gz_depth(dest/'depth_z.npy.gz', saved)
        gz_depth(dest/'depth_supervision.npy.gz', np.where(trusted, saved, 0))
        for name, value in [('mask', valid), ('depth_mask', trusted), ('mesh_hit', hit), ('inferred_geometry', inferred)]:
            Image.fromarray(value.astype(np.uint8)*255).save(dest/(name+'.png'))
        Image.fromarray(count).save(dest/'visibility_count.png')
        Image.fromarray(np.rint(weight*255).astype(np.uint8)).save(dest/'confidence.png')
        stats = dict(id=p['id'], mesh_hit_fraction=float(hit.mean()), rgb_valid_fraction=float(valid.mean()),
                     trusted_depth_fraction=float(trusted.mean()), inferred_hit_pixels=int(inferred.sum()),
                     visibility_count_median=float(np.median(count[valid])), seconds=time.monotonic()-start,
                     renderer_seconds=result['elapsed_seconds'], independent_depth_check=True)
        atomic_json(dest/'statistics.json', stats)
        atomic_json(dest/'complete.json', dict(request_sha256=sha(out/'request.json'),
                    hashes={q.name: sha(q) for q in sorted(dest.iterdir()) if q.is_file() and q.name != 'complete.json'}))
        print(f"view={p['id']} seconds={stats['seconds']:.1f} valid={valid.mean():.4f} depth={trusted.mean():.4f}", flush=True)
    atomic_json(out/'progress.json', dict(pid=os.getpid(), stage='render_subset_finished', utc=time.time()))


def transforms(frames, train, val):
    return dict(camera_model='OPENCV', orientation_override='none', frames=frames,
                train_filenames=train, val_filenames=val, test_filenames=val,
                depth_unit_scale_factor=1., depth_convention='camera_z',
                coordinate_system='mesh normalized; parser orientation/centering/scaling MUST be disabled')


def prepare_real(out):
    from render_patchmatch_camera_path import normalize_frame
    r = verify(out); rows, _, meta = cameras(FRAME)
    profiles = read(ROOT/'camera_profiles.json')['physical_cameras']
    if profiles != [x['physical_camera'] for x in rows]:
        raise ValueError('Profile ordering mismatch')
    loggain = np.load(ROOT/'parameters.npz')['log_gain']
    multipliers = np.exp(loggain-loggain.mean(0, keepdims=True))
    gain = read(ROOT/'exposure.json')['fixed_exposure_gain']
    cal = read(CALIBRATION); by_name = {f['physical_camera']: f for f in cal['frames']}
    orig = read(SOURCE/FRAME/'transforms.json')
    held = next(f for f in orig['frames'] if f['physical_camera'] == 'F004_B005_1210O9')
    held = deepcopy(held)
    for key in CAMERA_FIELDS:
        held[key] = deepcopy(by_name[held['physical_camera']][key])
    held = normalize_frame(held, cal, read(meta))
    held['file_path'] = str(SOURCE/FRAME/held['file_path'])
    entries = []
    for i, row in enumerate(rows+[held]):
        rgb = exr(row['file_path'])
        digest = sha(row['file_path'])
        if i < 62 and digest != r['source_hashes'][row['physical_camera']]:
            raise ValueError('Changed train source')
        target = out/'real/images'/f'{"train" if i < 62 else "eval"}_{i:04d}.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        response = multipliers[i] if i < 62 else np.ones(3)
        Image.fromarray(np.rint(display(rgb*response, gain)*255).clip(0,255).astype(np.uint8)).save(target)
        row = deepcopy(row); row.update(file_path=str(target.relative_to(out/'real')))
        row.pop('depth_file_path', None); row.pop('mask_path', None)
        entries.append(row)
    atomic_json(out/'real/transforms.json', transforms(entries, [x['file_path'] for x in entries[:62]], [entries[-1]['file_path']]))
    atomic_json(out/'real/protocol.json', dict(fixed_exposure=gain, train_camera_profiles=True,
                heldout_profile='identity; never fit to heldout RGB', per_image_gain=False,
                heldout_source=str(SOURCE/FRAME/next(f['file_path'] for f in orig['frames'] if f['physical_camera']==held['physical_camera'])),
                heldout_source_sha256=digest, heldout_rgb_only_read_for_real_validation_export=True,
                excluded_cameras=sorted(HELD_CAMERAS-{held['physical_camera']}), real_images_unmasked=True))
    print('real=62train+1eval exported, no EXR modified', flush=True)


def finalize(out):
    r = verify(out); frames = []; stats = []
    for p in r['plan']:
        dest = out/'synthetic/views'/p['id']; receipt = read(dest/'complete.json')
        if receipt['request_sha256'] != sha(out/'request.json'):
            raise ValueError('Mismatched view request')
        for n, digest in receipt['hashes'].items():
            if sha(dest/n) != digest:
                raise ValueError('Bad retained view hash')
        with gzip.open(dest/'depth_supervision.npy.gz', 'rb') as f:
            depth = np.load(f)
        rgb = np.asarray(Image.open(dest/'rgb.png')); mask = np.asarray(Image.open(dest/'mask.png')) > 0
        if rgb.shape != (1080, 1920, 3) or depth.shape != (1080, 1920) or not np.isfinite(depth).all() or (depth<0).any() or not mask.any():
            raise ValueError('Invalid retained arrays')
        row = deepcopy(p['camera']); base = 'views/'+p['id']+'/'
        row.update(file_path=base+'rgb.png', depth_file_path=base+'depth_supervision.npy.gz',
                   mask_path=base+'mask.png', confidence_file_path=base+'confidence.png',
                   full_depth_file_path=base+'depth_z.npy.gz', teacher_view_id=p['id'])
        frames.append(row); stats.append(read(dest/'statistics.json'))
    train = [x['file_path'] for x, p in zip(frames, r['plan']) if p['split']=='train']
    val = [x['file_path'] for x, p in zip(frames, r['plan']) if p['split']=='val']
    if len(train)!=300 or len(val)!=24 or set(train)&set(val):
        raise ValueError('Wrong splits')
    atomic_json(out/'synthetic/transforms.json', transforms(frames, train, val))
    review = out/'review'; review.mkdir(exist_ok=True)
    # All teacher views inspected in contact sheets; native crops for pilot and
    # angular extremes are separate, not claimed as 324 native detail reviews.
    for start in range(0, len(frames), 24):
        sheet = Image.new('RGB', (6*240, 4*455), '#303030'); draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(frames[start:start+24]):
            im = Image.open(out/'synthetic'/row['file_path']).transpose(Image.Transpose.ROTATE_90)
            im.thumbnail((240, 427)); x=(j%6)*240; y=(j//6)*455
            sheet.paste(im, (x,y+24)); draw.text((x+4,y+5),row['teacher_view_id'],fill='white')
        sheet.save(review/f'contact_{start//24:02d}.jpg', quality=93)
    atomic_json(out/'audit.json', dict(status='inventory_and_hashes_pass_visual_review_separate',
                request_sha256=sha(out/'request.json'), train_count=len(train), val_count=len(val),
                rgb_valid_fraction_min_median_max=np.quantile([s['rgb_valid_fraction'] for s in stats],[0,.5,1]).tolist(),
                trusted_depth_fraction_min_median_max=np.quantile([s['trusted_depth_fraction'] for s in stats],[0,.5,1]).tolist(),
                total_render_seconds=sum(s['renderer_seconds'] for s in stats),
                training_launched=False, stats=stats))
    hashes = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()
              and p.parts[len(out.parts)] in ['mesh','config','synthetic','real']}
    atomic_json(out/'dataset_hashes.json', hashes)
    print('audit train=300 val=24 hashes=pass', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['init','render','real','finalize'])
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--views', nargs='+')
    a = p.parse_args()
    if a.action=='init': initialize(a.output)
    elif a.action=='render': render(a.output, a.views)
    elif a.action=='real': prepare_real(a.output)
    else: finalize(a.output)


if __name__=='__main__': main()
