"""Prepare the immutable Luster 000470 subject dataset and train-only bounds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nerfstudio.cameras.camera_utils import auto_orient_and_center_poses
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text, read_images_text

EVAL_IDS = {12, 95, 150}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resize_intrinsics(params, original, resized):
    sx, sy = resized[0] / original[0], resized[1] / original[1]
    fx, fy, cx, cy = params
    return [fx * sx, fy * sy, cx * sx, cy * sy]


def ray_box_hits(origin, directions, bounds):
    direction = np.where(np.abs(directions) < 1e-12, 1e-12, directions)
    t0 = (bounds[0] - origin) / direction
    t1 = (bounds[1] - origin) / direction
    return np.minimum(t0, t1).max(-1) <= np.maximum(t0, t1).min(-1), np.maximum(t0, t1).min(-1) > 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--grid', type=int, default=256)
    p.add_argument('--search-radius-factor', type=float, default=2.6)
    p.add_argument('--reuse-images', action='store_true', help='Resume this preparation after an interrupted audit')
    p.add_argument('--rebuild-bounds', action='store_true', help='Re-audit bounds before any scene training')
    args = p.parse_args()
    root = args.root; source = root / 'source'; out = root / 'data'
    if (out / 'complete.json').exists() and not args.rebuild_bounds:
        raise RuntimeError('Prepared dataset already exists')
    for folder in ['images', 'original_hd', 'masks', 'reviews']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    cameras = read_cameras_text(source / 'frame/sparse/text/cameras.txt')
    images = read_images_text(source / 'frame/sparse/text/images.txt')
    if len(images) != 165 or len(cameras) != 165:
        raise ValueError('Expected all 165 cameras')
    rows = []; poses = []; raw_w2cs = []; masks = []; contained = []; manifest = {}
    for _, image in sorted(images.items()):
        c = cameras[image.camera_id]
        cid = int(image.name.split('_')[1])
        rgb_path = source / 'frame/images' / image.name
        mask_path = source / 'masks' / f'cam_{cid}' / '000470.png'
        # SAM3 cam164 has 2984 rows versus the calibrated 3000-row image.
        # Use the provided full-size plate-difference mask rather than invent
        # a crop offset or stretch the silhouette without calibration evidence.
        if cid == 164:
            mask_path = source / 'plate_difference_cam164.png'
        rgb_size = Image.open(rgb_path).size
        mask = np.array(Image.open(mask_path).convert('L'))
        if c.model != 'PINHOLE' or rgb_size != (c.width, c.height) or mask.shape != (c.height, c.width):
            raise ValueError(f'Camera/image/mask mismatch: {cid}')
        # Reviewed false positives on the far-left background stands. Neither
        # subject touches this strip; source masks remain unchanged on disk.
        if cid in {20,36}:
            mask[:, :680] = 0
        wh = (round(c.width * 1920 / max(c.width, c.height)), round(c.height * 1920 / max(c.width, c.height)))
        coverage = cv2.resize(mask, wh, interpolation=cv2.INTER_AREA)
        foreground = coverage >= 128
        contained.append(not (foreground[:8].any() or foreground[-8:].any() or foreground[:,:8].any() or foreground[:,-8:].any()))
        name = f'cam_{cid:03d}_000470.png'
        targets = [out / folder / name for folder in ['images', 'original_hd', 'masks']]
        if not args.reuse_images or not all(path.exists() and Image.open(path).size == wh for path in targets):
            rgb = np.array(Image.open(rgb_path).convert('RGB'))
            original = cv2.resize(rgb, wh, interpolation=cv2.INTER_AREA)
            masked = cv2.resize(rgb.astype(np.float32) * (mask[..., None] / 255.), wh, interpolation=cv2.INTER_AREA)
            Image.fromarray(np.rint(masked).clip(0, 255).astype(np.uint8)).save(targets[0], compress_level=1)
            Image.fromarray(original).save(targets[1], compress_level=1)
            Image.fromarray(coverage).save(targets[2], compress_level=1)
        fx, fy, cx, cy = resize_intrinsics(c.params, (c.width, c.height), wh)
        w2c = np.eye(4); w2c[:3, :3] = image.qvec2rotmat(); w2c[:3, 3] = image.tvec
        c2w = np.linalg.inv(w2c); c2w[:3, 1:3] *= -1
        poses.append(c2w); raw_w2cs.append(w2c)
        small = cv2.resize(coverage, (round(wh[0] / 4), round(wh[1] / 4)), interpolation=cv2.INTER_AREA)
        masks.append(cv2.dilate((small >= 128).astype(np.uint8), np.ones((3, 3), np.uint8)))
        rows.append(dict(file_path=f'images/{name}', physical_camera=f'cam_{cid:03d}', camera_id=cid,
                         w=wh[0], h=wh[1], fl_x=fx, fl_y=fy, cx=cx, cy=cy))
        for path in [rgb_path, mask_path]: manifest[str(path.relative_to(source))] = sha(path)
        print(f'prepared={len(rows)}/165', flush=True) if len(rows) % 25 == 0 else None
    for path in (source / 'frame/sparse/text').iterdir():
        manifest[str(path.relative_to(source))] = sha(path)
    manifest['bounds_8s.json'] = sha(source / 'bounds_8s.json')
    write(source / 'local_manifest.json', manifest)
    # Derive one leader-style coordinate transform from train poses only.
    train = [i for i, r in enumerate(rows) if r['camera_id'] not in EVAL_IDS]
    poses = np.array(poses)
    normalized, transform = auto_orient_and_center_poses(torch.tensor(poses[train], dtype=torch.float32), method='up', center_method='focus')
    scale = 1 / float(normalized[:, :3, 3].abs().max())
    transform4 = np.eye(4); transform4[:3] = transform.numpy()
    converted = transform4[None] @ poses
    converted[:, :3, 3] *= scale
    for row, pose in zip(rows, converted): row['transform_matrix'] = pose.tolist()
    # Conservative visual hull: only visible projections vote. Small mask errors
    # may disagree in up to 2% of visible train cameras; require ten witnesses.
    sphere = json.loads((source / 'bounds_8s.json').read_text())['subject_sphere']
    center = np.array(sphere['center']); radius = sphere['radius'] * args.search_radius_factor
    n = args.grid; pitch = 2 * radius / n
    axes = [torch.linspace(float(v-radius+pitch/2), float(v+radius-pitch/2), n, device='cuda') for v in center]
    points = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    visible = torch.zeros(len(points), device='cuda', dtype=torch.int16)
    outside = torch.zeros_like(visible)
    for j, i in enumerate(train):
        w2c = torch.tensor(raw_w2cs[i], device='cuda', dtype=torch.float32)
        xyz = points @ w2c[:3, :3].T + w2c[:3, 3]
        row = rows[i]; m = masks[i]; mh, mw = m.shape
        x = (xyz[:, 0] / xyz[:, 2] * row['fl_x'] + row['cx']) * mw / row['w']
        y = (xyz[:, 1] / xyz[:, 2] * row['fl_y'] + row['cy']) * mh / row['h']
        valid = (xyz[:, 2] > 0) & (x >= 0) & (y >= 0) & (x < mw) & (y < mh)
        support = torch.tensor(m, device='cuda')[y.long().clamp(0, mh-1), x.long().clamp(0, mw-1)] > 0
        if contained[i]:
            # A complete silhouette also constrains points outside its frustum.
            visible += 1; outside += (~(valid & support)).to(torch.int16)
        else:
            visible += valid.to(torch.int16); outside += (valid & ~support).to(torch.int16)
        if j % 30 == 0: print(f'hull_camera={j+1}/{len(train)}', flush=True)
    keep = (visible >= 10) & (outside.float() <= visible.float() * .02)
    labels, component_count = ndimage.label(keep.reshape(n,n,n).cpu().numpy())
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    component_rows = []
    for label in np.argsort(sizes)[-10:][::-1]:
        if not sizes[label]: continue
        ijk = np.argwhere(labels == label)
        component_rows.append(dict(label=int(label),voxels=int(sizes[label]),
                                   low=(center-radius+(ijk.min(0)+.5)*pitch).tolist(),
                                   high=(center-radius+(ijk.max(0)+.5)*pitch).tolist()))
    write(out/'hull_components.json',dict(components=component_rows,total=component_count))
    keep = torch.tensor((labels == int(sizes.argmax())).reshape(-1),device='cuda')
    hull = points[keep].cpu().numpy()
    if len(hull) < 100: raise RuntimeError('Visual hull empty or implausibly small')
    edge_margin = np.minimum(hull - (center-radius), center+radius-hull).min()
    if edge_margin < pitch * 1.1: raise RuntimeError('Visual hull reaches initial search bounds; enlarge search')
    nhull = (hull @ transform4[:3, :3].T + transform4[:3, 3]) * scale
    low = nhull.min(0) - pitch * scale; high = nhull.max(0) + pitch * scale
    padding = (high-low) * .05
    bounds = np.array([low-padding, high+padding])
    coverage_rows = []
    for row, pose in zip(rows, converted):
        mask = np.array(Image.open(out / 'masks' / Path(row['file_path']).name)) >= 128
        yy, xx = np.nonzero(mask)
        dirs = np.stack([(xx+.5-row['cx']) / row['fl_x'], -(yy+.5-row['cy']) / row['fl_y'], -np.ones(len(xx))], -1) @ pose[:3, :3].T
        hit, positive = ray_box_hits(pose[:3, 3], dirs, bounds)
        coverage_rows.append(dict(camera=row['physical_camera'], foreground_pixels=len(xx), aabb_hit_fraction=float((hit & positive).mean())))
    np.savez_compressed(out / 'hull.npz', points=nhull, raw_points=hull)
    audit = dict(grid=n, pitch_world=pitch, hull_voxels=len(hull), initial_search_margin=float(edge_margin),
                 bounds=bounds.tolist(), padding_fraction=.05, consensus=.98, minimum_witnesses=10,
                 train_cameras=len(train), normalization=transform4.tolist(), scale=scale, coverage=coverage_rows,
                 mask_exception={'cam_164':'full-size plate-difference mask; SAM3 has 2984 rather than 3000 rows'})
    audit['fully_contained_train_cameras']=[rows[i]['physical_camera'] for i in train if contained[i]]
    write(out / 'bounds_audit.json', audit)
    meta = dict(camera_model='OPENCV', coordinate_system='train_normalized_opengl', orientation_override='none',
                blur_aabb=bounds.tolist(), frames=rows,
                train_filenames=[r['file_path'] for r in rows if r['camera_id'] not in EVAL_IDS],
                val_filenames=[r['file_path'] for r in rows if r['camera_id'] in EVAL_IDS],
                test_filenames=[r['file_path'] for r in rows if r['camera_id'] in EVAL_IDS])
    write(out / 'transforms.json', meta)
    # All masks are reviewed; originals stay available for judging boundaries.
    for page in range(3):
        sheet = Image.new('RGB', (11*150, 5*220), (30, 30, 30)); draw = ImageDraw.Draw(sheet)
        for local, row in enumerate(rows[page*55:(page+1)*55]):
            name = Path(row['file_path']).name
            im = Image.open(out / 'original_hd' / name); im.thumbnail((148, 196))
            mask = Image.open(out / 'masks' / name).resize(im.size)
            im = Image.composite(Image.blend(im, Image.new('RGB', im.size, (0, 255, 0)), .25), im, mask)
            x=(local%11)*150; y=(local//11)*220
            sheet.paste(im, (x, y+20)); draw.text((x+2,y+2), row['physical_camera'], fill='white')
        sheet.save(out / 'reviews' / f'masks_{page}.jpg')
    write(out / 'complete.json', dict(images=165, train=162, eval=3, transforms_sha256=sha(out/'transforms.json'),
                                     minimum_aabb_hit=min(r['aabb_hit_fraction'] for r in coverage_rows)))
    print(json.dumps(json.loads((out/'complete.json').read_text())), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
