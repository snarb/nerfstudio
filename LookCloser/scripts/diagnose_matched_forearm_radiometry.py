"""Train-only same-surface radiometry; diagnostic, never a texture correction.

Use actual target triangle barycentrics, exact bilinear linear-EXR sampling,
and the existing four-tap mesh visibility gate. Compare fixed exposure with
and without frozen camera RGB gains. A valid warp is not proof of true shape.
"""
from pathlib import Path
import itertools
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import (
    ROOT as CAL, cameras, read, sha, atomic_json, exr, display, project,
)
from calibrated_depth_witness import response_gains
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from review_jaw_repair_transfer import verified_image, panel
from build_multiview_forearm_admission import ROOT as BASE, FRAME

ROOT = Path('/mnt/data/dec5_matched_forearm_radiometry')
BOX = (90, 1650, 430, 1920)
CAMERAS = [26, 31, 32, 33, 37, 42]


def bilinear(image, uv):
    """Border-clamped bilinear sample, matching grid_sample align_corners=False."""
    a = np.asarray(image)
    p = np.asarray(uv, float)
    if a.ndim not in (2, 3) or p.shape[-1] != 2 or not np.isfinite(p).all():
        raise ValueError('Invalid image or sampling coordinates')
    x = np.clip(p[..., 0], 0, a.shape[1]-1)
    y = np.clip(p[..., 1], 0, a.shape[0]-1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0+1, a.shape[1]-1), np.minimum(y0+1, a.shape[0]-1)
    wx, wy = x-x0, y-y0
    if a.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    return (a[y0, x0]*(1-wx)+a[y0, x1]*wx)*(1-wy)+(a[y1, x0]*(1-wx)+a[y1, x1]*wx)*wy


def paired_summary(a, b, valid):
    """Descriptive same-point differences, not a quality score or fitted profile."""
    x, y = np.asarray(a)[valid], np.asarray(b)[valid]
    if len(x) < 30:
        return dict(count=len(x), status='insufficient_overlap')
    delta = (y-x)*255
    return dict(count=len(x), median_rgb_delta=np.median(delta, axis=0).tolist(),
        median_absolute_rgb_difference=float(np.median(np.abs(delta))),
        p90_absolute_rgb_difference=float(np.percentile(np.abs(delta), 90)),
        median_log_ratio=np.median(np.log(np.maximum(y, 1e-7)/np.maximum(x, 1e-7)), axis=0).tolist())


def run():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'request.json').exists():
        raise ValueError('Use a fresh output root; incomplete results are not resumable')
    rows, _, meta = cameras(FRAME)
    image, receipt = verified_image(BASE/'rgb/moving', FRAME)
    assert receipt['source_cameras'] == [r['physical_camera'] for r in rows]
    meshpath = BASE/'mesh.ply'
    assert sha(meshpath) == receipt['mesh_sha256']
    q = read(BASE/'rgb/moving/request.json')
    maskspec = q['inventory'][0]['source_masks'];mr = Path(maskspec['root'])
    for n, key in [('masks.npz', 'masks_sha256'), ('cameras.json', 'cameras_sha256')]:
        assert sha(mr/n) == maskspec[key]
    masknames = read(mr/'cameras.json');masks = np.load(mr/'masks.npz')['masks']
    gains = response_gains(np.load(CAL/'parameters.npz')['log_gain'])
    profiles = read(CAL/'camera_profiles.json')
    assert profiles['physical_cameras'] == receipt['source_cameras']
    np.testing.assert_allclose(gains, profiles['rgb_gain'], rtol=1e-6)
    exposure = read(CAL/'exposure.json')['fixed_exposure_gain']
    assert exposure == receipt['fixed_exposure']
    files = [meshpath, BASE/'rgb/moving/request.json', BASE/'rgb/moving/frames'/FRAME/'complete.json',
             CAL/'parameters.npz', CAL/'camera_profiles.json', CAL/'exposure.json', Path(meta),
             mr/'masks.npz', mr/'cameras.json']
    files += [Path(rows[i]['file_path']) for i in CAMERAS]
    files += [Path(__file__).resolve().with_name(n) for n in [Path(__file__).name,
        'joint_temporal_texture.py', 'calibrated_depth_witness.py', 'bake_joint_temporal_mesh.py', 'diffusion_mesh_repair.py']]
    request = dict(frame=FRAME, source_indices=CAMERAS, box=BOX, target_camera=receipt['camera'],
        rows=[rows[i] for i in CAMERAS], hashes={str(p):sha(p) for p in files},
        target_rgb_used=False, geometry_changed=False, fitting_or_correction=False,
        fixed_exposure=exposure, source_gain=gains[CAMERAS].tolist(),
        notes='Same-point image comparison on inferred geometry. Mask/depth agreement does not prove anatomical correspondence.')
    atomic_json(ROOT/'request.json', request)
    mesh = o3d.io.read_triangle_mesh(str(meshpath));v, t = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles)
    scene = scene_for(v, t)
    d, ids, b = camera_depth(scene, receipt['camera'])
    saved = np.load(BASE/'rgb/moving/frames'/FRAME/'target_depth.npz')['depth']
    np.testing.assert_array_equal(np.isfinite(d), saved > 0)
    np.testing.assert_allclose(d[np.isfinite(d)], saved[saved > 0], atol=2e-6)
    x0, y0, x1, y1 = BOX
    d, ids, b = [np.rot90(a)[y0:y1, x0:x1] for a in [d, ids, b]]
    hit = np.isfinite(d)
    points = np.zeros((*d.shape, 3), np.float32)
    weights = np.column_stack((1-b[hit].sum(1), b[hit]))
    points[hit] = (v[t[ids[hit]]]*weights[..., None]).sum(1)
    uv, z = project(points[hit], [rows[i] for i in CAMERAS])
    sourceids = np.rot90(np.array(Image.open(BASE/'rgb/moving/frames'/FRAME/'source_ids.png')))[y0:y1, x0:x1]
    rgb_crop = image[y0:y1, x0:x1]
    linear = [];corrected = [];uncorrected = [];valids = [];replays = []
    native_panels = []
    # Fixed diagnostic points on the photographed forearm, not learned landmarks.
    markers = [(285, 1705), (280, 1750), (240, 1820), (180, 1870)]
    marker_colors = [(255, 70, 70), (70, 255, 70), (70, 140, 255), (255, 220, 70)]
    for ci, ri in enumerate(CAMERAS):
        row = rows[ri];raw = exr(row['file_path'])
        depth, _, _ = camera_depth(scene, row)
        depth = np.where(np.isfinite(depth), depth, 0)
        depth[masks[masknames.index(row['physical_camera'])] == 0] = 0
        p = uv[ci];zz = z[ci];sampled = bilinear(depth, p)
        good = (zz > 0) & (sampled > 0) & (np.abs(sampled-zz) < .0015*zz)
        good &= (p[:, 0] > 2) & (p[:, 0] < 1917) & (p[:, 1] > 2) & (p[:, 1] < 1077)
        for dx, dy in [(0,0), (1,0), (0,1), (1,1)]:
            tap = bilinear(depth, np.floor(p)+[dx, dy])
            good &= (tap > 0) & (np.abs(tap-zz) < .003*zz)
        lr = bilinear(raw, p).clip(0)
        lin = np.zeros_like(points);lin[hit] = lr
        ca = np.zeros_like(points);ca[hit] = display(lr*gains[ri], exposure)
        un = np.zeros_like(points);un[hit] = display(lr, exposure)
        ok = np.zeros_like(hit);ok[hit] = good
        chosen = (sourceids == ri) & hit
        error = np.abs(np.rint(ca[chosen]*255)-rgb_crop[chosen].astype(float))
        replays.append(dict(camera=row['physical_camera'], selected_pixels=int(chosen.sum()),
            max_uint8_error=float(error.max()) if error.size else None,
            selected_not_four_tap_visible=int((chosen & ~ok).sum())))
        # CUDA grid normalization differs at sub-millipixel precision from float64 replay.
        assert not error.size or error.max() <= 2, replays[-1]
        linear.append(lin);corrected.append(ca);uncorrected.append(un);valids.append(ok)
        shown = np.rint(display(raw*gains[ri], exposure)*255).clip(0,255).astype(np.uint8)
        portrait = Image.fromarray(np.rot90(shown));draw = ImageDraw.Draw(portrait)
        xy = []
        for (mx,my), color in zip(markers, marker_colors):
            if not hit[my-y0, mx-x0]:
                continue
            puv, _ = project(points[my-y0, mx-x0][None], [row]);u,w = puv[0,0]
            px, py = float(w), float(1919-u)
            xy.append((px,py));draw.ellipse((px-4, py-4, px+4, py+4), outline=color, width=2)
        assert xy
        lo = np.floor(np.min(xy,0)-55).astype(int);hi = np.ceil(np.max(xy,0)+55).astype(int)
        cropbox = (max(0,lo[0]),max(0,lo[1]),min(1080,hi[0]),min(1920,hi[1]))
        portrait.crop(cropbox).save(ROOT/(row['physical_camera']+'_native.png'))
        native_panels.append(dict(camera=row['physical_camera'], crop=list(map(int,cropbox)), marker_pixels=xy))
        print(row['physical_camera'], 'visible', int(ok.sum()), 'replay', replays[-1], flush=True)
    linear, corrected, uncorrected, valids = map(np.stack, [linear, corrected, uncorrected, valids])
    records = []
    yy = np.indices(hit.shape)[0]+y0
    for a, b in itertools.combinations(range(len(CAMERAS)), 2):
        warm_a = (corrected[a,...,0]-corrected[a,...,2])*255 > 8
        warm_b = (corrected[b,...,0]-corrected[b,...,2])*255 > 8
        domain = valids[a] & valids[b] & warm_a & warm_b
        for band, lo, hi in [('all',1650,1920), ('wrist',1650,1750), ('middle',1750,1830), ('lower',1830,1920)]:
            use = domain & (yy >= lo) & (yy < hi)
            records.append(dict(a=CAMERAS[a], b=CAMERAS[b], band=band,
                fixed_only=paired_summary(uncorrected[a], uncorrected[b], use),
                profiled=paired_summary(corrected[a], corrected[b], use),
                raw_linear=paired_summary(linear[a], linear[b], use)))
    for title, arrays in [('fixed_only',uncorrected), ('profiled',corrected)]:
        views = [rgb_crop] + [np.rint(np.where(valids[i,...,None], a, 0)*255).clip(0,255).astype(np.uint8) for i,a in enumerate(arrays)]
        panel(ROOT/(title+'_same_surface.png'), views, ['current RGB']+[str(i)+': '+rows[i]['physical_camera'] for i in CAMERAS], (0,0,x1-x0,y1-y0))
    np.savez_compressed(ROOT/'samples.npz', points=points, hit=hit, source=sourceids,
        linear=linear, profiled=corrected, fixed_only=uncorrected, valid=valids, uv=uv, z=z)
    atomic_json(ROOT/'result.json', dict(request_sha256=sha(ROOT/'request.json'), records=records,
        replay=replays, native_panels=native_panels, diagnostic_not_quality_metrics=True,
        visual_status='pending', production_updated=False))
    files = sorted(p for p in ROOT.iterdir() if p.is_file())
    atomic_json(ROOT/'complete.json', dict(hashes={str(p):sha(p) for p in files}))
    print('Completed same-surface radiometry', flush=True)


if __name__ == '__main__':
    run()
