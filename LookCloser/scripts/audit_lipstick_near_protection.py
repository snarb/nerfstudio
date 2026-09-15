"""Replay near-depth protection and show the actual protecting train pixels."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from diagnose_lipstick_near_protection import ROOT, DEPTH_ROOT, DIAGNOSTIC
from joint_temporal_texture import read, sha, atomic_json, project, cameras, exr, display, ROOT as COLOR
from calibrated_depth_witness import response_gains
from study_confidence_depth_prior import load_real, project_integer, support, unproject


def main():
    result = read(ROOT/'result.json')
    bindings = dict(result['input_hashes'], **result['helpers'])
    bindings[str(Path(__file__).with_name('calibrated_depth_witness.py'))] = sha(Path(__file__).with_name('calibrated_depth_witness.py'))
    bindings[str(Path(__file__).with_name('diagnose_lipstick_near_protection.py'))] = result['script_sha256']
    bindings[str(ROOT/'evidence.npz')] = result['evidence_sha256']
    for path, digest in bindings.items():
        assert sha(path) == digest, path
    rows, depths, receipt = load_real(DEPTH_ROOT, '000995')
    assert receipt == result['depth_receipt']
    evidence = np.load(ROOT/'evidence.npz')
    flat = evidence['points'].reshape(-1, 3)
    checks = 0
    for ci, (row, depth) in enumerate(zip(rows, depths)):
        uv, z = project_integer(row, flat)
        xy = np.rint(uv).astype(int)
        valid = (np.isfinite(uv).all(1) & np.isfinite(z) & (z > 0) &
                 (xy[:, 0] >= 0) & (xy[:, 0] < depth.shape[1]) &
                 (xy[:, 1] >= 0) & (xy[:, 1] < depth.shape[0]))
        index = np.flatnonzero(valid)
        d = depth[xy[index, 1], xy[index, 0]]
        near = np.zeros(len(flat), bool)
        near[index] = np.isfinite(d) & (d > 0) & (abs(d-z[index]) <= .0015)
        np.testing.assert_array_equal(near.reshape(-1, 4), evidence['near'][ci])
        take = np.flatnonzero(near)
        measured = depth[xy[take, 1], xy[take, 0]]
        counts = np.zeros(len(flat), int)
        if len(take):
            counts[take], _ = support(unproject(row, xy[take, 0], xy[take, 1], measured), row, rows, depths)
        np.testing.assert_array_equal(counts.reshape(-1, 4), evidence['other_view_counts'][ci])
        checks += len(take)
    for record in result['threshold_controls']:
        mask = evidence['near'] & (evidence['other_view_counts'] >= record['minimum_other_near_views'])
        kept = np.any(mask, axis=0).any(1)
        remaining = ~evidence['previously_removed']
        assert int(kept.sum()) == record['protected_diagnostic_faces']
        assert int((kept & remaining).sum()) == record['remaining_protected']
    source_rows, _, _ = cameras('000995')
    params = np.load(COLOR/'parameters.npz')['log_gain']
    gains = response_gains(params)
    exposure = read(COLOR/'exposure.json')['fixed_exposure_gain']
    color_receipt = read(DIAGNOSTIC/'request.json')['rgb_receipt']
    for name, key in [('parameters.npz', 'parameters_sha256'), ('camera_profiles.json', 'profiles_sha256'), ('exposure.json', 'exposure_sha256')]:
        assert sha(COLOR/name) == color_receipt[key]
        bindings[str(COLOR/name)] = color_receipt[key]
    representative = result['representative_triangle']
    triangle = evidence['points'][evidence['triangle_ids'] == representative][0]
    dest = ROOT/'review'
    dest.mkdir(exist_ok=False)
    files = []
    for camera in sorted(set(r['camera'] for r in result['representative_near_observations'])):
        ci, row = next((i, r) for i, r in enumerate(source_rows) if r['physical_camera'] == camera)
        source = row['file_path']
        assert sha(source) == color_receipt['source_rgb_hashes'][source]
        bindings[source] = sha(source)
        rgb = np.rint(display(exr(source)*gains[ci], exposure)*255).clip(0, 255).astype(np.uint8)
        uv, _ = project(triangle, [row]); uv = uv[0]
        observations = [r for r in result['representative_near_observations'] if r['camera'] == camera]
        pixels = np.array([r['pixel'] for r in observations])
        bounds = np.concatenate([uv, pixels])
        lo = np.maximum(np.floor(bounds.min(0)-45).astype(int), [0, 0])
        hi = np.minimum(np.ceil(bounds.max(0)+46).astype(int), [1920, 1080])
        x0, y0 = lo; x1, y1 = hi
        native = Image.fromarray(rgb).crop((x0, y0, x1, y1))
        marked = native.copy(); draw = ImageDraw.Draw(marked)
        local = uv-lo
        draw.line([tuple(p) for p in local[:3]]+[tuple(local[0])], fill='red', width=1)
        for p in pixels-lo:
            x, y = p; draw.ellipse((x-2, y-2, x+2, y+2), outline='lime', width=1)
        images = [im.transpose(Image.Transpose.ROTATE_90) for im in [native, marked]]
        w, h = images[0].size
        panel = Image.new('RGB', (w*2, h+25)); draw = ImageDraw.Draw(panel)
        for i, (im, title) in enumerate(zip(images, ['actual train RGB', 'triangle / measured tap'])):
            panel.paste(im, (i*w, 25)); draw.text((i*w+2, 4), title, fill='white')
        path = dest/(camera+'.png'); panel.save(path)
        files.append(dict(path=str(path), sha256=sha(path), camera=camera,
            native_crop=[int(x0), int(y0), int(x1), int(y1)], source_sha256=sha(source)))
    atomic_json(ROOT/'audit.json', dict(status='passed', result_sha256=sha(ROOT/'result.json'),
        checked_bindings=bindings, near_measurements_replayed=checks,
        camera_sample_checks=len(rows)*len(flat), corroboration_helper_reused=True,
        independent_near_pixel_arithmetic=True, threshold_summaries_replayed=True,
        files=files, script_sha256=sha(__file__), visual_status='pending', geometry_changed=False))
    print('near protection audit passed', checks, 'near observations', len(rows)*len(flat), 'camera samples', flush=True)


if __name__ == '__main__':
    main()
