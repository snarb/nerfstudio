"""Posthoc native strict/interpolation comparison; never admission input."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from study_multiview_face_prior import read, save, sha, portrait_to_native
from study_confidence_depth_prior import unproject
from study_mhr_local_head_prior import RGB, FRAME
from admit_mhr_local_patch_depth import Scene2

ROOT = Path('/mnt/data/dec5_mhr_local_confidence_admission')


def difference(a, b):
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    delta = np.zeros(a.shape)
    delta[both] = abs(a[both] - b[both])
    changed = (fa != fb) | (delta > 1e-8)
    return changed, dict(changed_pixels=int(changed.sum()),
                         new_interpolated_hits=int((fb & ~fa).sum()),
                         lost_interpolated_hits=int((fa & ~fb).sum()),
                         finite_depth_difference_pixels=int((delta > 1e-8).sum()),
                         maximum_common_depth_difference=float(delta.max()),
                         p95_nonzero_depth_difference=float(np.percentile(delta[delta > 1e-8], 95)) if (delta > 1e-8).any() else None)


def main():
    import open3d as o3d
    from bake_joint_temporal_mesh import camera_depth
    dest = ROOT / 'branch_difference'
    dest.mkdir(exist_ok=False)
    request = read(ROOT / 'request.json')
    bindings = {str(ROOT / 'request.json'): sha(ROOT / 'request.json')}
    rows = [i['camera'] for i in read(RGB / FRAME / 'input.json')['inputs']]
    boxes = {r['camera']: r for r in read(RGB / 'inference.json')['records'] if r['frame'] == FRAME and r['detected'] == 1}
    parent = Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')
    spots = Path('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')
    moving = next(r['camera'] for r in read(parent)['inventory'] if r['frame_id'] == FRAME)
    x0, y0, x1, y1 = next(r['bbox_inclusive'] for r in read(spots)['selected_components'] if r['frame_id'] == FRAME)
    roi = (x0, y0, x1 + 1, y1 + 1)
    crop = (x0 - 65, y0 - 65, x1 + 66, y1 + 66)
    for path in [parent, spots, RGB / 'inference.json', RGB / FRAME / 'input.json']:
        bindings[str(path)] = sha(path)
    statistics, files = [], []
    for arm in request['arms']:
        models = []
        for branch in ['strict', 'interpolated']:
            path = ROOT / arm / branch / 'mesh.ply'
            result = read(path.parent / 'result.json')
            assert sha(path) == result['hashes']['mesh.ply']
            bindings[str(path)] = sha(path)
            mesh = o3d.io.read_triangle_mesh(str(path))
            mesh.compute_triangle_normals()
            models.append((Scene2(np.asarray(mesh.vertices), np.asarray(mesh.triangles)), np.asarray(mesh.triangle_normals)))
        for prefix in ['G004_B', 'M004_B', 'E004_B', 'requested_hole']:
            arrays, clays = [], []
            if prefix != 'requested_hole':
                row = next(r for r in rows if r['physical_camera'].startswith(prefix))
                name = row['physical_camera']
                bx0, by0, bx1, by1 = boxes[name]['native_review_box']
                by1 = min(1550, by1 + 100)
                yy, xx = np.mgrid[by0:by1, bx0:bx1]
                xy = portrait_to_native(np.c_[xx.ravel(), yy.ravel()])
                center = np.asarray(row['transform_matrix'])[:3, 3]
                direction = unproject(row, xy[:, 0], xy[:, 1], np.ones(len(xy))) - center
                unit = direction / np.linalg.norm(direction, axis=1, keepdims=True)
                rays = o3d.core.Tensor(np.c_[np.broadcast_to(center, direction.shape), direction].astype(np.float32))
                for scene, normals in models:
                    hit = scene.cast_rays(rays)
                    d, ids = hit['t_hit'].numpy(), hit['primitive_ids'].numpy()
                    ok = np.isfinite(d)
                    color = np.full((len(d), 3), 20, np.uint8)
                    color[ok] = (70 + 170 * abs(np.sum(normals[ids[ok]] * -unit[ok], axis=1)))[:, None]
                    arrays.append(d.reshape(yy.shape))
                    clays.append(color.reshape(*yy.shape, 3))
            else:
                name = prefix
                for scene, normals in models:
                    d, ids, _ = camera_depth(scene, moving)
                    ok = np.isfinite(d)
                    color = np.full((*d.shape, 3), 20, np.uint8)
                    color[ok] = (60 + 170 * abs(normals[ids[ok]] @ np.array([.3, .4, .866])))[:, None]
                    arrays.append(np.rot90(d))
                    clays.append(np.rot90(color))
            changed, stat = difference(*arrays)
            stat.update(arm=arm, camera=name, scope='full_moving_frame' if prefix == 'requested_hole' else 'native_face_crop')
            if prefix == 'requested_hole':
                rx0, ry0, rx1, ry1 = roi
                _, stat['fixed_roi'] = difference(*(a[ry0:ry1, rx0:rx1] for a in arrays))
                cx0, cy0, cx1, cy1 = crop
                clays = [a[cy0:cy1, cx0:cx1] for a in clays]
                changed = changed[cy0:cy1, cx0:cx1]
            statistics.append(stat)
            marked = clays[1].copy()
            marked[changed] = [255, 40, 180]
            images = [Image.fromarray(a) for a in [*clays, marked]]
            w, h = images[0].size
            panel = Image.new('RGB', (w * 3, h + 24))
            draw = ImageDraw.Draw(panel)
            for index, (label, im) in enumerate(zip(['strict', 'interpolated', 'changed pixels (magenta)'], images)):
                panel.paste(im, (index * w, 24))
                draw.text((index * w + 2, 4), label, fill='white')
            path = dest / f'{arm}_{name}.png'
            panel.save(path)
            files.append(dict(path=str(path), sha256=sha(path)))
        print(arm, 'branch differences reviewed', flush=True)
    save(dest / 'result.json', dict(statistics=statistics, files=files, input_hashes=bindings,
                                    script_sha256=sha(__file__), target_used_posthoc_only=True,
                                    native_lattice=True, depth_equality_tolerance=1e-8,
                                    production_accepted=False))


if __name__ == '__main__':
    main()
