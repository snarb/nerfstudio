"""Supplementary full-frame locations omitted by the fixed jaw review crop."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
from review_mhr_admission_branch_difference import ROOT, difference
from admit_mhr_local_patch_depth import Scene2
from study_multiview_face_prior import read, save, sha


def main():
    import open3d as o3d
    from bake_joint_temporal_mesh import camera_depth
    dest = ROOT / 'branch_difference_locations'
    dest.mkdir(exist_ok=False)
    source = Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')
    camera = next(r['camera'] for r in read(source)['inventory'] if r['frame_id'] == '001193')
    bindings = {str(source): sha(source)}
    records, files = [], []
    for arm in ['smooth025', 'smooth100', 'smooth400']:
        arrays, images = [], []
        for branch in ['strict', 'interpolated']:
            path = ROOT / arm / branch / 'mesh.ply'
            bindings[str(path)] = sha(path)
            mesh = o3d.io.read_triangle_mesh(str(path))
            mesh.compute_triangle_normals()
            normals = np.asarray(mesh.triangle_normals)
            scene = Scene2(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
            d, ids, _ = camera_depth(scene, camera)
            ok = np.isfinite(d)
            color = np.full((*d.shape, 3), 20, np.uint8)
            color[ok] = (60 + 170 * abs(normals[ids[ok]] @ np.array([.3, .4, .866])))[:, None]
            arrays.append(np.rot90(d))
            images.append(np.rot90(color))
        changed, stats = difference(*arrays)
        yy, xx = np.nonzero(changed)
        records.append(dict(arm=arm, statistics=stats, portrait_xy=np.c_[xx, yy].tolist(),
                            strict_depth=arrays[0][yy, xx].tolist(), interpolated_depth=arrays[1][yy, xx].tolist()))
        components, count = label(changed, np.ones((3, 3)))
        for component in range(1, count + 1):
            y, x = np.nonzero(components == component)
            x0, x1 = max(0, x.min() - 35), min(changed.shape[1], x.max() + 36)
            y0, y1 = max(0, y.min() - 35), min(changed.shape[0], y.max() + 36)
            crops = [im[y0:y1, x0:x1].copy() for im in images]
            marked = crops[1].copy()
            marked[changed[y0:y1, x0:x1]] = [255, 40, 180]
            h, w = marked.shape[:2]
            panel = Image.new('RGB', (3 * w, h + 24))
            draw = ImageDraw.Draw(panel)
            for index, (title, crop) in enumerate(zip(['strict', 'interpolated', 'delta'], [*crops, marked])):
                panel.paste(Image.fromarray(crop), (index * w, 24))
                draw.text((index * w + 2, 4), title, fill='white')
            path = dest / f'{arm}_component{component}.png'
            panel.save(path)
            files.append(dict(path=str(path), sha256=sha(path), portrait_box=[int(x0), int(y0), int(x1), int(y1)]))
    save(dest / 'result.json', dict(records=records, files=files, input_hashes=bindings,
                                    script_sha256=sha(__file__), helper_sha256=sha(Path(__file__).with_name('review_mhr_admission_branch_difference.py')),
                                    target_used_posthoc_only=True, production_accepted=False))


if __name__ == '__main__':
    main()
