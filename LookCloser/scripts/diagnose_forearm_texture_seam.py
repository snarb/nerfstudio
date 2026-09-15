"""Separate source switches, old/new mesh boundaries and depth jumps in RGB."""
from pathlib import Path
import colorsys
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json
from build_multiview_forearm_admission import ROOT as INPUT, FRAME
from review_jaw_repair_transfer import verified_image, panel
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from hard_surface_texture import face_adjacency

ROOT = Path('/mnt/data/dec5_forearm_seam_attribution')


def run():
    ROOT.mkdir(exist_ok=True)
    if (ROOT / 'result.json').exists():
        raise ValueError('Completed diagnosis already exists')
    q = read(INPUT / 'request.json');r = read(INPUT / 'result.json')
    assert r['mesh_sha256'] == sha(INPUT / 'mesh.ply')
    original = o3d.io.read_triangle_mesh(q['source_mesh']);mesh = o3d.io.read_triangle_mesh(str(INPUT / 'mesh.ply'))
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles);old_count = len(original.triangles)
    edges = face_adjacency(t);cross = (edges[:, 0] < old_count) != (edges[:, 1] < old_count)
    scene = scene_for(v, t);records = []
    palette = np.zeros((256, 3), np.uint8)
    for i in range(62):
        palette[i] = np.array(colorsys.hsv_to_rgb((i*.61803398875) % 1, .7, .95))*255
    for view, box in [('H004_A005_1210M6', (40, 1650, 300, 1920)), ('moving', (90, 1650, 430, 1920))]:
        base = INPUT / 'rgb' / view;im, receipt = verified_image(base, FRAME)
        folder = base / 'frames' / FRAME
        d, ids, _ = camera_depth(scene, receipt['camera'])
        # Native train-view targets inherit the renderer's source-mask wrapper;
        # virtual moving targets do not. Replay it, never compare masked saved
        # depth to an unmasked fresh raycast and mislabel the difference geometry.
        entry = read(base / 'request.json')['inventory'][0]
        spec = entry['source_masks'];mask_root = Path(spec['root'])
        for filename, key in [('masks.npz', 'masks_sha256'), ('cameras.json', 'cameras_sha256')]:
            assert sha(mask_root / filename) == spec[key]
        names = read(mask_root / 'cameras.json');masked_pixels = 0
        if receipt['camera']['physical_camera'] in names:
            mask = np.load(mask_root / 'masks.npz')['masks'][names.index(receipt['camera']['physical_camera'])]
            masked_pixels = int((np.isfinite(d) & ~mask.astype(bool)).sum())
            d = d.copy();d[~mask.astype(bool)] = np.inf
        d, ids = np.rot90(d), np.rot90(ids)
        saved = np.rot90(np.load(folder / 'target_depth.npz')['depth'])
        np.testing.assert_array_equal(np.isfinite(d), saved > 0)
        np.testing.assert_allclose(d[np.isfinite(d)], saved[saved > 0], atol=2e-6)
        sources = np.rot90(np.array(Image.open(folder / 'source_ids.png')))
        labels = np.load(folder / 'face_source_labels.npy')
        valid = np.isfinite(d);preferred = np.full(d.shape, -1, int);preferred[valid] = labels[ids[valid]]
        new = valid & (ids >= old_count)
        x0, y0, x1, y1 = box
        rgb = im[y0:y1, x0:x1];src = sources[y0:y1, x0:x1];zz = d[y0:y1, x0:x1]
        added = new[y0:y1, x0:x1];known = valid[y0:y1, x0:x1];pref = preferred[y0:y1, x0:x1]
        warm = rgb[..., 0].astype(float)-rgb[..., 2] > 8
        switches = []
        for axis in [0, 1]:
            a = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
            b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
            domain = known[a] & known[b] & warm[a] & warm[b] & (src[a] != 255) & (src[b] != 255)
            color_jump = np.abs(rgb[a].astype(float)-rgb[b]).mean(2)
            source_change = src[a] != src[b];mesh_change = added[a] != added[b]
            with np.errstate(invalid='ignore'):
                depth_jump = np.abs(zz[a]-zz[b])
            strong = domain & (color_jump >= 10)
            switches.append(dict(axis=axis, available_edges=int(domain.sum()), strong_rgb_edges=int(strong.sum()),
                strong_source_switch=int((strong & source_change).sum()),
                strong_old_new_boundary=int((strong & mesh_change).sum()),
                strong_both=int((strong & source_change & mesh_change).sum()),
                strong_same_source=int((strong & ~source_change).sum()),
                strong_depth_step_over_0005=int((strong & (depth_jump > .0005)).sum())))
        geometry = np.zeros_like(rgb);geometry[known] = [140, 140, 140];geometry[added] = [215, 130, 65]
        fallback = np.zeros_like(rgb);fallback[known & (src != 255)] = [70, 70, 70]
        fallback[known & (src != 255) & (src != pref)] = [220, 50, 190]
        path = ROOT / (view+'_maps.png')
        panel(path, [rgb, palette[src], geometry, fallback], ['RGB', 'actual source id', 'gray old / orange added', 'magenta pixel fallback'], (0, 0, x1-x0, y1-y0))
        count = {int(i): int((src == i).sum()) for i in np.unique(src) if i != 255}
        legend = Image.new('RGB', (500, 24*(len(count)+1)), 'black');draw = ImageDraw.Draw(legend)
        draw.text((4, 3), view, fill='white')
        for j, (i, n) in enumerate(sorted(count.items(), key=lambda x:-x[1])):
            y = (j+1)*24;draw.rectangle((4, y+2, 20, y+18), fill=tuple(palette[i]))
            draw.text((28, y+2), f'{i}: {receipt["source_cameras"][i]} ({n} pixels)', fill='white')
        legend.save(ROOT / (view+'_legend.png'))
        np.savez_compressed(ROOT / (view+'_arrays.npz'), rgb=rgb, source=src, preferred=pref, depth=zz, added=added, valid=known)
        records.append(dict(view=view, box=box, edge_diagnostics=switches, source_counts=count,
                            panel=str(path), panel_sha256=sha(path), source_names=receipt['source_cameras'],
                            source_masked_target_hits=masked_pixels))
    atomic_json(ROOT / 'result.json', dict(input_request_sha256=sha(INPUT / 'request.json'),
        mesh_sha256=sha(INPUT / 'mesh.ply'), script_sha256=sha(__file__), records=records,
        old_new_topological_adjacency_edges=int(cross.sum()), diagnostic_not_quality_metrics=True,
        geometry_changed=False, texture_changed=False, visual_status='pending'))
    print('old/new graph links', int(cross.sum()), [(r['view'], r['edge_diagnostics']) for r in records], flush=True)


if __name__ == '__main__':
    run()
