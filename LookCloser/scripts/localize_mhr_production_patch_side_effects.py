"""Native matched side-effect crops: uncolored geometry and newly black RGB.

These are diagnostic pixel inventories, not quality metrics or masks used by
reconstruction. Never edits predictions or changes the acceptance thresholds.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from PIL import Image, ImageDraw
from run_mhr_production_patch_control import ROOT, OUT, FRAME, CANDIDATES, ARM, configure
from study_multiview_face_prior import read, save, sha


def side_effect_masks(base_rgb, base_depth, rgb, depth):
    old, new = base_depth > 0, depth > 0
    black_old, black_new = base_rgb.max(2) == 0, rgb.max(2) == 0
    return dict(new_geometry_without_rgb=(~old & new & black_new),
                newly_black_rgb=(~black_old & black_new),
                lost_geometry=(old & ~new))


def residual_mask_attribution():
    import open3d as o3d
    from joint_temporal_texture import project
    from study_mhr_local_head_prior import RGB
    from study_multiview_face_prior import CROP
    import admit_mhr_local_patch_depth as admission
    configure()
    _, rows, _, masks, names, proof = admission.inputs()
    residual = np.load(ROOT/'residual_hole/evidence.npz')
    raw = residual['raw_proposal_id']
    raw = raw[(raw >= 0) & (residual['nearest_safe_distance'] < 1e-6)]
    proposals = np.load(CANDIDATES/ARM/'proposal_evidence.npz')['proposals'][raw]
    v = np.asarray(o3d.io.read_triangle_mesh(str(CANDIDATES/ARM/'local_raw.ply')).vertices)
    points = np.concatenate([v[proposals], v[proposals].mean(1)[:, None]], axis=1).reshape(-1, 3)
    source = read(RGB/FRAME/'input.json')
    dest = ROOT/'residual_hole/mask_attribution'
    dest.mkdir(exist_ok=False)
    records, files, coords = [], [], {}
    for row in rows:
        name = row['physical_camera']
        uv, z = project(points, [row]); uv, z = uv[0], z[0]
        available = (z > 0) & (uv[:, 0] > 2) & (uv[:, 0] < 1917) & (uv[:, 1] > 2) & (uv[:, 1] < 1077)
        xy = np.rint(uv).astype(int)
        inside = np.zeros(len(uv), bool)
        mask = masks[names.index(name)]
        inside[available] = mask[xy[available, 1], xy[available, 0]]
        outside = available & ~inside
        record = dict(camera=name, outside_samples=int(outside.sum()),
                      rejected_ray_facets=int(outside.reshape(-1, 4).any(1).sum()))
        records.append(record)
        coords[name] = uv
        if not outside.any():
            continue
        item = next(r for r in source['inputs'] if r['camera']['physical_camera'] == name)
        assert sha(item['path']) == item['sha256']
        portrait = np.column_stack((uv[:, 1], 1919-uv[:, 0]))
        x0, y0 = np.floor(portrait[available].min(0)-45).astype(int)
        x1, y1 = np.ceil(portrait[available].max(0)+46).astype(int)
        x0, x1 = max(0, x0), min(1080, x1)
        y0, y1 = max(CROP[1], y0), min(CROP[3], y1)
        assert x1 > x0 and y1 > y0
        crop = (int(x0), int(y0), int(x1), int(y1))
        image = Image.open(item['path']).convert('RGB').crop((x0, y0-CROP[1], x1, y1-CROP[1]))
        marked = image.copy(); draw = ImageDraw.Draw(marked)
        for (x, y), ok, visible in zip(portrait, inside, available):
            if visible:
                x, y = x-x0, y-y0
                draw.ellipse((x-1, y-1, x+1, y+1), fill='lime' if ok else 'red')
        binary = Image.fromarray((np.rot90(mask)[y0:y1, x0:x1] > 0).astype(np.uint8)*255).convert('RGB')
        w, h = image.size
        panel = Image.new('RGB', (3*w, h+24)); draw = ImageDraw.Draw(panel)
        for col, (title, im) in enumerate([('train RGB', image), ('candidate samples', marked), ('person mask', binary)]):
            panel.paste(im, (col*w, 24)); draw.text((col*w+2, 4), title, fill='white')
        path = dest/(name+'.png'); panel.save(path)
        files.append(dict(path=str(path), sha256=sha(path)))
        record.update(native_portrait_crop=list(crop), source_path=item['path'], source_sha256=item['sha256'])
    np.savez_compressed(dest/'projections.npz', **coords)
    save(dest/'result.json', dict(records=records, files=files, input_binding=proof,
        source_input_sha256=sha(RGB/FRAME/'input.json'), script_sha256=sha(__file__),
        residual_evidence_sha256=sha(ROOT/'residual_hole/evidence.npz'),
        raw_candidate_sha256=sha(CANDIDATES/ARM/'local_raw.ply'),
        projections_sha256=sha(dest/'projections.npz'),
        samples='three candidate vertices plus centroid; one facet per missing ray, repetitions retained',
        posthoc_only=True, masks_unchanged=True))
    print('residual veto cameras', [r for r in records if r['outside_samples']], flush=True)


def main():
    import localize_mhr_silhouette_patch_occlusion as occlusion
    occlusion.OUT = OUT
    occlusion.main()
    dest = OUT/'black_pixel_review'
    dest.mkdir(exist_ok=False)
    bindings, records, files = {}, [], []
    for view in ['old_moving', 'F004_E', 'M004_B', 'C004_E']:
        frames = {}
        for variant in ['baseline', 'strict', 'interpolated']:
            folder = OUT/'rgb'/view/variant/'frames'/FRAME
            receipt = read(folder/'complete.json')
            for name, digest in receipt['hashes'].items():
                assert sha(folder/name) == digest
            bindings[str(folder/'complete.json')] = sha(folder/'complete.json')
            frames[variant] = (np.asarray(Image.open(folder/'frame.png')),
                               np.rot90(np.load(folder/'target_depth.npz')['depth']))
        base, bd = frames['baseline']
        for variant in ['strict', 'interpolated']:
            im, depth = frames[variant]
            masks = side_effect_masks(base, bd, im, depth)
            for kind, mask in masks.items():
                components, n = label(mask)
                regions = []
                for i in range(1, n+1):
                    yy, xx = np.where(components == i)
                    crop = (max(0, int(xx.min())-55), max(0, int(yy.min())-55),
                            min(1080, int(xx.max())+56), min(1920, int(yy.max())+56))
                    marked = im.copy()
                    marked[components == i] = [255, 0, 255]
                    w, h = crop[2]-crop[0], crop[3]-crop[1]
                    panel = Image.new('RGB', (3*w, h+24))
                    draw = ImageDraw.Draw(panel)
                    for col, (title, array) in enumerate([('baseline', base), (variant, im), (kind, marked)]):
                        panel.paste(Image.fromarray(array).crop(crop), (col*w, 24))
                        draw.text((col*w+2, 4), title, fill='white')
                    path = dest/f'{view}_{variant}_{kind}_{i}.png'
                    panel.save(path)
                    files.append(dict(path=str(path), sha256=sha(path)))
                    regions.append(dict(pixels=len(xx), crop=list(crop), panel=str(path)))
                records.append(dict(view=view, variant=variant, kind=kind, pixels=int(mask.sum()), regions=regions))
    save(dest/'result.json', dict(records=records, files=files, input_hashes=bindings,
        script_sha256=sha(__file__), helper_sha256=sha(occlusion.__file__),
        posthoc_only=True, artifact_free_approval=False, production_modified=False))
    print([(r['view'], r['variant'], r['kind'], r['pixels']) for r in records], flush=True)
    residual_mask_attribution()


if __name__ == '__main__':
    main()
