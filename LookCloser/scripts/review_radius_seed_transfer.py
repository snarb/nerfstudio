"""Matched baseline/nearest24/all-radius review, with explicit regression crops."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects
from review_local_mhr_transfer import panel, enclosed_misses
from study_confidence_depth_prior import project_integer
from study_multiview_face_prior import read, save, sha


def counts(reference_rgb, reference_depth, rgb, depth, crop):
    old, new = reference_depth > 0, depth > 0
    holes = enclosed_misses(reference_depth, crop)
    delta = np.zeros(depth.shape); common = old & new
    delta[common] = depth[common] - reference_depth[common]
    return dict(original_enclosed_misses=int(holes.sum()), remaining_enclosed_misses=int((holes & ~new).sum()),
        new_geometry=int((new & ~old).sum()), lost_geometry=int((old & ~new).sum()),
        new_uncolored_geometry=int((new & ~old & (rgb.max(2) == 0)).sum()),
        newly_black_rgb=int(((reference_rgb.max(2) > 0) & (rgb.max(2) == 0)).sum()),
        nearer_over003=int((delta < -.003).sum()), farther_over003=int((delta > .003).sum()),
        max_common_depth_change=float(abs(delta).max()))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root.resolve(); dest = root / 'review'; assert not dest.exists()
    q = read(root / 'request.json'); adapter = read(root / 'admission/rgb_adapter.json')
    assert adapter['request_sha256'] == sha(root / 'request.json')
    config = read(Path(q['source_candidate_root']) / 'config.json')
    fit = np.load(Path(config['spec']['prior']) / 'fit.npz')
    neutral, vertices = fit['neutral'], fit['vertices']
    points = vertices[(neutral[:, 1] > 135) & (neutral[:, 1] < 180) & (abs(neutral[:, 0]) < 16)]
    bindings = {str(path): sha(path) for path in [root / 'request.json', root / 'admission/rgb_adapter.json',
        Path(config['spec']['prior']) / 'fit.npz', Path(__file__)]}
    records = []; dest.mkdir()
    for view in adapter['views']:
        data = {}; camera = None
        for variant in ['baseline', 'nearest24', 'interpolated']:
            folder = root / 'admission/rgb' / view / variant / 'frames' / q['frame']
            receipt = read(folder / 'complete.json'); request = folder.parent.parent / 'request.json'
            assert receipt['request_sha256'] == sha(request)
            for rel, digest in receipt['hashes'].items(): assert sha(folder / rel) == digest, rel
            r = read(folder / 'result.json')
            if camera is None: camera = r['camera']
            else: assert camera == r['camera']
            data[variant] = (np.asarray(Image.open(folder / 'frame.png').convert('RGB')),
                np.rot90(np.load(folder / 'target_depth.npz')['depth']))
            bindings[str(folder / 'complete.json')] = sha(folder / 'complete.json')
            bindings[str(request)] = sha(request)
        xy, z = project_integer(camera, points); xy = xy[z > 0]
        portrait = np.c_[xy[:, 1], camera['w']-1-xy[:, 0]]
        lo = np.floor(portrait.min(0)-30).astype(int); hi = np.ceil(portrait.max(0)+30).astype(int)
        crop = [max(0,int(lo[0])),max(0,int(lo[1])),min(1080,int(hi[0])+1),min(1920,int(hi[1])+1)]
        assert crop[2] > crop[0] and crop[3] > crop[1]
        images = [(name, pair[0]) for name, pair in data.items()]
        panel(images, dest / (view + '_native.png'), crop)
        small = [(name, np.asarray(Image.fromarray(rgb).resize((540, 960), Image.Resampling.LANCZOS))) for name, rgb in images]
        panel(small, dest / (view + '_overview.png'), [0, 0, 540, 960])
        record = dict(view=view, crop=crop, versus_baseline={}, side_effect_panels=[])
        for name, (rgb, depth) in data.items():
            record['versus_baseline'][name] = counts(*data['baseline'], rgb, depth, crop)
        old_rgb, old_depth = data['nearest24']; new_rgb, new_depth = data['interpolated']
        record['versus_nearest24'] = counts(old_rgb, old_depth, new_rgb, new_depth, crop)
        delta = np.zeros(new_depth.shape); common = (old_depth > 0) & (new_depth > 0)
        delta[common] = new_depth[common] - old_depth[common]
        masks = dict(new_black=(old_rgb.max(2) > 0) & (new_rgb.max(2) == 0),
            lost_geometry=(old_depth > 0) & (new_depth <= 0),
            new_uncolored=(old_depth <= 0) & (new_depth > 0) & (new_rgb.max(2) == 0),
            common_depth_change=abs(delta) > .003)
        for kind, mask in masks.items():
            labels, n = label(mask); areas = np.bincount(labels.ravel()); boxes = find_objects(labels)
            for rank, index in enumerate(sorted(range(1,n+1), key=lambda i: int(areas[i]), reverse=True)[:2]):
                yy, xx = boxes[index-1]
                box = [max(0,xx.start-30),max(0,yy.start-30),min(1080,xx.stop+30),min(1920,yy.stop+30)]
                name = f'{view}_{kind}_{rank}.png'; panel(images, dest / name, box)
                record['side_effect_panels'].append(dict(path=name, kind=kind, pixels=int(areas[index]), crop=box))
        records.append(record)
        print(view, record['versus_nearest24'], flush=True)
    save(dest / 'result.json', dict(frame=q['frame'], records=records, input_hashes=bindings,
        images={p.name: sha(p) for p in dest.glob('*.png')}, visual_status='pending',
        enclosed_misses_not_confirmed_anatomical_holes=True, diagnostic_counts_not_quality_metrics=True,
        production_accepted=False))


if __name__ == '__main__': main()
