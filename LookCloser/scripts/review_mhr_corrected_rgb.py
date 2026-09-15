"""Verified native RGB/depth deltas and crops for corrected 001193 candidates."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label, find_objects
from study_multiview_face_prior import read, save, sha


def panel(path, images, crop):
    w, h = crop[2]-crop[0], crop[3]-crop[1]
    out = Image.new('RGB', (2*w, h+24)); draw = ImageDraw.Draw(out)
    for i, (name, image) in enumerate(images.items()):
        out.paste(Image.fromarray(image).crop(crop), (i*w, 24))
        draw.text((i*w+2, 4), name, fill='white')
    out.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-root', type=Path, required=True); args = p.parse_args()
    root = args.candidate_root.resolve() / 'admission'
    adapter = read(root / 'rgb_adapter.json'); dest = root / 'rgb_review'
    assert not dest.exists()
    residual_path = Path('/mnt/data/dec5_mhr_production_patch_001193/residual_hole/evidence.npz')
    xy = np.load(residual_path)['portrait_xy']
    spot_path = Path('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')
    spot = next(r['bbox_inclusive'] for r in read(spot_path)['selected_components'] if r['frame_id'] == '001193')
    bindings = {str(p): sha(p) for p in [residual_path, spot_path, root / 'rgb_adapter.json', Path(__file__)]}
    dest.mkdir(); records = []; arrays = {}
    for view in adapter['views']:
        images, depths, sources = {}, {}, {}
        for variant in ['baseline', 'interpolated']:
            folder = root / 'rgb' / view / variant / 'frames/001193'
            receipt = read(folder / 'complete.json')
            request = folder.parent.parent / 'request.json'
            assert receipt['request_sha256'] == sha(request)
            for name, h in receipt['hashes'].items():
                assert sha(folder / name) == h, name
            for path in [folder / 'complete.json', request]: bindings[str(path)] = sha(path)
            images[variant] = np.asarray(Image.open(folder / 'frame.png').convert('RGB'))
            depths[variant] = np.rot90(np.load(folder / 'target_depth.npz')['depth'])
            sources[variant] = np.rot90(np.asarray(Image.open(folder / 'source_ids.png')))
        base, new = images.values(); bd, nd = depths.values(); bs, ns = sources.values()
        assert base.shape == new.shape == (1920, 1080, 3)
        overview = {name: np.asarray(Image.fromarray(im).resize((540, 960), Image.Resampling.LANCZOS))
                    for name, im in images.items()}
        panel(dest / (view + '_overview.png'), overview, (0, 0, 540, 960))
        bh, nh = bd > 0, nd > 0; common = bh & nh
        nearer = np.zeros(bh.shape, bool); nearer[common] = nd[common] < bd[common] - 1e-6
        stable = np.zeros(bh.shape, bool); stable[common] = abs(nd[common] - bd[common]) <= 1e-6
        new_black = (base.max(2) > 0) & (new.max(2) == 0)
        new_uncolored = (~bh & nh) & (new.max(2) == 0)
        lost = bh & ~nh
        record = dict(view=view, new_geometry=int((nh & ~bh).sum()), lost_geometry=int(lost.sum()),
            new_geometry_without_rgb=int(new_uncolored.sum()), newly_black_rgb=int(new_black.sum()),
            nearer_common_geometry=int(nearer.sum()), stable_geometry_source_changes=int((stable & (bs != ns)).sum()))
        if view == 'F004_E':
            x, y = xy.T
            record['fixed_residual'] = dict(rays=len(x), original_hits=int(bh[y,x].sum()),
                corrected_hits=int(nh[y,x].sum()), corrected_colored_hits=int((nh[y,x] & (new[y,x].max(1)>0)).sum()))
            crop = (630, 1080, 775, 1205)
        elif view == 'old_moving':
            x0,y0,x1,y1 = spot; crop = (max(0,x0-90),max(0,y0-120),min(1080,x1+91),min(1920,y1+131))
        else:
            crop = (400, 750, 850, 1350)
        panel(dest / (view + '_native.png'), images, crop)
        components, count = label(new_black); slices = find_objects(components)
        areas = np.bincount(components.ravel()); areas[0] = 0
        record['new_black_components'] = []
        for rank, k in enumerate(np.argsort(areas[1:])[::-1][:3] + 1):
            sl = slices[k-1]; y0,y1 = sl[0].start,sl[0].stop; x0,x1 = sl[1].start,sl[1].stop
            crop = (max(0,x0-30),max(0,y0-30),min(1080,x1+30),min(1920,y1+30))
            path = dest / f'{view}_new_black_{rank}.png'; panel(path, images, crop)
            record['new_black_components'].append(dict(pixels=int(areas[k]), box=[x0,y0,x1,y1], path=path.name))
        arrays[view+'_new_black'] = new_black; arrays[view+'_lost_geometry'] = lost
        records.append(record)
    np.savez_compressed(dest / 'evidence.npz', **arrays)
    save(dest / 'result.json', dict(records=records, input_hashes=bindings,
        outputs={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
        matched_diagnostics_not_heldout_metrics=True, visual_status='pending', production_accepted=False))
    print(records, flush=True)


if __name__ == '__main__': main()
