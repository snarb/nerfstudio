"""Evaluation-only native/source-projection evidence at fixed reference tiles.

Tile locations never feed source selection. This writes diagnostic crops only,
not a texture candidate, geometry, source assignment, or confidence cache.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from luster_hybrid_common import (
    assert_source, geometry, image, native_rows, project, read, sha,
    source_path, unproject, write,
)
from supported_luster_hybrid import geometric_support


def run(root, reference, candidate, cameras, count, output):
    assert count > 0
    tiles_path = root/'evaluation'/f'{reference}_spatial/regression_tiles.json'
    tiles = read(tiles_path)['rows']
    old = root/'candidates'/candidate
    out = root/'evaluation'/output
    out.mkdir(parents=True, exist_ok=True)
    rows = native_rows()
    records = []
    inputs = {str(tiles_path): sha(tiles_path)}
    for target in cameras:
        receipt_path = old/f'cam{target:03d}_receipt.json'
        receipt = read(receipt_path)
        config = receipt['config']
        warp_root = Path(receipt['warp_root'])
        labels_path = old/f'cam{target:03d}_sources.npy'
        labels = np.load(labels_path, mmap_mode='r')
        g = geometry(root, target)
        baseline_path = root/'baseline'/f'cam{target:03d}.png'
        candidate_path = old/f'cam{target:03d}.png'
        baseline = Image.open(baseline_path).convert('RGB')
        hybrid = Image.open(candidate_path).convert('RGB')
        for p in [receipt_path, labels_path, baseline_path, candidate_path,
                  root/'geometry'/f'cam{target:03d}'/'receipt.json']:
            inputs[str(p)] = sha(p)
        target_tiles = [t for t in tiles if t['camera'] == target][:count]
        for index, tile in enumerate(target_tiles):
            box = tile['box']; x0, y0, x1, y1 = box
            assert x1-x0 == y1-y0 == 128
            sl = np.s_[y0:y1, x0:x1]; lab = labels[sl]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            xy = np.stack([xx, yy], -1).astype(np.float32)
            world = unproject(np.asarray(g['depth_z_q50'][sl]), rows[target], xy)
            ids, counts = np.unique(lab[lab >= 0], return_counts=True)
            for cid, pixels in sorted(zip(ids, counts), key=lambda v: (-v[1], v[0]))[:3]:
                cid = int(cid); assert_source(rows, target, cid)
                uv, _ = project(world, rows[cid]); selected = lab == cid
                assert np.isfinite(uv[selected]).all()
                low = np.floor(uv[selected].min(0)-8).astype(int)
                high = np.ceil(uv[selected].max(0)+9).astype(int)
                photo_path = source_path(cid)
                photo = Image.open(photo_path).convert('RGB')
                native_box = [max(0, int(low[0])), max(0, int(low[1])),
                              min(photo.width, int(high[0])), min(photo.height, int(high[1]))]
                d = warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}'
                wr = read(d/'receipt.json'); wx, wy, _, _ = wr['box']
                assert sha(photo_path) == wr['source_sha256']
                crop = np.s_[y0-wy:y1-wy, x0-wx:x1-wx]
                a = {k: np.load(d/f'{k}.npy', mmap_mode='r')[crop] for k in wr['arrays']}
                support = geometric_support(a, config)
                assert support[selected].all()
                stem = f'cam{target:03d}_tile{index}_src{cid:03d}'
                native = photo.crop(native_box)
                native.save(out/f'{stem}_native.png')
                image(out/f'{stem}_projected.png', a['rgb'])
                image(out/f'{stem}_eligible.png', np.repeat(support[..., None], 3, -1).astype(np.float32))
                image(out/f'{stem}_selected.png', np.repeat(selected[..., None], 3, -1).astype(np.float32))
                panels = []
                scale = min(2., 320/native.width, 320/native.height)
                panels.append((f'Native crop; preview {scale:.2f}x', native.resize(
                    (round(native.width*scale), round(native.height*scale)), Image.Resampling.NEAREST)))
                for title, mask in [('Geometrically eligible photo, 2x', support), ('Selected photo pixels, 2x', selected)]:
                    value = np.where(mask[..., None], np.clip(a['rgb'], 0, 1), .5)
                    panels.append((title, Image.fromarray(np.rint(value*255).astype(np.uint8)).resize((256, 256), Image.Resampling.NEAREST)))
                panels += [('Frozen LookCloser, 2x', baseline.crop(box).resize((256, 256), Image.Resampling.NEAREST)),
                           ('Candidate, 2x', hybrid.crop(box).resize((256, 256), Image.Resampling.NEAREST))]
                canvas = Image.new('RGB', (320*len(panels), 352), (25, 25, 25)); draw = ImageDraw.Draw(canvas)
                for j, (title, panel) in enumerate(panels):
                    draw.text((j*320+3, 3), title, fill='white'); canvas.paste(panel, (j*320, 32))
                canvas.save(out/f'{stem}_preview.png')
                for p in [photo_path, d/'receipt.json']:
                    inputs[str(p)] = sha(p)
                records.append(dict(camera=target, tile=index, target_box=box, source=cid,
                                    selected_pixels=int(pixels), native_box=native_box,
                                    files={p.name: sha(p) for p in out.glob(f'{stem}_*.png')}))
    write(out/'receipt.json', dict(usage='evaluation_only_no_prediction_feedback',
        reference=reference, candidate=candidate, rows=records, inputs=inputs,
        code_sha256=sha(Path(__file__)), lookup_shift=False,
        note='Only top three existing source-label counts per fixed evaluation tile are shown.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference', default='single_photometric_depth25_seam')
    p.add_argument('--candidate', default='single_photometric_depth25_seam')
    p.add_argument('--cameras', nargs='+', type=int, default=[7]); p.add_argument('--count', type=int, default=2)
    p.add_argument('--output', default='selected_reference_source_patches')
    a = p.parse_args(); run(a.root, a.reference, a.candidate, a.cameras, a.count, a.output)
