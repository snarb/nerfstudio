"""Preserve raw ray integration inside the union of foreground silhouettes.

Depth-gated RGB is not a valid foreground source at occlusions: a single depth
window can remove the rear person around a bat/hat. MatAnyone is positive-only
here: it may identify additional interior, but cannot delete strict foreground.
Real holes in the union are retained (no blanket hole filling or closing).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


def restore(raw, cleaned, strict_alpha, matte):
    union = (strict_alpha > .95) | (matte > .95)
    distance = ndi.distance_transform_edt(union)
    weight = np.clip((distance - 2) / 3, 0, 1)
    result = np.uint8(np.clip(
        cleaned * (1 - weight[..., None]) + raw * weight[..., None] + .5,
        0, 255))
    assert np.array_equal(result[distance >= 5], raw[distance >= 5])
    assert np.array_equal(result[distance <= 2], cleaned[distance <= 2])
    return result, weight


def run(root):
    root = Path(root)
    op = root / 'operations'
    base = op / 'flying_strict_low_20261112'
    previous = op / 'flying_exterior_only_20261112'
    out = op / 'flying_union_interior_20261112'
    out.mkdir(exist_ok=True)
    (out / 'path.json').write_bytes((base / 'path.json').read_bytes())
    for name, target in [('codec', base / 'codec'), ('gt', previous / 'gt')]:
        if not (out / name).exists():
            (out / name).symlink_to(target, target_is_directory=True)
    records = []
    for i in range(70):
        folder = base / 'neural' / f'{i:03}'
        raw = np.array(Image.open(folder / 'raw.png'))
        old = np.array(Image.open(previous / 'neural' / f'{i:03}' / 'strict.png'))
        alpha = np.load(folder / 'strict.npz')['accumulation'][..., 0]
        matte = np.array(Image.open(op / 'body_matte_cleanup_20261112' /
                                   'sequence' / 'alpha' / f'{i:03}.png')) / 255
        clean, weight = restore(raw, old, alpha, matte)
        dest = out / 'neural' / f'{i:03}'
        dest.mkdir(parents=True, exist_ok=True)
        Image.fromarray(clean).save(dest / 'strict.png')
        Image.fromarray(np.uint8(weight * 255)).save(dest / 'raw_weight.png')
        records.append(dict(index=i, restored_pixels=int((weight > 0).sum()),
                            raw_interior_max_difference=0, exterior_max_difference=0))
        if i in [5, 13, 14, 15, 16, 50, 51, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]:
            panel = Image.new('RGB', (1920, 540))
            for col, (im, label) in enumerate([(old, 'previous'), (clean, 'raw union interior')]):
                q = Image.fromarray(im).resize((960, 540))
                ImageDraw.Draw(q).text((10, 10), f'{i}: {label}', fill='white')
                panel.paste(q, (960 * col, 0))
            panel.save(out / f'compare_{i:03}.jpg')
        print('RESTORED', i, flush=True)
    (out / 'interior_receipt.json').write_text(json.dumps(dict(
        policy='positive-only union; raw RGB in interior; no hole filling; unchanged outer boundary',
        gt='unchanged from exterior-only revision', frames=records), indent=2))
    sys.path.insert(0, str(root / 'tools'))
    from flying_delivery import assemble
    from flying_encode import encode
    assemble(out)
    encode(out)
    old = out / 'Marshawn908_1267_flying_strict_BC_slowmo_24fps.mp4'
    new = out / 'Marshawn_union_interior_preserved.mp4'
    old.replace(new)
    receipt = json.loads((out / 'delivery_receipt.json').read_text())
    receipt['video'] = str(new)
    (out / 'delivery_receipt.json').write_text(json.dumps(receipt, indent=2))
    print('DONE', new, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    run(p.parse_args().root)
