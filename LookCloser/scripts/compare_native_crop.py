"""Save matched, unscaled HC face/cloth/foot patches from checkpoint renders."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PATCHES = {
    'black_head': ('black', (550, 80, 1360, 800)),
    'grey_head': ('grey', (920, 770, 1464, 1320)),
    'grey_cloth': ('grey', (460, 1080, 1050, 1500)),
    'feet': (None, (40, 1650, 1450, 2360)),
}


def compare(data, predictions, output):
    import numpy as np
    from PIL import Image, ImageDraw

    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = np.array(Image.open(data/'images/HC.png').convert('RGB'))
    if target.shape != (2360, 1464, 3):
        raise ValueError('These fixed frame1660 HC diagnostic coordinates require the sealed native crop')
    with np.load(data/'semantics/HC.npz', allow_pickle=False) as source:
        masks = {actor: source[actor].copy() for actor in ('black', 'grey')}
    images = {'GT': target}
    for name, path in predictions.items():
        if not name or name == 'GT' or '/' in name or '..' in name:
            raise ValueError('Unique safe label required')
        pixels = np.array(Image.open(path).convert('RGB'))
        if pixels.shape != target.shape:
            raise ValueError('Rendered native camera does not match the fixed ground-truth raster')
        images[name] = pixels
    results, previews = [], []
    for patch, (actor, bounds) in PATCHES.items():
        x0, y0, x1, y1 = bounds
        mask = masks[actor] if actor is not None else masks['black'] | masks['grey']
        local = mask[y0:y1, x0:x1]
        strip = Image.new('RGB', ((x1-x0)*len(images), y1-y0+30), (25, 25, 25))
        preview = Image.new('RGB', (400*len(images), 430), (25, 25, 25))
        for column, (name, pixels) in enumerate(images.items()):
            crop = Image.fromarray(pixels[y0:y1, x0:x1])
            crop.save(output/f'{patch}_{name}.png')
            strip.paste(crop, (column*(x1-x0), 30))
            ImageDraw.Draw(strip).text((column*(x1-x0)+8, 8), name, fill='white')
            small = crop.copy()
            small.thumbnail((400, 400), Image.Resampling.LANCZOS)
            preview.paste(small, (column*400, 30))
            ImageDraw.Draw(preview).text((column*400+8, 8), patch+' / '+name, fill='white')
            if name != 'GT':
                delta = (pixels[y0:y1, x0:x1].astype(np.float64)-target[y0:y1, x0:x1])/255.
                mse = float((delta[local]**2).mean())
                results.append(dict(patch=patch, run=name, actor=actor or 'union',
                                    actor_pixels=int(local.sum()), psnr=None if mse == 0 else -10*math.log10(mse),
                                    exact_match=mse == 0, bounds=list(bounds), resized=False))
        strip.save(output/f'{patch}_comparison_1to1.png')
        previews.append(preview)
    contact = Image.new('RGB', (400*len(images), 430*len(previews)), (25, 25, 25))
    for row, preview in enumerate(previews):
        contact.paste(preview, (0, row*430))
    contact.save(output/'contact_preview.jpg', quality=92)
    record = dict(camera='HC', frame=1660, patch_definitions='fixed GT coordinates; actor mask intersection',
                  no_image_resizing_for_saved_patches_or_PSNR=True, preview_only_is_resized=True, records=results)
    (output/'patch_metrics.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--render', action='append', default=[], help='LABEL=/absolute/path/to/native/HC.png')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    predictions = {}
    for value in args.render:
        name, path = value.split('=', 1)
        if name in predictions:
            raise ValueError('Duplicate run label')
        predictions[name] = Path(path)
    compare(args.data, predictions, args.output)


if __name__ == '__main__':
    main()
