"""Compare saved opacity outside fixed EVAL masks; no empty-space certification.

Euclidean image-plane margins are diagnostics, not targets for training or
proof that hair, occlusions, or floor have been labelled correctly.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_alpha(path):
    header = Path(path).read_bytes()[:29]
    if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR' or header[24:26] != bytes([16, 0]):
        raise ValueError('Expected scalar 16-bit PNG: ' + str(path))
    raw = np.asarray(Image.open(path))
    if raw.ndim != 2 or raw.min() < 0 or raw.max() > 65535:
        raise ValueError('Invalid opacity raster')
    return raw.astype(np.float64) / 65535


def main(args):
    if args.output.exists():
        raise ValueError('Preserve previous generation')
    arms = {}
    for value in args.arm:
        label, path = value.split('=', 1)
        if label in arms:
            raise ValueError('Duplicate arm label')
        arms[label] = Path(path)
    ready_path = args.data / 'ready.json'
    ready = json.loads(ready_path.read_text())
    records = {row['camera']: row for row in ready['records']}
    inputs, results = {str(ready_path): sha(ready_path)}, []
    args.output.mkdir(parents=True)
    for camera in ('DC', 'HC', 'KA'):
        mask_path = args.data / 'semantics' / (camera + '.npz')
        assert sha(mask_path) == records[camera]['semantic_sha256']
        with np.load(mask_path, allow_pickle=False) as data:
            if data['black'].dtype != np.bool_ or data['grey'].dtype != np.bool_:
                raise ValueError('Expected original boolean V1 masks')
            union = data['black'] | data['grey']
        if not union.any():
            raise ValueError('Empty actor union')
        distance = distance_transform_edt(~union)
        inputs[str(mask_path)] = sha(mask_path)
        tiles = []
        for label, renders in arms.items():
            path = renders / (camera + '_alpha.png')
            alpha = read_alpha(path)
            if alpha.shape != union.shape:
                raise ValueError('Wrong fixed evaluation resolution')
            inputs[str(path)] = sha(path)
            for margin in (0, 3, 8, 16, 32):
                region = distance > margin
                values = alpha[region]
                results.append(dict(camera=camera, arm=label,
                    euclidean_margin_HD_pixels=margin, exterior_pixels=int(region.sum()),
                    alpha_sum=float(values.sum()), alpha_mean=float(values.mean()),
                    above01_pixels=int((values > .1).sum()), above05_pixels=int((values > .5).sum()),
                    above09_pixels=int((values > .9).sum())))
            # White is alpha inside the nominal union; orange is alpha outside.
            colored = np.repeat(alpha[..., None], 3, axis=2)
            colored[~union] *= np.asarray([1., .35, 0.])
            tile = Image.fromarray(np.uint8(np.clip(colored, 0, 1) * 255 + .5))
            tile.thumbnail((640, 338))
            sheet = Image.new('RGB', (640, 362), (24, 24, 24))
            sheet.paste(tile, (0, 24))
            ImageDraw.Draw(sheet).text((4, 4), label + ': white mask / orange exterior', fill='white')
            tiles.append(sheet)
        contact = Image.new('RGB', (640 * len(tiles), 362))
        for index, tile in enumerate(tiles):
            contact.paste(tile, (640 * index, 0))
        contact.save(args.output / (camera + '_opacity.jpg'), quality=92)
    receipt = dict(kind='FIXED_V1_MASK_EXTERIOR_OPACITY_DIAGNOSTIC',
        inputs_sha256=inputs, helper_sha256=sha(__file__), rows=results,
        model_loaded=False, model_or_grid_modified=False, primary_metrics_recomputed=False,
        limitation='Mask-exterior opacity is not certified nonactor geometry. Fixed margins preserve uncertainty explicitly; no new labels, carving or training objective.')
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), rows=len(results))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--arm', action='append', required=True, help='label=render_directory')
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
