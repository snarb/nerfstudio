"""Export every XYZ slice of an offline grid-viewer specification for review.

Blue = support layer only; white = active inside support; red = active outside
support. A blue cell is permitted, not necessarily occupied or physically free.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from export_occupancy_grid_viewer import load_layers


def unpack(layer, shape):
    packed = np.frombuffer(base64.b64decode(layer['bits']), dtype=np.uint8)
    return np.unpackbits(packed, bitorder='big')[:int(np.prod(shape))].reshape(shape).astype(bool)


def slice_rgb(active, support, fixed_axis, index):
    # Remaining axes are horizontal then vertical in increasing XYZ order.
    a = np.take(active, index, axis=fixed_axis).T[::-1]
    s = np.take(support, index, axis=fixed_axis).T[::-1]
    rgb = np.zeros((*a.shape, 3), dtype=np.uint8)
    rgb[:] = (12, 14, 18)
    rgb[s] = (30, 65, 100)
    rgb[a & s] = (232, 232, 220)
    rgb[a & ~s] = (240, 45, 45)
    return rgb


def export(spec, output, active_name, support_name):
    payload = load_layers(spec)
    layers = {x['name']: x for x in payload['layers']}
    active = unpack(layers[active_name], payload['shape'])
    support = unpack(layers[support_name], payload['shape'])
    output.mkdir(parents=True, exist_ok=True)
    pages = []
    bounds = np.array(payload['bounds'])
    for fixed in range(3):
        horizontal, vertical = [i for i in range(3) if i != fixed]
        tile_w, tile_h = active.shape[horizontal], active.shape[vertical]
        for start in range(0, active.shape[fixed], 64):
            indices = list(range(start, min(start + 64, active.shape[fixed])))
            page = Image.new('RGB', (8*(tile_w+8), 46+8*(tile_h+21)), (22, 22, 22))
            draw = ImageDraw.Draw(page)
            draw.text((8, 5), f'{active_name} / {support_name}; fixed {"XYZ"[fixed]}; +{"XYZ"[horizontal]} right, +{"XYZ"[vertical]} up', fill='white')
            draw.text((8, 22), 'WHITE active   BLUE permitted support only   RED active outside support; not a physical emptiness certificate', fill='white')
            for n, index in enumerate(indices):
                x, y = (n % 8)*(tile_w+8), 46+(n//8)*(tile_h+21)
                coordinate = bounds[0, fixed]+(index+.5)/active.shape[fixed]*(bounds[1, fixed]-bounds[0, fixed])
                draw.text((x, y), f'{index:03d} {coordinate:+.4f}', fill='white')
                page.paste(Image.fromarray(slice_rgb(active, support, fixed, index)), (x, y+16))
            path = output / f'{"XYZ"[fixed]}_{start:03d}_{indices[-1]:03d}.png'
            page.save(path)
            pages.append({'path': str(path), 'fixed_axis': 'XYZ'[fixed], 'slice_indices': indices, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    receipt = dict(kind='EXHAUSTIVE_ORTHOGONAL_SLICE_PAGES', coordinate_frame=payload['coordinate_frame'], bounds=payload['bounds'], shape=payload['shape'], active_layer={k:v for k,v in layers[active_name].items() if k!='bits'}, support_layer={k:v for k,v in layers[support_name].items() if k!='bits'}, active_outside_support=int(np.count_nonzero(active & ~support)), pages=pages, all_slices_exported=sum(payload['shape']), manual_review_complete=False, physical_geometry_certified=False)
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--active', required=True)
    parser.add_argument('--support', required=True)
    args = parser.parse_args()
    result = export(json.loads(args.spec.read_text()), args.output, args.active, args.support)
    print(json.dumps({'pages': len(result['pages']), 'slices': result['all_slices_exported'], 'outside_support': result['active_outside_support']}))
