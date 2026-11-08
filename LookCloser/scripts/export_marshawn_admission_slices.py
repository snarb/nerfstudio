#!/usr/bin/env python3
"""Export every orthogonal slice of a source admission grid for manual review.

Gray: base admission; blue: additional pad1 allowance; black: not admitted.
Admission is permission to sample, not a physical occupancy certificate.
The input directory needs manifest.json, semantic_grid.npz and grid_receipt.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(admission, output):
    manifest = json.loads((admission / 'manifest.json').read_text())
    for name in ['semantic_grid.npz', 'grid_receipt.json']:
        if sha(admission / name) != manifest['files_sha256'][name]:
            raise ValueError('Admission payload differs from its manifest: ' + name)
    with np.load(admission / 'semantic_grid.npz', allow_pickle=False) as data:
        base = data['semantic_gate0'].copy()
        padded = data['semantic_gate_pad1'].copy()
        aabb = data['aabb'].tolist()
    if (base.ndim != 3 or base.dtype != np.bool_ or padded.dtype != np.bool_
            or base.shape != padded.shape or np.any(base & ~padded)):
        raise ValueError('Expected compatible boolean base/pad1 admission arrays')
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for axis in range(3):
        plane = np.take(base, 0, axis=axis)
        height, width = plane.shape
        tile_width, tile_height = width + 4, height + 20
        count = base.shape[axis]
        page = Image.new('RGB', (16 * tile_width, ((count + 15) // 16) * tile_height), (14, 14, 14))
        draw = ImageDraw.Draw(page)
        for index in range(count):
            core = np.take(base, index, axis=axis)
            halo = np.take(padded, index, axis=axis)
            rgb = np.zeros((*core.shape, 3), np.uint8)
            rgb[core] = [180, 180, 180]
            rgb[halo & ~core] = [30, 160, 210]
            x, y = (index % 16) * tile_width, (index // 16) * tile_height
            page.paste(Image.fromarray(rgb), (x, y + 16))
            draw.text((x + 2, y + 1), str(index), fill='white')
        path = output / ('ALL%d_slices_axis%d.png' % (count, axis))
        page.save(path)
        files[path.name] = sha(path)
    result = dict(manifest_sha256=sha(admission / 'manifest.json'),
                  source_grid_sha256=sha(admission / 'semantic_grid.npz'),
                  counts=dict(base=int(base.sum()), pad1=int(padded.sum()), total=int(base.size)),
                  aabb=aabb, shape=list(base.shape), files_sha256=files,
                  legend='gray=base allowance; blue=additional pad1 allowance; black=not admitted',
                  physical_occupancy_certificate=False, source_arrays_modified=False)
    (output / 'slice_export.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.admission, args.output)))
