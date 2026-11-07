"""CPU contact pages for every native-cell slice of a separate candidate.

Colours show changed traversal support, never a physical free/occupied label.
Each cascade128 contains 384 distinct slices across XYZ, six pages.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from export_marshawn_semantic_cell_candidates_cpu import read, sha, write


def render(candidate, output):
    candidate = Path(candidate); output = Path(output)
    meta = read(candidate/'metadata.json')
    if meta.get('schema') != 'LC_SEMANTIC_CELL_CANDIDATE_V1' or sha(candidate/'candidate.npz') != meta['candidate_sha256']:
        raise ValueError('Bound candidate archive required')
    with np.load(candidate/'candidate.npz', allow_pickle=False) as z:
        source, r0, r1 = [z[k].copy() for k in ('original_active', 'active_r0', 'active_r1')]
        bounds = z['cascade_aabbs'].reshape(-1, 2, 3)
    if source.shape != r0.shape or r0.shape != r1.shape or source.ndim != 4:
        raise ValueError('Matching cascade XYZ masks required')
    output.mkdir(parents=True, exist_ok=False)
    pages = []; slices = []; tile = 160; top = 76; columns = 8; rows = 8
    for cascade in range(len(source)):
        for axis in range(3):
            count = source.shape[axis+1]
            for first in range(0, count, rows*columns):
                page = Image.new('RGB', (columns*tile, rows*(tile+22)+top), (16, 16, 16)); draw = ImageDraw.Draw(page)
                draw.text((8, 6), f'{meta["selector"]}; cascade{cascade}; axis {"XYZ"[axis]}; source-prior {meta["initial_probability"]}', fill='white')
                draw.text((8, 24), 'cyan: candidate R0 | gold: new R1 | magenta: source-active removed by R0', fill='white')
                draw.text((8, 42), 'Traversal support only. Blank is NOT certified physical free space.', fill='white')
                for index in range(first, min(first+rows*columns, count)):
                    original = np.take(source[cascade], index, axis=axis)
                    a = np.take(r0[cascade], index, axis=axis); b = np.take(r1[cascade], index, axis=axis)
                    rgb = np.zeros(original.shape+(3,), np.uint8)
                    rgb[original & ~a] = [190, 65, 160]
                    rgb[a] = [40, 210, 220]
                    rgb[b & ~a] = [255, 190, 45]
                    # Remaining axes keep increasing order; transpose maps first
                    # remaining axis horizontally, flip makes second axis up.
                    image = Image.fromarray(np.flip(rgb.transpose(1, 0, 2), axis=0)).resize((tile, tile), Image.Resampling.NEAREST)
                    cell = index-first; x = cell % columns*tile; y = top+cell//columns*(tile+22)
                    page.paste(image, (x, y+22))
                    lo = bounds[cascade, 0, axis]+index/count*(bounds[cascade, 1, axis]-bounds[cascade, 0, axis])
                    hi = bounds[cascade, 0, axis]+(index+1)/count*(bounds[cascade, 1, axis]-bounds[cascade, 0, axis])
                    draw.text((x+2, y+2), f'{"XYZ"[axis]}{index:03d} {lo:.5f}', fill='white')
                    slices.append(dict(cascade=cascade, axis='XYZ'[axis], index=index, lower=float(lo), upper=float(hi),
                        source_active=int(original.sum()), candidate_r0=int(a.sum()), candidate_r1=int(b.sum())))
                name=f'cascade{cascade}_{"XYZ"[axis]}_{first:03d}.png'; page.save(output/name)
                pages.append(dict(file=name, sha256=sha(output/name), first=first, count=min(rows*columns, count-first)))
    receipt = dict(CPU_only=True, candidate_metadata_sha256=sha(candidate/'metadata.json'), candidate_archive_sha256=meta['candidate_sha256'],
        total_slices=len(slices), pages=pages, slices=slices, physical_free_space_claim=False, native_model_loaded=False)
    write(output/'receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--candidate', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = render(a.candidate, a.output); print(json.dumps(dict(total_slices=result['total_slices'], pages=len(result['pages']))))
