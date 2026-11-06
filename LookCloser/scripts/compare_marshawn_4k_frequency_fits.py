#!/usr/bin/env python3
"""Compare actual same-source SSIM censor masks, separately from max levels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch


def compare(campaign, first_generation='resolution_4k_AA_frequency', second_generation='resolution_4k_AA_frequency_4000'):
    first = campaign / first_generation / 'result'
    second = campaign / second_generation / 'result'
    semantics = np.load(campaign / 'resolution_4k_train63/semantics/AA.npz')
    a, b = [json.loads((p / 'complete.json').read_text()) for p in (first, second)]
    assert a['source_image_sha256'] == b['source_image_sha256']
    assert {k: v for k, v in a['recipe'].items() if k != 'train_steps_per_level'} == {k: v for k, v in b['recipe'].items() if k != 'train_steps_per_level'}
    masks = [np.asarray(Image.open(p / 'unresolved_patch_mask.png')) > 0 for p in (first, second)]
    for p, r in [(first, a), (second, b)]:
        for rel, key in [('unresolved_patch_mask.png', 'censor_mask_sha256'), ('maps/train_AA.pt', 'map_sha256')]:
            assert hashlib.sha256((p / rel).read_bytes()).hexdigest() == r[key]
    levels = [np.rint(np.log(torch.load(p / 'maps/train_AA.pt', map_location='cpu', weights_only=True).numpy()/16)/np.log(8192/16)*15).astype(int) for p in (first, second)]
    def coverage(x):
        return x.reshape(253, 8, 480, 8).mean((1, 3))
    union = coverage((semantics['black'] > 0) | (semantics['grey'] > 0))
    groups = {'all': np.ones_like(union, bool), 'actor_interior': union >= .9, 'actor_boundary': (union > 0) & (union < .9), 'outside_actors': union == 0, 'black_interior': coverage(semantics['black'] > 0) >= .9, 'grey_interior': coverage(semantics['grey'] > 0) >= .9}
    budgets = [a['recipe']['train_steps_per_level'], b['recipe']['train_steps_per_level']]
    result = {'comparison': f'{budgets[0]} vs {budgets[1]} updates per level, same seed and source', 'results': {}, 'elapsed_seconds': [a['elapsed_seconds'], b['elapsed_seconds']], 'gpu_peak_bytes': [a['gpu_peak_memory_bytes'], b['gpu_peak_memory_bytes']], 'no_automatic_quality_approval': True}
    for name, group in groups.items():
        result['results'][name] = {'patches': int(group.sum()), 'unresolved_fractions': [float(m[group].mean()) for m in masks], 'maxlevel_fractions': [float((l[group] == 15).mean()) for l in levels], 'previously_failed_now_passed': int((group & masks[0] & ~masks[1]).sum()), 'previously_passed_now_failed': int((group & ~masks[0] & masks[1]).sum())}
    (second / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    source = Image.open(campaign / 'resolution_4k_train63/images/train_AA.png').convert('RGB')
    spec = json.loads((campaign / 'resolution_4k_AA_frequency/spec_silver.json').read_text())
    for name, crop in spec['actor_crops'].items():
        pieces = [source.crop(crop)]
        for mask in masks:
            rgb = np.zeros((*mask.shape, 3), np.uint8)
            rgb[mask] = (255, 0, 90)
            rgb[~mask] = (0, 210, 100)
            overlay = Image.fromarray(rgb).resize(source.size, Image.Resampling.NEAREST)
            pieces.append(Image.blend(source, overlay, .4).crop(crop))
        sheet = Image.new('RGB', (pieces[0].width * 3, pieces[0].height + 35))
        draw = ImageDraw.Draw(sheet)
        for i, (piece, label) in enumerate(zip(pieces, ['actual source'] + [f'{budget}: pink failed, green passed' for budget in budgets])):
            sheet.paste(piece, (i * piece.width, 35)); draw.text((i * piece.width + 4, 10), label, fill='white')
        sheet.save(second / ('censor_comparison_' + name + '.png'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--first-generation', default='resolution_4k_AA_frequency')
    parser.add_argument('--second-generation', default='resolution_4k_AA_frequency_4000')
    a = parser.parse_args()
    compare(a.campaign, a.first_generation, a.second_generation)
