"""Admit broad appearance from the same selected native TRAIN photograph.

This is a controlled interpolation between the existing detail correction and
literal full texture. Labels, q50 lookup and cubic visibility are unchanged.
Other sources supply only seam confidence, never output color.
"""
from pathlib import Path
import numpy as np
from luster_hybrid_common import assert_source, geometry, image, log, native_rows, read, sha, write
from transfer_luster_hybrid import gaussian
from supported_luster_hybrid import geometric_support
from band_witness_luster_hybrid import assemble


def correction(base, photo, support, sigma, broad_gain):
    assert 0 <= broad_gain <= 1 and sigma > 0
    if broad_gain == 1:
        return photo - base
    sf = support.astype(np.float32)
    mass = gaussian(sf, sigma)
    def low(rgb):
        return gaussian(rgb * sf[..., None], sigma) / np.maximum(mass[..., None], 1e-8)
    pb, bb = low(photo), low(base)
    # Preserve the reference algebra exactly at the matched zero-gain control.
    return (photo - pb) - (base - bb) + broad_gain * (pb - bb)


def combine(root, parent, name, broad_gain):
    old = root / 'candidates' / parent
    out = root / 'candidates' / name
    out.mkdir(parents=True, exist_ok=True)
    config = {**read(old / 'config.json'), 'mode': 'single_source_current_full_texture',
              'parent': parent, 'broad_gain': broad_gain}
    assert config['filter_support'] == 'geometry' and config['adaptive_seam']
    assert 0 <= broad_gain <= 1
    if (out / 'config.json').exists():
        assert read(out / 'config.json') == config
    else:
        write(out / 'config.json', config)
    rows = native_rows()
    for target in [7, 12, 95, 150]:
        if (out / f'cam{target:03d}_receipt.json').exists():
            continue
        receipt = read(old / f'cam{target:03d}_receipt.json')
        warp_root = Path(receipt['warp_root'])
        base = np.asarray(geometry(root, target)['rgb'])
        labels = np.load(old / f'cam{target:03d}_sources.npy', mmap_mode='r')
        deltas, supports = {}, {}
        modules = ['band_witness_luster_hybrid.py', 'supported_luster_hybrid.py',
                   'transfer_luster_hybrid.py', 'luster_hybrid_common.py']
        inputs = {str(Path(__file__).with_name(m)): sha(Path(__file__).with_name(m)) for m in modules}
        box = None
        for cid in receipt['sources']:
            assert_source(rows, target, cid)
            if not np.any(labels == cid):
                continue
            d = warp_root / 'warps' / f'cam{target:03d}' / f'src{cid:03d}'
            r = read(d / 'receipt.json')
            if box is None:
                box = r['box']
            assert box == r['box']
            x0, y0, x1, y1 = box
            sl = np.s_[y0:y1, x0:x1]
            a = {k: np.load(d / f'{k}.npy', mmap_mode='r') for k in r['arrays']}
            support = geometric_support(a, config)
            assert np.all(support[labels[sl] == cid])
            deltas[cid] = correction(base[sl], np.asarray(a['rgb']), support, config['sigma'], broad_gain)
            supports[cid] = support
            inputs[str(d / 'receipt.json')] = sha(d / 'receipt.json')
        assert box is not None
        local, local_weight = assemble(base[sl], deltas, supports, labels[sl], config)
        result = base.copy()
        result[sl] = local
        weight = np.zeros(base.shape[:2], np.float32)
        weight[sl] = local_weight
        assert np.isfinite(result).all() and np.array_equal(result[weight == 0], base[weight == 0])
        np.save(out / f'cam{target:03d}.npy', result)
        np.save(out / f'cam{target:03d}_confidence.npy', weight)
        for suffix in ['sources.npy', 'sources.png', 'reasons.npy']:
            (out / f'cam{target:03d}_{suffix}').symlink_to(old / f'cam{target:03d}_{suffix}')
        image(out / f'cam{target:03d}.png', result)
        image(out / f'cam{target:03d}_transfer.png', np.repeat(weight[..., None], 3, -1))
        for suffix in ['receipt.json', 'sources.npy']:
            p = old / f'cam{target:03d}_{suffix}'
            inputs[str(p)] = sha(p)
        write(out / f'cam{target:03d}_receipt.json', {
            **receipt, 'config': config, 'code_sha256': sha(Path(__file__)),
            'additional_input_hashes': inputs,
            'changed_fraction': float(np.any(result != base, axis=-1).mean()),
            'single_source_color': True, 'uses_target_rgb': False,
            'seam_uses_actual_admitted_correction': True})
        log(root, 'current_full_texture_complete', camera=target, candidate=name)
