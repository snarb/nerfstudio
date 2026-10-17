"""Fit slow color/contrast of one native photo to frozen LookCloser appearance.

No target photograph, label reassignment, image registration or new model is
used. One selected source supplies all texture; other sources only seam costs.
"""
from pathlib import Path
import numpy as np
from luster_hybrid_common import assert_source, geometry, image, log, native_rows, read, sha, write
from transfer_luster_hybrid import gaussian
from supported_luster_hybrid import geometric_support
from band_witness_luster_hybrid import assemble


def correction(base, photo, support, base_sigma, fit_sigma, fit_affine):
    """A spatially smooth color fit; the photo coordinates never change."""
    assert base.shape == photo.shape == (*support.shape, 3)
    assert base_sigma > 0 and fit_sigma >= 0
    sf = support.astype(np.float32)
    def mean(rgb, sigma):
        mass = gaussian(sf, sigma)
        return gaussian(rgb * sf[..., None], sigma) / np.maximum(mass[..., None], 1e-8), mass
    pl, _ = mean(photo, base_sigma)
    bl, _ = mean(base, base_sigma)
    reference = (photo - pl) - (base - bl)
    if fit_sigma == 0:
        return reference
    mp, mass = mean(photo, fit_sigma)
    mb, _ = mean(base, fit_sigma)
    if fit_affine:
        # Suppress the finest unresolved detail only for estimating the color
        # transform; the output samples remain the original cubic photo values.
        pf, _ = mean(photo, 2.)
        bf, _ = mean(base, 2.)
        ep, _ = mean(pf, fit_sigma)
        eb, _ = mean(bf, fit_sigma)
        epp, _ = mean(pf * pf, fit_sigma)
        epb, _ = mean(pf * bf, fit_sigma)
        variance = np.maximum(epp - ep * ep, 0)
        covariance = epb - ep * eb
        slope = np.clip((covariance + .0001) / (variance + .0001), .5, 2.)
        # Slow parameter variation prevents tiny fitted patches imprinting edges.
        slope, _ = mean(slope, 16.)
        bias = mb - slope * mp
        fitted = slope * photo + bias - base
    else:
        fitted = (photo - mp) - (base - mb)
    # Large neighborhoods near silhouettes retain the proven detail correction.
    admission = np.clip((mass - .6) / .2, 0, 1)
    return reference + admission[..., None] * (fitted - reference)


def combine(root, parent, name, fit_sigma, fit_affine):
    old = root / 'candidates' / parent
    out = root / 'candidates' / name
    out.mkdir(parents=True, exist_ok=True)
    config = {**read(old / 'config.json'), 'mode': 'single_source_local_affine_texture',
              'parent': parent, 'fit_sigma': fit_sigma, 'fit_affine': fit_affine,
              'fit_ridge': .0001, 'fit_slope_bounds': [.5, 2.],
              'fit_input_sigma': 2., 'fit_slope_sigma': 16., 'fit_support_ramp': [.6, .8]}
    assert config['filter_support'] == 'geometry' and config['adaptive_seam']
    assert fit_sigma >= 0
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
            deltas[cid] = correction(base[sl], np.asarray(a['rgb']), support, config['sigma'], fit_sigma, fit_affine)
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
        log(root, 'local_affine_texture_complete', camera=target, candidate=name)
