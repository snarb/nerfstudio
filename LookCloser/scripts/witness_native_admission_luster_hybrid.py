"""Conservative single-photo admission from native TRAIN witness utility.

Only scalar confidence comes from witnesses. The current selected photograph,
fixed q50 surface and cubic lookup are preserved. No evaluation image is read.
"""
from pathlib import Path
import numpy as np
from luster_hybrid_common import assert_source, geometry, image, log, native_rows, read, sha, write
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import gaussian
from current_full_texture_luster_hybrid import correction
from band_witness_luster_hybrid import assemble


def admission(cost, support, start=.4, stop=.6):
    """Neutral/beneficial evidence is unchanged; strong adverse evidence fades."""
    assert 0 <= start < stop <= 1 and cost.shape == support.shape
    assert np.isfinite(cost).all() and np.all((cost >= 0) & (cost <= 1))
    risk = 2 * (cost - .5)
    gate = np.clip((stop - risk) / (stop - start), 0, 1)
    sf = support.astype(np.float32)
    mass = gaussian(sf, 2.)
    smooth = gaussian(gate * sf, 2.) / np.maximum(mass, 1e-8)
    return np.where(support, np.clip(smooth, 0, 1), 1).astype(np.float32)


def combine(root, parent, name, target):
    old = root / 'candidates' / parent
    out = root / 'candidates' / name
    out.mkdir(parents=True, exist_ok=True)
    config = {**read(old / 'config.json'), 'mode': 'single_source_witness_native_admission',
              'parent': parent, 'admission_risk_start': .4, 'admission_risk_stop': .6,
              'admission_sigma': 2.}
    if (out / 'config.json').exists():
        assert read(out / 'config.json') == config
    else:
        write(out / 'config.json', config)
    if (out / f'cam{target:03d}_receipt.json').exists():
        return
    receipt = read(old / f'cam{target:03d}_receipt.json')
    warp_root = Path(receipt['warp_root'])
    base = np.asarray(geometry(root, target)['rgb'])
    labels = np.load(old / f'cam{target:03d}_sources.npy', mmap_mode='r')
    rows = native_rows()
    deltas, supports, gates = {}, {}, {}
    modules = ['band_witness_luster_hybrid.py', 'supported_luster_hybrid.py',
               'transfer_luster_hybrid.py', 'luster_hybrid_common.py', 'current_full_texture_luster_hybrid.py']
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
        ed = root / 'witness_native_restoration_projected' / f'cam{target:03d}' / f'src{cid:03d}'
        er = read(ed / 'receipt.json')
        assert er['box'] == box and er['target'] == target and er['source'] == cid
        assert sha(ed / 'cost.npy') == er['hashes']['cost.npy']
        gate = admission(np.load(ed / 'cost.npy', mmap_mode='r'), support)
        deltas[cid] = correction(base[sl], np.asarray(a['rgb']), support, config['sigma'], 0.) * gate[..., None]
        supports[cid] = support
        gates[cid] = gate
        for p in [d / 'receipt.json', ed / 'receipt.json', ed / 'cost.npy']:
            inputs[str(p)] = sha(p)
    assert box is not None
    local, seam_weight = assemble(base[sl], deltas, supports, labels[sl], config)
    selected_gate = np.zeros(labels[sl].shape, np.float32)
    for cid, gate in gates.items():
        selected = labels[sl] == cid
        selected_gate[selected] = gate[selected]
    result = base.copy()
    result[sl] = local
    weight = np.zeros(base.shape[:2], np.float32)
    weight[sl] = seam_weight * selected_gate
    final_labels = np.where(weight > 0, labels, -1).astype(np.int16)
    reasons = np.load(old / f'cam{target:03d}_reasons.npy').copy()
    reasons[(labels >= 0) & (weight == 0)] = 6
    assert np.isfinite(result).all() and np.array_equal(result[weight == 0], base[weight == 0])
    for suffix, array in [('', result), ('_confidence', weight), ('_sources', final_labels), ('_reasons', reasons), ('_admission', selected_gate)]:
        np.save(out / f'cam{target:03d}{suffix}.npy', array)
    image(out / f'cam{target:03d}.png', result)
    image(out / f'cam{target:03d}_transfer.png', np.repeat(weight[..., None], 3, -1))
    palette = np.random.default_rng(42).uniform(.15, 1, (166, 3)).astype(np.float32)
    colors = palette[np.maximum(final_labels, 0)]
    colors[final_labels < 0] = 0
    image(out / f'cam{target:03d}_sources.png', colors)
    for suffix in ['receipt.json', 'sources.npy', 'reasons.npy']:
        p = old / f'cam{target:03d}_{suffix}'
        inputs[str(p)] = sha(p)
    write(out / f'cam{target:03d}_receipt.json', {
        **receipt, 'config': config, 'code_sha256': sha(Path(__file__)),
        'additional_input_hashes': inputs, 'admission_box': box,
        'admission_sha256': sha(out / f'cam{target:03d}_admission.npy'),
        'changed_fraction': float(np.any(result != base, axis=-1).mean()),
        'reasons': {**receipt['reasons'], '6': 'strong native TRAIN witness restoration contradiction'},
        'confidence_is_effective_transfer_weight': True,
        'uses_target_rgb': False, 'seam_uses_actual_admitted_correction': True})
    log(root, 'witness_native_admission_complete', camera=target, candidate=name)
