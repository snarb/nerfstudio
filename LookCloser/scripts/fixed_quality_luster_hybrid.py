"""Reject unreliable selected-photo detail without changing automatic source labels.

Projected TRAIN texture and frozen LookCloser supply the evidence. Evaluation
images/masks and other source RGB never enter the output or confidence.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import geometry, image, read, write, sha, log, native_rows, assert_source
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import gaussian
cv2.setNumThreads(4)


def detail_confidence(base, photo, valid, settings):
    """Measure aligned detail on valid support, then fade only rejected pixels."""
    support = valid.astype(np.float32)
    def low(rgb, sigma):
        mass = gaussian(support, sigma)
        return gaussian(rgb * support[..., None], sigma) / np.maximum(mass[..., None], 1e-8)
    b = low(base, 1.) - low(base, 4.)
    p = low(photo, 1.) - low(photo, 4.)
    mass = gaussian(support, 3.)
    def average(x):
        return gaussian(x * support, 3.) / np.maximum(mass, 1e-8)
    eb2 = average(np.mean(b*b, axis=-1))
    ep2 = average(np.mean(p*p, axis=-1))
    cross = average(np.mean(b*p, axis=-1))
    eb = np.sqrt(np.maximum(eb2, 0))
    ratio = np.sqrt(np.maximum(ep2, 0)) / np.maximum(eb, 1e-8)
    correlation = cross / np.maximum(np.sqrt(np.maximum(eb2*ep2, 0)), 1e-8)
    reliable = eb >= settings['energy_floor']
    good = (~reliable) | ((ratio >= settings['sharpness_min']) & (correlation >= settings['phase_min']))
    # Keep full strength away from failures; fade toward exact LC around them.
    confidence = np.clip((gaussian((good & valid).astype(np.float32), 2.) / np.maximum(gaussian(support, 2.), 1e-8) - .5) / .5, 0, 1)
    confidence *= good & valid
    return confidence.astype(np.float32), ratio, correlation


def combine(root, parent, name, settings):
    old = root/'candidates'/parent
    out = root/'candidates'/name
    out.mkdir(parents=True, exist_ok=True)
    config = {**read(old/'config.json'), 'mode':'single_source_fixed_quality', 'parent':parent, **settings}
    if (out/'config.json').exists():
        assert read(out/'config.json') == config
    else:
        write(out/'config.json', config)
    rows = native_rows()
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():
            continue
        receipt = read(old/f'cam{target:03d}_receipt.json')
        warp_root = Path(receipt.get('warp_root', root))
        base = np.asarray(geometry(root,target)['rgb'])
        labels = np.load(old/f'cam{target:03d}_sources.npy')
        weight = np.load(old/f'cam{target:03d}_confidence.npy')
        first = warp_root/'warps'/f'cam{target:03d}'/f"src{receipt['sources'][0]:03d}"
        x0,y0,x1,y1 = read(first/'receipt.json')['box']
        sl = np.s_[y0:y1,x0:x1]
        gate = np.zeros(labels.shape, np.float32)
        ratio_map = np.zeros(labels.shape, np.float32)
        phase_map = np.zeros(labels.shape, np.float32)
        for cid in receipt['sources']:
            selected = labels[sl] == cid
            if not selected.any():
                continue
            assert_source(rows,target,cid)
            d = warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}'
            r = read(d/'receipt.json')
            a = {k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
            valid = geometric_support(a, config)
            confidence,ratio,phase = detail_confidence(base[sl],a['rgb'],valid,config)
            gate[sl][selected] = confidence[selected]
            ratio_map[sl][selected] = ratio[selected]
            phase_map[sl][selected] = phase[selected]
        pred = np.load(old/f'cam{target:03d}.npy')
        result = np.clip(base + gate[...,None] * (pred-base),0,1)
        weight *= gate
        labels[weight == 0] = -1
        reasons = np.load(old/f'cam{target:03d}_reasons.npy').copy()
        reasons[(gate == 0) & (np.load(old/f'cam{target:03d}_sources.npy') >= 0)] = 6
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        for suffix,array in [('',result),('_sources',labels),('_confidence',weight),('_reasons',reasons),('_quality_gate',gate),('_sharpness_ratio',ratio_map),('_phase',phase_map)]:
            np.save(out/f'cam{target:03d}{suffix}.npy',array)
        image(out/f'cam{target:03d}.png',result)
        image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        palette=np.random.default_rng(42).uniform(.15,1,(166,3)).astype(np.float32)
        colors=palette[np.maximum(labels,0)]; colors[labels<0]=0
        image(out/f'cam{target:03d}_sources.png',colors)
        inputs = {str(old/f'cam{target:03d}{suffix}'):sha(old/f'cam{target:03d}{suffix}') for suffix in ['.npy','_sources.npy','_confidence.npy','_reasons.npy','_receipt.json']}
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,'changed_fraction':float(np.any(result!=base,axis=-1).mean()),'reasons':{**receipt['reasons'],'6':'selected photo fails detail quality'}})
        log(root,'fixed_source_quality_complete',camera=target,candidate=name)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='single_cubic_fixed');p.add_argument('--name',required=True)
    p.add_argument('--sharpness-min',type=float,default=.8);p.add_argument('--phase-min',type=float,default=-1.);p.add_argument('--energy-floor',type=float,default=.004)
    a=p.parse_args()
    combine(a.root,a.parent,a.name,dict(sharpness_min=a.sharpness_min,phase_min=a.phase_min,energy_floor=a.energy_floor))
