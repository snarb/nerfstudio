"""Scalar curvature-phase agreement at coherent frozen-render edges.

This measures Laplacian phase at unchanged projected coordinates. It estimates
no displacement and performs no registration or texture-coordinate correction.
"""
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import assert_source, geometry, native_rows, read, sha, write
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import gaussian


def curvature_phase(base, photo, support):
    sf = support.astype(np.float32)
    luma = np.array([.2126, .7152, .0722], np.float32)
    mass2, mass3, mass4 = [gaussian(sf, s) for s in [2., 3., 4.]]
    def low(rgb):
        return gaussian((rgb @ luma) * sf, 2.) / np.maximum(mass2, 1e-8)
    b, p = low(base), low(photo)
    bx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
    by = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
    def tensor_mean(v):
        return gaussian(v * sf, 4.) / np.maximum(mass4, 1e-8)
    xx, yy, xy = tensor_mean(bx * bx), tensor_mean(by * by), tensor_mean(bx * by)
    energy = np.maximum(xx + yy, 0)
    coherence = np.sqrt(np.maximum((xx - yy)**2 + 4 * xy * xy, 0)) / np.maximum(energy, 1e-12)
    bl = cv2.Laplacian(b, cv2.CV_32F, ksize=3, scale=.25)
    pl = cv2.Laplacian(p, cv2.CV_32F, ksize=3, scale=.25)
    def mean(v):
        return gaussian(v * sf, 3.) / np.maximum(mass3, 1e-8)
    bb, pp, cross = mean(bl * bl), mean(pl * pl), mean(bl * pl)
    corr = np.clip(cross / np.maximum(np.sqrt(np.maximum(bb * pp, 0)), 1e-12), -1, 1)
    known = support & (energy >= .002**2) & (coherence >= .8)
    known &= (bb >= .0002**2) & (pp >= .0002**2) & (mass4 >= .9)
    known &= cv2.erode(sf, np.ones((13, 13), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return np.where(known, .5 * (1 - corr), .5).astype(np.float32), known


def quality(root, target, source, config):
    assert_source(native_rows(), target, source)
    out = root / 'edge_position_quality' / f'cam{target:03d}' / f'src{source:03d}'
    d = root / 'variants/cubic75/warps' / f'cam{target:03d}' / f'src{source:03d}'
    gates = {k: config[k] for k in ['alpha', 'depth_error', 'roundtrip', 'uncertainty', 'max_score', 'min_resolution']}
    if (out / 'receipt.json').exists():
        old = read(out / 'receipt.json')
        assert old['code_sha256'] == sha(Path(__file__)) and old['gates'] == gates
        assert all(sha(out / k) == v for k, v in old['hashes'].items())
        return out
    out.mkdir(parents=True, exist_ok=True)
    r = read(d / 'receipt.json')
    x0, y0, x1, y1 = r['box']
    arrays = {k: np.load(d / f'{k}.npy', mmap_mode='r') for k in r['arrays']}
    support = geometric_support(arrays, config)
    base = np.asarray(geometry(root, target)['rgb'][y0:y1, x0:x1])
    cost, known = curvature_phase(base, np.asarray(arrays['rgb']), support)
    np.save(out / 'cost.npy', cost)
    np.save(out / 'known.npy', known)
    inputs = {str(p): sha(p) for p in [d / 'receipt.json', root / 'geometry' / f'cam{target:03d}' / 'receipt.json']}
    for module in ['supported_luster_hybrid.py', 'transfer_luster_hybrid.py', 'luster_hybrid_common.py']:
        p = Path(__file__).with_name(module)
        inputs[str(p)] = sha(p)
    write(out / 'receipt.json', dict(target=target, source=source, box=r['box'], gates=gates,
        code_sha256=sha(Path(__file__)), inputs=inputs,
        hashes={k: sha(out / k) for k in ['cost.npy', 'known.npy']},
        feature='G2 luma Laplacian phase, sigma3 moments, coherent strong frozen-render edges',
        uses_target_rgb=False, uses_masks=False, lookup_shift=False))
    return out
