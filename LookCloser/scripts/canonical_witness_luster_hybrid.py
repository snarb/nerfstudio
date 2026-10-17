"""Remove slow view-color differences from a TRAIN confidence reference only.

This never supplies output texture, changes geometry, shifts a lookup or fits
model parameters. The zero-strength control is the untouched TRAIN reference.
"""
import numpy as np
from transfer_luster_hybrid import gaussian
from train_perceptual_utility_luster_hybrid import perceptual_utility as raw_utility


def canonical_reference(witness, baseline, support, strength=1., sigma=6.):
    assert witness.shape==baseline.shape==(*support.shape,3)
    assert 0<=strength<=1 and sigma>0
    assert np.isfinite(witness[support]).all() and np.isfinite(baseline[support]).all()
    if strength==0:return witness.copy()
    mass=gaussian(support.astype(np.float32),sigma)
    def low(rgb):
        return gaussian(np.where(support[...,None],rgb,0),sigma)/np.maximum(mass[...,None],1e-8)
    value=np.clip(witness+strength*(low(baseline)-low(witness)),0,1)
    return np.where(support[...,None],value,baseline).astype(np.float32)


def perceptual_utility(candidate,witness,baseline,support,metric):
    reference=canonical_reference(witness,baseline,support)
    return raw_utility(candidate,reference,baseline,support,metric)
