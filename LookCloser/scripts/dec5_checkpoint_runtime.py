"""Restore non-tensor render state for sequence checkpoints, including old ones."""
import math


def restore(model,checkpoint,request):
    runtime=checkpoint.get('runtime',{})
    declared=request.get('fr_switch_after')
    expected=None if declared is None else (1. if checkpoint['step']<=declared else .3)
    strength=runtime.get('feature_reweighting_strength',expected)
    if strength is None:return None  # Legacy fixed-phase configs remain authoritative.
    strength=float(strength)
    if not math.isfinite(strength) or strength<0:raise ValueError('Invalid checkpoint feature-reweighting strength')
    if expected is not None and abs(strength-expected)>1e-8:
        raise ValueError('Checkpoint runtime disagrees with declared sequence phase')
    model.config.feature_reweighting_strength=strength
    model.field.feature_reweighting_strength=strength
    return strength
