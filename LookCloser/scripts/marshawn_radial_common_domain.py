"""Known inverse radiometric transform for cross-run comparisons; no fitting."""
import numpy as np
from marshawn_radial_photometry import radial_ev

def prediction_in_old_scalar_domain(srgb, frame, fit, old_ev):
    """Known inverse color mapping for a FAIR old-GT comparison; never fits RGB.

    Prefer unquantized model output. Fully saturated display values cannot be
    uniquely inverted; their number is returned and must be reported.
    """
    value = np.asarray(srgb, dtype=np.float64)
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
        raise ValueError('Expected finite native sRGB prediction in [0,1]')
    display_linear = np.where(value <= .04045, value/12.92, ((value+.055)/1.055)**2.4)
    saturated = int(np.count_nonzero(display_linear >= 1))
    radiance = display_linear / np.maximum(1-display_linear, np.finfo(np.float64).eps)
    evs = {r['camera']: r['scalar_correction_EV'] for r in fit['camera_rows']}
    evs.update({n: r['ev'] for n, r in fit['eval_predictions'].items()})
    total_ev = radial_ev(frame, fit['k']) - fit['radial_global_center_EV'] + evs[frame['physical_camera']]
    radiance *= np.exp2(float(old_ev)-total_ev[..., None])
    x = radiance/(1+radiance)
    encoded = np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)
    return encoded, saturated

