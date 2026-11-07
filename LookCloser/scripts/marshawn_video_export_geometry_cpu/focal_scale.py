"""Smooth reference-HD focal; never switch the denominator to a nearest camera."""
import numpy as np
from scipy.interpolate import CubicSpline
ANCHORS=('IC','JC','KC','LC','MC')
def focal_scale(target_row,time_seconds,calibration_rows):
 t=float(time_seconds)
 if not np.isfinite(t)or not 0<=t<=3:raise ValueError('Neural path clock must lie in [0,3]')
 rows=[calibration_rows[n]for n in ANCHORS]
 if any((r['w'],r['h'])!=(1920,1012)for r in rows):raise ValueError('Use canonical original HD calibration')
 log=np.log([[r['fl_x'],r['fl_y']]for r in rows]);u=t/3;u=u*u*u*(10+u*(-15+6*u));reference=np.exp(CubicSpline(np.linspace(0,1,5),log,bc_type='natural')(u));scale=np.sqrt(target_row['fl_x']*target_row['fl_y']/np.prod(reference))
 if not np.isfinite(scale)or scale<=0:raise ValueError('Positive finite angular pixel scale required')
 return float(scale),dict(reference_HD_fx=float(reference[0]),reference_HD_fy=float(reference[1]),path_clock=float(u),reference_anchors=ANCHORS,nearest_camera_switching=False)
