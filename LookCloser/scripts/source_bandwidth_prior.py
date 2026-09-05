"""Train-only relative-bandwidth prior for hard source-camera selection.

Gaussian filtering is used only to measure patch correspondence, never to render.
Symmetric half-shift registration gives both patches the same extra interpolation.
No eval RGB, semantic region, camera-name exception or source RGB mixture is used.
"""
from __future__ import annotations
import cv2
import numpy as np
from audit_source_epipolar_residuals import peak_offset
from audit_warped_source_registration import relative_blur_profile


def bandwidth_observations(primary,source,primary_valid,source_valid,*,stride=48):
    if primary.shape!=source.shape or primary.ndim!=2:
        raise ValueError('Expected equal grayscale rasters')
    rows=[];h,w=primary.shape
    for y in range(32,h-32,stride):
        for x in range(32,w-32,stride):
            if not primary_valid[y-32:y+32,x-32:x+32].all():continue
            if not source_valid[y-32:y+32,x-32:x+32].all():continue
            ref=primary[y-24:y+24,x-24:x+24]
            if ref.std()<.008:continue
            search=source[y-32:y+32,x-32:x+32]
            scores=cv2.matchTemplate(search,ref,cv2.TM_CCOEFF_NORMED)
            v,u=np.unravel_index(scores.argmax(),scores.shape)
            if scores[v,u]<.8 or not (0<u<16 and 0<v<16):continue
            dx,dy=peak_offset(scores,u,v)+[u-8,v-8]
            a=cv2.getRectSubPix(primary,(48,48),(x-.5-float(dx)/2,y-.5-float(dy)/2))
            b=cv2.getRectSubPix(source,(48,48),(x-.5+float(dx)/2,y-.5+float(dy)/2))
            profile=relative_blur_profile(a,b)
            signed=profile['sigma_pixels']**2*(1 if profile['blurred_side']=='primary' else -1)
            if profile['ncc_gain']<.005:signed=0.
            rows.append({'x':x,'y':y,'dx':float(dx),'dy':float(dy),
                         'held':bool(((x//128)*73856093^(y//128)*19349663)%5==0),
                         'relative_blur_variance':float(signed),'profile':profile})
    return rows


def bandwidth_source_costs(rgb,valid,penalty=.01):
    """A constant per-source unary penalty, qualified on separate spatial blocks."""
    if (rgb.ndim!=4 or rgb.shape[-1]!=3 or valid.shape!=rgb.shape[:-1]
            or not np.isfinite(rgb).all() or not np.isfinite(penalty) or penalty<0):
        raise ValueError('Invalid train-source RGB/visibility/bandwidth penalty')
    gray=np.ascontiguousarray(rgb@np.array([.2126,.7152,.0722],np.float32),dtype=np.float32)
    costs=np.zeros(valid.shape,np.float32);stats=[]
    for rank in range(1,len(rgb)):
        rows=bandwidth_observations(gray[0],gray[rank],valid[0],valid[rank])
        fit=[r['relative_blur_variance'] for r in rows if not r['held']]
        held=[r['relative_blur_variance'] for r in rows if r['held']]
        robust=float(np.median(fit)) if fit else 0.
        qualified=(len(fit)>=20 and len(held)>=5 and robust>0
                   and np.mean(np.array(fit)>0)>=.6 and np.mean(np.array(held)>0)>=.6)
        cost=penalty*robust if qualified else 0.
        costs[rank]=cost
        stats.append({'rank':rank,'fit':len(fit),'held':len(held),'qualified':bool(qualified),
                      'fit_median_relative_variance':robust,'held_median_relative_variance':float(np.median(held)) if held else None,
                      'unary_cost':float(cost),'observations':rows})
    return costs,{'enabled':True,'uses_eval_rgb':False,'uses_semantic_masks':False,
                  'modifies_rgb':False,'source_averaging':False,'penalty_per_pixel_variance':penalty,
                  'primary_penalty_zero':True,'sources':stats}
