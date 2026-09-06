"""Conservative native-image noise-floor diagnostic and opt-in NLM control.

The single-image Haar statistic is not an identified sensor-noise or lens-PSF
measurement. Texture, JPEG coding and prior image processing can contribute.
Filtering is deterministic per native train image, independent of target view.
"""
from __future__ import annotations
import math
import cv2
import numpy as np


def high_frequency_floor(rgb8, tile_size=32):
    if (rgb8.ndim!=3 or rgb8.shape[-1]!=3 or rgb8.dtype!=np.uint8
            or min(rgb8.shape[:2])<tile_size or tile_size<8 or tile_size%2):
        raise ValueError('Need uint8 RGB image and an even native tile size >=8')
    gray=rgb8.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)
    base=cv2.GaussianBlur(gray,(0,0),2.)
    rows=[]
    for y in range(0,gray.shape[0]-tile_size+1,tile_size):
        for x in range(0,gray.shape[1]-tile_size+1,tile_size):
            patch=gray[y:y+tile_size,x:x+tile_size];low=base[y:y+tile_size,x:x+tile_size]
            if not 25<float(patch.mean())<230:continue
            # Unit-L2-norm HH coefficients: independent Gaussian noise retains
            # its variance. Median-centered MAD is robust to sparse edges.
            hh=(patch[::2,::2]-patch[1::2,::2]-patch[::2,1::2]+patch[1::2,1::2])*.5
            sigma=float(np.median(np.abs(hh-np.median(hh)))/.6744897501960817)
            rows.append(dict(x=x,y=y,base_std=float(low.std()),hh_sigma_rgb8=sigma))
    if len(rows)<8:raise ValueError('Too few unclipped native tiles for a floor diagnostic')
    cutoff=float(np.quantile([r['base_std'] for r in rows],.25))
    admitted=[r for r in rows if r['base_std']<=cutoff]
    values=np.array([r['hh_sigma_rgb8'] for r in admitted])
    return dict(method='HH-MAD on lowest-base-variation quartile of unclipped native tiles',
        tile_size=tile_size,eligible_tiles=len(rows),admitted_tiles=len(admitted),base_std_cutoff=cutoff,
        sigma_rgb8=float(np.median(values)),sigma_rgb8_p10=float(np.quantile(values,.1)),
        sigma_rgb8_p90=float(np.quantile(values,.9)),physical_sensor_noise_identified=False,
        uses_eval_rgb=False,uses_semantic_masks=False,tiles=admitted)


def filter_native_rgb(rgb8, strength, *, floor=None):
    if not math.isfinite(strength) or strength<0 or strength>2:
        raise ValueError('NLM strength must be finite in 0..2')
    if rgb8.ndim!=3 or rgb8.shape[-1]!=3 or rgb8.dtype!=np.uint8:raise ValueError('Need native uint8 RGB')
    if strength==0:return rgb8.copy(),dict(enabled=False,strength=0,h_rgb8=0,exact_identity=True)
    floor=high_frequency_floor(rgb8) if floor is None else floor
    sigma=floor['sigma_rgb8']
    if not math.isfinite(sigma) or sigma<0:raise ValueError('Invalid native high-frequency floor')
    h=float(np.clip(sigma,.5,3.)*strength)
    # Channelwise NLM avoids an additional Lab/color transform. Each output uses
    # only samples from this native camera image; no random detail is generated.
    filtered=np.stack([cv2.fastNlMeansDenoising(np.ascontiguousarray(rgb8[...,c]),None,h,7,21) for c in range(3)],-1)
    return filtered,dict(enabled=True,strength=strength,h_rgb8=h,template_window=7,search_window=21,
        method='independent-channel native RGB8 nonlocal means',native_camera_only=True,
        uses_eval_rgb=False,uses_semantic_masks=False,view_independent=True,source_camera_averaging=False,
        random_detail_added=False,changed_native_pixels=int((filtered!=rgb8).any(-1).sum()),
        maximum_rgb8_change=int(np.abs(filtered.astype(np.int16)-rgb8.astype(np.int16)).max()))
