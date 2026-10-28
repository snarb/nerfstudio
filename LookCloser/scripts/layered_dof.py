"""Far-background defocus before alpha composition of an in-focus subject.

Inputs are floating point RGB, not BGR. ``foreground_premult`` is the
foreground radiance contribution (alpha already applied), without the old
background. No silhouette erosion, component deletion or color selectors are
used. A correct renderer alpha and a complete clean background are required;
this is not an RGB-only segmentation or geometry repair algorithm.
"""
from __future__ import annotations

import cv2
import numpy as np
import math
from scipy.ndimage import distance_transform_edt


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.maximum(x, 0) ** (1 / 2.4) - 0.055)


def extend_background_coc(coc, alpha, support_threshold=0.001,
                          background_guard_pixels=0.):
    """Extend observed background blur radius under the occluding subject.

    Subject depth must not control the blur of background pixels behind it.
    Mixed-image depth can also be contaminated just outside the alpha support.
    An optional guard excludes that uncertain band from CoC observations; it
    changes neither alpha nor RGB. Use a depth-confidence footprint when known.
    This is only a nearest-background extrapolation, not recovered metric depth.
    If a complete calibrated background depth is available, use it instead.
    """
    if coc.shape != alpha.shape or coc.ndim != 2:
        raise ValueError("Expected matching 2D CoC and alpha")
    if not np.isfinite(background_guard_pixels) or background_guard_pixels < 0:
        raise ValueError("Background guard must be finite and nonnegative")
    missing = alpha > support_threshold
    if background_guard_pixels > 0 and missing.any():
        distance = distance_transform_edt(~missing)
        missing = distance <= background_guard_pixels
    if np.all(missing):
        raise ValueError("No background samples to extend; provide background CoC")
    indices = distance_transform_edt(missing, return_distances=False,
                                    return_indices=True)
    return coc[indices[0], indices[1]]


def gaussian_defocus(rgb, coc, sigmas=(4., 8., 16., 24., 36., 48.)):
    """Gaussian approximation; CoC and sigmas use the same pixel units.

    This intentionally matches the historical Gaussian scale interpolation;
    it is not a physically accurate aperture/bokeh simulator.
    """
    if rgb.shape != (*coc.shape, 3) or not np.isfinite(rgb).all():
        raise ValueError("Expected finite RGB and matching background CoC")
    if not np.isfinite(coc).all() or np.any(coc < 0) or coc.max() > sigmas[-1]:
        raise ValueError("CoC is outside the configured Gaussian scales")
    out = rgb.copy()
    previous = 0.
    for sigma in sigmas:
        if sigma <= previous:
            raise ValueError("Gaussian scales must increase")
        value = cv2.GaussianBlur(rgb, (0, 0), sigma,
                                 borderType=cv2.BORDER_REFLECT_101)
        amount = np.clip((coc - previous) / (sigma - previous), 0, 1)[..., None]
        out = out * (1 - amount) + value * amount
        previous = sigma
    return out


def _prepare_layers(foreground_premult, alpha, background, source_space, composite_space):
    if source_space not in ("srgb", "linear") or composite_space not in ("srgb", "linear"):
        raise ValueError("Spaces must be 'srgb' or 'linear'")
    if foreground_premult.shape != background.shape or alpha.shape != background.shape[:2]:
        raise ValueError("Layer sizes must match")
    if not all(np.isfinite(x).all() for x in (foreground_premult, alpha, background)):
        raise ValueError("Layers must be finite")
    if alpha.min() < -1e-5 or alpha.max() > 1 + 1e-5:
        raise ValueError("Alpha outside [0, 1]")
    alpha = np.clip(alpha, 0, 1)  # only numerical integration overshoot
    a = alpha[..., None]
    if np.any(foreground_premult < -1e-5) or np.any(foreground_premult > a + 1e-5):
        raise ValueError("Foreground is not a valid premultiplied [0, 1] layer")
    fg = foreground_premult
    bg = background
    if source_space != composite_space:
        straight = np.divide(fg, a, out=np.zeros_like(fg), where=a > 0)
        convert = srgb_to_linear if source_space == "srgb" else linear_to_srgb
        fg = convert(np.clip(straight, 0, 1)) * a
        bg = convert(np.clip(bg, 0, 1))
    return fg, a, bg


def compose_in_focus(foreground_premult, alpha, background, background_coc,
                     source_space="srgb", composite_space="linear"):
    """Blur only the clean background, then perform premultiplied composition.

    ``source_space`` describes the radiance field's actual blend space. Old
    pipelines often accumulated sRGB-valued training pixels. Those contributions
    must first be unassociated in that space, then decoded, then reassociated.
    Simply decoding premultiplied sRGB is incorrect. No alpha edits are made.
    """
    fg, a, bg = _prepare_layers(foreground_premult, alpha, background,
                               source_space, composite_space)
    return fg + (1 - a) * gaussian_defocus(bg, background_coc)


def disk_defocus(rgb, radius_pixels):
    """Uniform circular aperture convolution of a COMPLETE distant plate.

    Pixel areas are integrated on a fixed 8x8 subpixel grid. No depth extension,
    silhouette mask, or subject color enters this convolution. This assumes a
    distant background with constant blur radius; it is not near-scene DoF.
    """
    if not math.isfinite(radius_pixels) or radius_pixels < 0:
        raise ValueError("Aperture radius must be finite and nonnegative")
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Expected finite RGB")
    if radius_pixels == 0:
        return rgb.copy()
    half = math.ceil(radius_pixels + .5)
    yy, xx = np.mgrid[-half:half+1, -half:half+1]
    kernel = np.zeros_like(xx, dtype=np.float64)
    offsets = (np.arange(8) + .5) / 8 - .5
    for dy in offsets:
        for dx in offsets:
            kernel += (xx + dx)**2 + (yy + dy)**2 <= radius_pixels**2
    if kernel.sum() == 0:
        # A subpixel aperture entirely contained in the central pixel.
        return rgb.copy()
    kernel /= kernel.sum()
    return cv2.filter2D(rgb, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def compose_distant_in_focus(foreground_premult, alpha, background, radius_pixels,
                             source_space="srgb", composite_space="linear"):
    """Artistic all-in-focus foreground over an independently defocused plate.

    Opaque foreground is retained exactly in the selected compositing space;
    genuine partial coverage retains its original alpha. This is explicitly a
    layer effect, not a claim that a deep subject fits one optical focus plane.
    Background completion and foreground correctness remain caller obligations.
    """
    fg, a, bg = _prepare_layers(foreground_premult, alpha, background,
                               source_space, composite_space)
    return fg + (1 - a) * disk_defocus(bg, radius_pixels)
