"""Opt-in adaptation of integer-centered COLMAP depth to Open3D VBG indexing.

COLMAP 5509fffe MVS unprojects (column,row), without a half-pixel offset.
Open3D 0.19 tensor integration truncates positive projected coordinates to int.
Adding .5 to K's principal point for the integration lookup implements nearest
integer depth sampling. Block discovery keeps the original unprojection K.
This does not establish that COLMAP's convention matches physical RGB centers.
"""
from __future__ import annotations

import numpy as np


def nearest_integer_lookup_intrinsics(intrinsics):
    matrix = np.asarray(intrinsics)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Need a finite 3x3 intrinsic matrix')
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.array_equal(matrix[2], [0, 0, 1]):
        raise ValueError('Need positive focal lengths and a pinhole bottom row')
    result = matrix.astype(np.float64, copy=True)
    result[0, 2] += .5
    result[1, 2] += .5
    return result


class IntegerCenteredDepthVolume:
    """Delegate all VBG operations except the depth-only integration lookup.

    The six-argument depth-only integration API is deliberately explicit. Color
    integration and differently calibrated depth/color images are not supported.
    """

    def __init__(self, volume):
        self.volume = volume

    def __getattr__(self, name):
        return getattr(self.volume, name)

    def integrate(self, block_coords, depth_image, intrinsic, extrinsic,
                  depth_scale=1., depth_max=4., trunc_voxel_multiplier=8.):
        import open3d as o3d
        corrected = o3d.core.Tensor(nearest_integer_lookup_intrinsics(intrinsic.cpu().numpy()))
        return self.volume.integrate(block_coords, depth_image, corrected, extrinsic,
                                     depth_scale=depth_scale, depth_max=depth_max,
                                     trunc_voxel_multiplier=trunc_voxel_multiplier)
