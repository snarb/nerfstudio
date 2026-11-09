"""Pull normalized image-grid resolution back to a bounded 3D encoding domain."""
from __future__ import annotations
import torch
from torch import Tensor


def normalized_frequency_resolution(
    positions: Tensor,
    camera_to_worlds: Tensor,
    focal_xy: Tensor,
    image_wh: Tensor,
    encoding_aabb: Tensor,
    image_resolution: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return local required grid resolution and valid front-facing points.

    The 2D regressor encodes (pixel_x / W, pixel_y / H), while the 3D field
    encodes (world - box_min) / box_extent. Pulling back normalized image
    frequencies therefore uses d(image_uv)/d(encoding_xyz), not pixel focal
    length divided by ray distance. Its largest singular value is an explicit
    conservative scalar convention for anisotropic bounds and off-axis rays.
    It preserves camera calibration and is invariant to uniform scene scaling
    and image resizing. Supports undistorted perspective cameras only.
    """
    if encoding_aabb.shape != (2, 3) or not torch.all(encoding_aabb[1] > encoding_aabb[0]):
        raise ValueError('Frequency projection needs a positive bounded encoding AABB')
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError('Expected N world positions')
    if not torch.all(image_wh > 0) or not torch.all(focal_xy > 0):
        raise ValueError('Image dimensions and focal lengths must be positive')
    rotation = camera_to_worlds[:, :3, :3]
    local = torch.bmm(rotation.transpose(1, 2), (positions-camera_to_worlds[:, :3, 3])[..., None])[..., 0]
    x, y, negative_z = local.unbind(-1)
    z = -negative_z
    valid = torch.isfinite(local).all(-1) & (z > 0)
    valid &= torch.isfinite(image_resolution) & (image_resolution > 0)
    safe_z = torch.where(valid, z, torch.ones_like(z))
    fx, fy = (focal_xy / image_wh).unbind(-1)
    zero = torch.zeros_like(z)
    jac = torch.stack((fx/safe_z, zero, fx*x/safe_z.square(),
                       zero, -fy/safe_z, -fy*y/safe_z.square()), -1).reshape(-1, 2, 3)
    jac = jac @ rotation.transpose(1, 2)
    jac = jac * (encoding_aabb[1]-encoding_aabb[0])[None, None, :]
    jac = torch.where(valid[:, None, None], jac, torch.zeros_like(jac))
    # Largest eigenvalue of JJ^T, avoiding thousands of tiny SVDs on the GPU.
    gram = jac @ jac.transpose(1, 2)
    a, b, d = gram[:, 0, 0], gram[:, 0, 1], gram[:, 1, 1]
    stretch = ((a+d+torch.sqrt((a-d).square()+4*b.square()))*.5).clamp_min(0).sqrt()
    resolution = torch.where(valid, image_resolution*stretch, torch.zeros_like(z))
    valid &= torch.isfinite(resolution) & (resolution > 0)
    return resolution, valid
