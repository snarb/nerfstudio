"""Project measured AABB edges onto unchanged camera renders for visual review."""
from itertools import product
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def project_box(bounds, camera):
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
        raise ValueError("Expected finite ordered AABB")
    bits = np.asarray(list(product([0, 1], repeat=3)))
    corners = np.where(bits, bounds[1], bounds[0])
    pose = np.asarray(camera["transform_matrix"], dtype=np.float64)
    local = (corners - pose[:3, 3]) @ pose[:3, :3]
    depth = -local[:, 2]  # Nerfstudio/OpenGL camera looks along local -Z.
    if (depth <= 1e-6).any():
        raise ValueError("This overlay requires all box vertices in front of camera")
    # Camera principal point uses COLMAP half-pixel centers; PIL uses array indices.
    uv = np.stack([camera["fl_x"] * local[:, 0] / depth + camera["cx"] - .5,
                   -camera["fl_y"] * local[:, 1] / depth + camera["cy"] - .5], axis=-1)
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if np.count_nonzero(bits[i] != bits[j]) == 1]
    assert len(edges) == 12
    return corners, uv, edges


def overlay_box(source, bounds, camera, label="", dash=14, gap=10, width=3):
    image = source.convert("RGB").copy()
    if image.size != (camera["w"], camera["h"]):
        raise ValueError("Render and calibrated camera raster differ")
    _, uv, edges = project_box(bounds, camera)
    draw = ImageDraw.Draw(image)
    for i, j in edges:
        start, end = uv[i], uv[j]
        length = float(np.linalg.norm(end - start))
        if length == 0:
            continue
        direction = (end - start) / length
        for offset in np.arange(0, length, dash + gap):
            a, b = start + direction * offset, start + direction * min(offset + dash, length)
            xy = [tuple(a), tuple(b)]
            draw.line(xy, fill=(12, 12, 12), width=width + 2)
            draw.line(xy, fill=(255, 35, 35), width=width)
    if label:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        box = draw.textbbox((12, 10), label, font=font)
        draw.rectangle((box[0] - 5, box[1] - 4, box[2] + 5, box[3] + 4), fill=(12, 12, 12))
        draw.text((12, 10), label, font=font, fill=(255, 255, 255))
    return image, {"projected_corners_array_xy": uv.tolist(), "edges": edges,
                   "line_style": "red dashed, 14px dash / 10px gap / 3px width",
                   "occlusion_style": "all twelve edges visible, including edges behind actors",
                   "camera_convention": "OpenGL -Z forward; COLMAP centers converted to array indices"}
