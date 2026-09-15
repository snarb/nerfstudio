"""Extend inferred texture regions to nearby, similarly oriented original faces."""
import numpy as np
import open3d as o3d
from component_texture_owner import components
from diffusion_mesh_repair import scene_for


def texture_regions(vertices, triangles, original_count, join_original=False, distance=.0015, normal_cosine=.8):
    v, t = np.asarray(vertices), np.asarray(triangles)
    regions = np.full(len(t), -1, int);regions[original_count:] = components(t, original_count)
    if join_original and original_count and len(t) > original_count:
        tv = v[t];normals = np.cross(tv[:, 1]-tv[:, 0], tv[:, 2]-tv[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
        centers = tv[:original_count].mean(1)
        nearest = scene_for(v, t[original_count:]).compute_closest_points(o3d.core.Tensor(centers.astype(np.float32)))
        xyz, ids = nearest['points'].numpy(), nearest['primitive_ids'].numpy()
        separation = np.linalg.norm(xyz-centers, axis=1)
        agreement = np.abs(np.sum(normals[:original_count]*normals[ids+original_count], axis=1))
        accept = (separation <= distance) & (agreement >= normal_cosine)
        regions[np.flatnonzero(accept)] = regions[ids[accept]+original_count]
    return regions


def visible_face_weights(ids, depth, face_count, regions):
    known = np.isfinite(depth) & (ids < face_count)
    count = np.bincount(ids[known].astype(int), minlength=face_count).astype(float)
    count[regions < 0] = 0
    return count
