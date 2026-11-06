"""Recover original SIFT graph observations without substituting projected UV.

Projection scopes candidate detections only. Original directed descriptor-ratio
edges identify the feature component, and its measured detections are then
triangulated. This recovers already fitted TRAIN evidence, not independent GT.
"""
import itertools

import cv2
import numpy as np

from audit_marshawn_geometry import projection


def triangulate_detections(component, rows, features):
    equations = []
    for camera, key in component:
        matrix, *_ = projection(rows[camera])
        u, v = features[camera]['uv'][key]
        equations.extend([u*matrix[2]-matrix[0], v*matrix[2]-matrix[1]])
    _, _, vh = np.linalg.svd(equations)
    if abs(vh[-1, 3]) < 1e-12:
        raise ValueError('Degenerate triangulation')
    return vh[-1, :3]/vh[-1, 3]


def recover_track(xyz, cameras, rows, features, allowed_pairs, tolerance=1e-8):
    """Return actual feature indices/UV only for an exact original track replay."""
    if len(cameras) < 3 or len(set(cameras)) != len(cameras):
        raise ValueError('At least three distinct original source cameras required')
    nodes, edges = {}, []
    for camera in cameras:
        matrix, *_ = projection(rows[camera])
        q = matrix @ np.r_[xyz, 1.]
        if q[2] <= 0:
            raise ValueError('Witness behind source camera')
        nodes[camera] = np.flatnonzero(np.linalg.norm(
            features[camera]['uv']-q[:2]/q[2], axis=1) < 1.001).tolist()
        if not nodes[camera]:
            raise ValueError('Original detected feature unavailable')
    for a, b in itertools.combinations(nodes, 2):
        if (a, b) not in allowed_pairs:
            if (b, a) not in allowed_pairs:
                continue
            a, b = b, a
        matches = cv2.BFMatcher().knnMatch(features[a]['des'][nodes[a]], features[b]['des'], k=2)
        for ia, pair in zip(nodes[a], matches):
            if len(pair) != 2:
                continue
            m, n = pair
            if m.trainIdx in nodes[b] and m.distance < .65*n.distance:
                edges.append(((a, ia), (b, m.trainIdx)))
    graph = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    seen, options = set(), []
    for node in graph:
        if node in seen:
            continue
        component = {node}
        while True:
            expanded = component | set.union(*(graph[n] for n in component))
            if expanded == component:
                break
            component = expanded
        seen |= component
        if len(component) != len(cameras) or {n[0] for n in component} != set(cameras):
            continue
        component = sorted(component)
        recovered = triangulate_detections(component, rows, features)
        error = float(np.max(np.abs(recovered-xyz)))
        options.append((error, component, recovered))
    if not options or min(x[0] for x in options) > tolerance:
        raise ValueError('Descriptor graph does not reproduce original witness')
    error, component, recovered = min(options, key=lambda x: x[0])
    observations, leave_one_out = [], []
    for j, (camera, key) in enumerate(component):
        matrix, *_ = projection(rows[camera])
        q = matrix @ np.r_[recovered, 1.]
        uv = features[camera]['uv'][key]
        if q[2] <= 0:
            raise ValueError('Recovered feature behind camera')
        residual = float(np.linalg.norm(q[:2]/q[2]-uv))
        if residual > 1.001:
            raise ValueError('Original detector residual threshold exceeded')
        observations.append(dict(camera=camera, keypoint_index=int(key), detected_uv=uv.tolist(),
                                 fitted_error_hd_px=residual))
        if len(component) >= 4:
            left = triangulate_detections(component[:j]+component[j+1:], rows, features)
            held = matrix @ np.r_[left, 1.]
            leave_one_out.append(dict(camera=camera,
                heldout_raw_feature_error_hd_px=float(np.linalg.norm(held[:2]/held[2]-uv)),
                xyz_shift_world=float(np.linalg.norm(left-recovered)),
                positive_heldout_depth=bool(held[2] > 0)))
    return dict(observations=observations, original_xyz_max_abs=error,
        xyz_original=recovered.tolist(), fit_error_max_hd_px=max(x['fitted_error_hd_px'] for x in observations),
        connected_original_ratio_edges=[[list(a), list(b)] for a, b in edges
                                        if a in component and b in component],
        leave_one_source_view_out=leave_one_out,
        limitation='Existing rig and descriptor-selected TRAIN graph; fit residuals are not independent calibration uncertainty')
