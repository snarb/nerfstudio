"""CPU support bound from trilinear LOGIT nodes to native occupancy cells.

This certifies support of a specified semantic density selector, not physical
empty space or actor truth. It never changes a model, occupancy or source mask.
The node field is XYZ; native cascade cell bounds are supplied explicitly.
"""
from __future__ import annotations

import numpy as np


def _bounds(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (2, 3) or not np.isfinite(value).all() or (value[1] <= value[0]).any():
        raise ValueError("Finite increasing [2,3] XYZ bounds required")
    return value


def fresh_dilation(core, radius):
    """Fresh 26-neighbour native-cell halo, separate from semantic node support."""
    core = np.asarray(core)
    if core.dtype != np.bool_ or core.ndim != 3 or radius not in (0, 1):
        raise ValueError("Boolean XYZ core and radius 0 or 1 required")
    if radius == 0:
        return core.copy()
    padded = np.pad(core, 1)
    out = np.zeros_like(core)
    for x in range(3):
        for y in range(3):
            for z in range(3):
                out |= padded[x:x+core.shape[0], y:y+core.shape[1], z:z+core.shape[2]]
    return out


class LogitNodeCellBridge:
    """Conservative maximum over every node basis touching each native cell.

    Trilinear interpolation is a convex combination of involved logits. Thus a
    cell cannot pass a monotone selector if every involved node's upper bound
    fails it. Bounding nodes can be looser than the exact piecewise-trilinear
    maximum; that only retains extra cells. The 8 native corners alone are NOT
    a valid maximum when the cell crosses semantic knot planes.

    numerical_logit_guard is explicit arithmetic slack, not a geometric buffer.
    A deployment claim for float CUDA needs a bound/probe for its implementation;
    CPU tests alone certify real interpolation plus the declared slack. Unknown
    or nonfinite node values fail closed rather than being interpreted as free.
    """
    def __init__(self, logits, bounds, *, selector, align_corners=True,
                 world_to_box, coordinate_guard=2e-7, numerical_logit_guard=1e-5):
        values = np.asarray(logits)
        if values.ndim != 3 or min(values.shape) < 2 or not np.isfinite(values).all():
            raise ValueError("Finite XYZ logit nodes, each axis >=2, required")
        if not isinstance(align_corners, (bool, np.bool_)):
            raise ValueError("Explicit boolean align_corners required")
        frame = np.asarray(world_to_box, dtype=np.float64)
        if frame.shape != (4, 4) or not np.array_equal(frame, np.eye(4)):
            raise ValueError("Direct LC floor-frame identity required; transform is not inferred")
        if not np.isfinite([coordinate_guard, numerical_logit_guard]).all() or min(coordinate_guard, numerical_logit_guard) < 0:
            raise ValueError("Finite nonnegative numerical guards required")
        self.bounds = _bounds(bounds)
        self.resolution = np.asarray(values.shape)
        self.align_corners = bool(align_corners)
        self.coordinate_guard = float(coordinate_guard)
        self.numerical_logit_guard = float(numerical_logit_guard)
        self.selector = selector
        if selector == 'soft_probability':
            # Finite sigmoid has positive support in exact arithmetic. Small p
            # cannot imply small selected density when native sigma is unbounded.
            potential = np.ones(values.shape, dtype=bool)
        elif selector in ('hard_theta01', 'hard_theta05', 'exactzero_smoothstep01_05'):
            theta = .5 if selector == 'hard_theta05' else .1
            threshold = np.log(theta)-np.log1p(-theta)
            upper = values.astype(np.float64)+self.numerical_logit_guard
            potential = upper > threshold if selector.startswith('exactzero') else upper >= threshold
        else:
            raise ValueError("Unsupported explicit physical selector")
        self.potential_nodes = potential
        self.summed = np.pad(potential.astype(np.int64).cumsum(0).cumsum(1).cumsum(2), ((1, 0),)*3)

    def _coordinates(self, points):
        unit = (points-self.bounds[0])/(self.bounds[1]-self.bounds[0])
        if self.align_corners:
            return unit*(self.resolution-1)
        return unit*self.resolution-.5

    def cells_supported(self, lower, upper):
        lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
        if lo.ndim != 2 or lo.shape != hi.shape or lo.shape[1] != 3 or not np.isfinite([lo, hi]).all() or (hi < lo).any():
            raise ValueError("Finite ordered [N,3] cell bounds required")
        # The actual head returns p=0 outside its inclusive bounds. This is an
        # explicit selector definition, not a semantic/free-space inference.
        intersects = ((hi >= self.bounds[0]-self.coordinate_guard) & (lo <= self.bounds[1]+self.coordinate_guard)).all(1)
        first = np.floor(self._coordinates(np.maximum(lo-self.coordinate_guard, self.bounds[0]))).astype(np.int64)
        last = np.floor(self._coordinates(np.minimum(hi+self.coordinate_guard, self.bounds[1]))).astype(np.int64)+1
        first = np.clip(first, 0, self.resolution-1)
        last = np.clip(last, 0, self.resolution-1)+1
        # Entirely outside cells can have reversed ranges; they are masked out
        # but clamp for a well-formed summed-volume query in all cases.
        last = np.maximum(last, first+1)
        a, b, c = first.T
        x, y, z = last.T
        s = self.summed
        count = s[x, y, z]-s[a, y, z]-s[x, b, z]-s[x, y, c]+s[a, b, z]+s[a, y, c]+s[x, b, c]-s[a, b, c]
        return intersects & (count > 0)

    def map_cascades(self, cascade_aabbs, resolution, *, radius=0, chunk=131072):
        bounds = np.asarray(cascade_aabbs, dtype=np.float64)
        if bounds.ndim == 2 and bounds.shape[1] == 6:
            bounds = bounds.reshape(-1, 2, 3)
        if bounds.ndim != 3 or bounds.shape[1:] != (2, 3) or not len(bounds):
            raise ValueError("Actual cascade AABBs [L,6] or [L,2,3] required")
        r = np.asarray(resolution)
        if r.ndim == 0:
            r = np.repeat(r, 3)
        if r.shape != (3,) or not np.issubdtype(r.dtype, np.integer) or (r < 1).any() or chunk < 1:
            raise ValueError("Explicit integer native XYZ resolution and positive chunk required")
        result = []
        for box in bounds:
            box = _bounds(box)
            total = int(np.prod(r)); core = np.empty(total, dtype=bool)
            for start in range(0, total, chunk):
                ids = np.arange(start, min(start+chunk, total))
                ijk = np.stack([ids//(r[1]*r[2]), ids//r[2] % r[1], ids % r[2]], -1)
                lo = box[0]+ijk/r*(box[1]-box[0])
                hi = box[0]+(ijk+1)/r*(box[1]-box[0])
                core[start:start+len(ids)] = self.cells_supported(lo, hi)
            result.append(fresh_dilation(core.reshape(tuple(r)), radius))
        return np.stack(result)
