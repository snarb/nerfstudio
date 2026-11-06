"""Experimental point lookup in source semantic cells, without traversal changes.

Not installed by any production model. This preserves closed-boundary tolerance
and outside-domain UNKNOWN admission from WholeCellAdmission.point_admitted.
The existing field AABB selector remains responsible for rejecting outside rays.
"""
import torch


class SourcePointAdmission(torch.nn.Module):
    def __init__(self, allowed, bounds, epsilon=2e-7):
        super().__init__()
        if allowed.dtype != torch.bool or allowed.ndim != 3 or min(allowed.shape) < 1:
            raise ValueError('Boolean XYZ source mask required')
        if not 0 <= epsilon < .01:
            raise ValueError('Explicit small coordinate epsilon required')
        bounds = torch.as_tensor(bounds, device=allowed.device, dtype=torch.float64)
        if bounds.shape != (2, 3) or not bool(torch.isfinite(bounds).all()) or not bool((bounds[1] > bounds[0]).all()):
            raise ValueError('Increasing finite source bounds required')
        resolution = torch.tensor(allowed.shape, device=allowed.device, dtype=torch.long)
        pitch = (bounds[1]-bounds[0])/resolution
        if epsilon*2 >= float(pitch.min()):
            raise ValueError('Tolerance may intersect at most two source cells per axis')
        self.epsilon = float(epsilon)
        self.register_buffer('mask', allowed, persistent=False)
        self.register_buffer('bounds', bounds, persistent=False)
        self.register_buffer('resolution', resolution, persistent=False)
        self.register_buffer('pitch', pitch, persistent=False)

    def forward(self, positions):
        shape = positions.shape[:-1]
        if positions.shape[-1] != 3:
            raise ValueError('XYZ positions required')
        # Source bounds are float64. Use them directly; no second world_to_box
        # and no quantization to the native traversal cell boundaries.
        p = positions.reshape(-1, 3).to(dtype=torch.float64)
        outside = ((p < self.bounds[0]-self.epsilon) | (p > self.bounds[1]+self.epsilon)).any(-1)
        low = (p-self.bounds[0]-self.epsilon)/self.pitch
        high = (p-self.bounds[0]+self.epsilon)/self.pitch
        first = torch.floor(torch.nextafter(low, torch.full_like(low, -torch.inf))).long()
        last = torch.floor(torch.nextafter(high, torch.full_like(high, torch.inf))).long()
        first = torch.maximum(torch.zeros_like(first), torch.minimum(first, self.resolution-1))
        last = torch.maximum(torch.zeros_like(last), torch.minimum(last, self.resolution-1))
        result = outside.clone()
        for x in (first[:, 0], last[:, 0]):
            for y in (first[:, 1], last[:, 1]):
                for z in (first[:, 2], last[:, 2]):
                    result |= self.mask[x, y, z]
        return result.reshape(shape)
