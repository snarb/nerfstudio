"""Explicit geometric buffer around original source cells; no grid mutation."""
import torch


class GeometricSourcePointAdmission(torch.nn.Module):
    def __init__(self, allowed, bounds, *, margin, epsilon=2e-7, base_class):
        super().__init__()
        if not torch.isfinite(torch.tensor([margin, epsilon], dtype=torch.float64)).all() or min(margin, epsilon) < 0:
            raise ValueError('Finite nonnegative geometric margin and numeric epsilon required')
        self.geometric_margin = float(margin)
        self.numerical_epsilon = float(epsilon)
        # Proven source-cell intersection, now with an explicit geometric cube.
        # Base class also enforces radius < half the minimum source cell pitch.
        self.lookup = base_class(allowed, bounds, epsilon=margin+epsilon)

    def forward(self, positions):
        p = positions.to(dtype=torch.float64)
        outside_unknown = ((p < self.lookup.bounds[0]-self.numerical_epsilon) |
                           (p > self.lookup.bounds[1]+self.numerical_epsilon)).any(-1)
        return self.lookup(positions) | outside_unknown
