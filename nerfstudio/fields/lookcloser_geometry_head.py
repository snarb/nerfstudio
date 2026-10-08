"""Independent, initially zero signed-distance residual for surface bootstrapping."""
import torch
from torch import Tensor, nn


class LookCloserGeometryHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        # A copied radiance field may have very large density logits. Geometry
        # starts from its explicit shape, independently of that density output.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: Tensor, normalized_positions: Tensor) -> Tensor:
        # Signed distances and their gradients use FP32 even in an AMP trainer.
        with torch.autocast(device_type=features.device.type, enabled=False):
            inputs = torch.cat([features.float(), normalized_positions.float() - .5], -1)
            return self.network(inputs)
