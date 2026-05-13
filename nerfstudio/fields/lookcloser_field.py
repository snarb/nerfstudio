"""
Frequency-Aware Field for LookCloser (FA-NeRF).
Implements the modified hash encoding with frequency-dependent feature re-weighting.
"""

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import tinycudann as tcnn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field
from nerfstudio.model_components.lookcloser_grid import FrequencyGridManager


class LookCloserField(Field):
    """
    LookCloser Field that adapts feature weights based on a frequency grid.

    Args:
        aabb: Parameters of scene aabb bounds.
        freq_grid: The FrequencyGridManager instance to query levels from.
        geo_feat_dim: Dimension of the geometry feature output.
        num_levels: Number of hash grid levels.
        max_res: Maximum resolution of the hash grid.
        log2_hashmap_size: Size of the hash map (2^N).
        spatial_distortion: Spatial distortion to apply to the scene.
    """

    def __init__(
            self,
            aabb: Tensor,
            freq_grid: FrequencyGridManager,
            geo_feat_dim: int = 15,
            num_levels: int = 16,
            min_res: int = 16,
            max_res: int = 2048,
            log2_hashmap_size: int = 19,
            spatial_distortion=None,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim
        self.num_levels = num_levels
        self.freq_grid = freq_grid
        self.spatial_distortion = spatial_distortion

        # Eq. 6 Parameters
        self.l_min = 0.0
        self.l_max = float(num_levels - 1)

        # 1. Hash Encoding (Instant-NGP style)
        # Calculate per-level scale 'b'
        per_level_scale = np_exp((np_log(max_res) - np_log(min_res)) / (num_levels - 1))

        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": per_level_scale,
            },
        )
        self.n_features = num_levels * 2

        # 2. Geometry MLP (Density Decoder)
        # Input: 32 (features) -> Output: 16 (1 density + 15 geometry features)
        self.mlp_geo = tcnn.Network(
            n_input_dims=self.n_features,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        # 3. Color MLP (Appearance Decoder)
        # Input: 15 (geo features) + 16 (SH encoding for view dir) -> Output: 3 (RGB)
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.mlp_color = tcnn.Network(
            n_input_dims=self.geo_feat_dim + 16,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def get_weights(self, l_grid: Tensor, batch_size: int) -> Tensor:
        """
        Calculates weights based on the grid frequency 'l' acting as a threshold.
        If feature_level <= l_grid: weight = 1.0
        If feature_level >  l_grid: weight = w_curve(l_grid) [Eq. 6]

        Args:
            l_grid: (B, 1) Float tensor of max frequency levels (the grid values).
            batch_size: Number of samples.

        Returns:
            weights: (B, num_levels * 2) Tensor of weights for the flattened feature vector.
        """
        device = l_grid.device

        # 1. Feature Levels (0..15)
        # Shape: (B, 16)
        feature_levels = (
            torch.arange(self.num_levels, device=device)
            .expand(batch_size, self.num_levels)
            .float()
        )
        l_grid_expanded = l_grid.expand(batch_size, self.num_levels)

        # 2. Calculate Damping Factor w_l (Eq. 6)
        range_sq = (self.l_max - self.l_min) ** 2

        # Denominator: (l_max - l_grid + 1)^2
        denom = (self.l_max - l_grid_expanded + 1) ** 2
        denom_clamped = torch.clamp(denom, min=1.0, max=range_sq)

        # Argument: sqrt( range^2 / denom )
        erf_arg = torch.sqrt(range_sq / denom_clamped)

        # Erf Approximation
        # erf(x) ≈ sign(x) * sqrt(1 - exp(-4/pi * x^2))
        w_factor = torch.sqrt(
            1.0 - torch.exp(-(4.0 / torch.pi) * (erf_arg ** 2))
        )

        # 3. Apply One-Sided Masking
        # Keep low-freq features as is (1.0)
        mask_keep = (feature_levels <= l_grid_expanded).float()
        # Dampen high-freq features
        mask_decay = (feature_levels > l_grid_expanded).float()

        final_weights = (mask_keep * 1.0) + (mask_decay * w_factor)

        # Expand to match feature dimensions (2 features per level)
        # Shape becomes (B, 32)
        return final_weights.repeat_interleave(2, dim=1)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """
        Computes density and geometry features with frequency-aware re-weighting.
        """
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(
                ray_samples.frustums.get_positions(), self.aabb
            )

        # Clip positions to [0, 1] for tcnn
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]

        prefix_shape = positions.shape[:-1]
        positions_flat = positions.view(-1, 3)

        # 1. Query Frequency Grid
        # Get the max frequency level allowed at these positions
        l_grid = self.freq_grid.query(positions_flat)  # (N, 1)

        # 2. Raw Hash Encoding
        features = self.encoding(positions_flat)  # (N, 32)

        # 3. Feature Re-weighting
        weights = self.get_weights(l_grid, batch_size=positions_flat.shape[0])
        weighted_features = features * weights

        # 4. Geometry Decoding
        h = self.mlp_geo(weighted_features)

        # Split output
        density_before_activation = h[..., 0:1]
        geo_feat = h[..., 1:]

        # Rectify density
        density = F.softplus(density_before_activation + 1.0)

        # Reshape back to ray samples structure
        density = density.view(*prefix_shape, 1)
        geo_feat = geo_feat.view(*prefix_shape, self.geo_feat_dim)

        # Apply valid mask
        density = density * selector[..., None]

        return density, geo_feat

    def get_outputs(
            self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        """
        Computes color using the geometry features (density_embedding) and view direction.
        """
        assert density_embedding is not None

        # Prepare View Directions
        directions = ray_samples.frustums.directions
        prefix_shape = directions.shape[:-1]
        directions_flat = directions.reshape(-1, 3)

        d_encoded = self.direction_encoding(directions_flat)

        # Flatten density embedding
        geo_feat_flat = density_embedding.reshape(-1, self.geo_feat_dim)

        # Concatenate and Decode
        color_input = torch.cat([geo_feat_flat, d_encoded], dim=-1)
        rgb = self.mlp_color(color_input)

        # Reshape
        rgb = rgb.view(*prefix_shape, 3)

        return {FieldHeadNames.RGB: rgb}


# Numpy helpers for the init math (avoids importing numpy just for two calls if we want to stay pure torch-ish,
# but python math/numpy is fine here)
def np_exp(x):
    import numpy as np
    return np.exp(x)


def np_log(x):
    import numpy as np
    return np.log(x)