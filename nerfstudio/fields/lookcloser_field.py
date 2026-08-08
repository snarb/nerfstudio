"""
Frequency-Aware Field for LookCloser (FA-NeRF).
Implements the modified hash encoding with frequency-dependent feature re-weighting.
"""

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field
from nerfstudio.model_components.lookcloser_grid import FrequencyGridManager
from nerfstudio.utils.external import TCNN_EXISTS, tcnn, tcnn_import_exception
from nerfstudio.utils.hdr import activate_hdr_rgb

TCNNNetworkJITScope = Literal["both", "geometry", "color"]


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
            log2_hashmap_size: int = 23,
            features_per_level: int = 2,
            hidden_dim: int = 64,
            geo_num_layers: int = 1,
            color_num_layers: int = 2,
            sh_degree: int = 4,
            enable_feature_reweighting: bool = True,
            feature_reweighting_strength: float = 1.0,
            appearance_embedding_dim: int = 0,
            num_images: int = 0,
            tcnn_network_jit: bool = False,
            tcnn_network_jit_scope: TCNNNetworkJITScope = "both",
            rgb_output_parameterization: Literal["sigmoid", "linear_softplus", "pq_code"] = "sigmoid",
            hdr_linear_scale: float = 1.0,
            hdr_initial_radiance: float = 0.5,
            pq_nits_per_scene_unit: float = 100.0,
            pq_black_nits: float = 0.005,
            pq_peak_nits: float = 10_000.0,
            hdr_softplus_beta: float = 1.0,
            pq_code_temperature: float = 1.0,
            spatial_distortion=None,
    ) -> None:
        super().__init__()
        if not TCNN_EXISTS:
            raise ImportError(
                "LookCloserField requires tinycudann. Install the CUDA extension or avoid importing this field."
            ) from tcnn_import_exception

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.enable_feature_reweighting = enable_feature_reweighting
        self.feature_reweighting_strength = float(feature_reweighting_strength)
        self.freq_grid = freq_grid
        self.spatial_distortion = spatial_distortion
        self.appearance_embedding_dim = int(appearance_embedding_dim)
        self.num_images = int(num_images)
        self.rgb_output_parameterization = rgb_output_parameterization
        self.hdr_linear_scale = float(hdr_linear_scale)
        self.hdr_initial_radiance = float(hdr_initial_radiance)
        self.pq_nits_per_scene_unit = float(pq_nits_per_scene_unit)
        self.pq_black_nits = float(pq_black_nits)
        self.pq_peak_nits = float(pq_peak_nits)
        self.hdr_softplus_beta = float(hdr_softplus_beta)
        self.pq_code_temperature = float(pq_code_temperature)
        if self.appearance_embedding_dim > 0:
            if self.num_images <= 0:
                raise ValueError("num_images must be > 0 when appearance embeddings are enabled.")
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.embedding_appearance = None

        # Eq. 6 Parameters
        self.l_min = 0.0
        self.l_max = float(num_levels - 1)
        self.register_buffer(
            "_feature_weight_lut",
            torch.empty((0, num_levels * features_per_level), dtype=torch.float32),
            persistent=False,
        )
        self._feature_weight_lut_strength: Optional[float] = None

        # 1. Hash Encoding (Instant-NGP style)
        # Calculate per-level scale 'b'
        per_level_scale = np_exp((np_log(max_res) - np_log(min_res)) / (num_levels - 1))

        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": per_level_scale,
            },
        )
        self.n_features = num_levels * features_per_level

        # 2. Geometry MLP (Density Decoder)
        # Input: 32 (features) -> Output: 16 (1 density + 15 geometry features)
        self.mlp_geo = tcnn.Network(
            n_input_dims=self.n_features,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": geo_num_layers,
            },
        )

        # 3. Color MLP (Appearance Decoder)
        # Input: 15 (geo features) + 16 (SH encoding for view dir) -> Output: 3 (RGB)
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": sh_degree,
            },
        )
        direction_dim = sh_degree * sh_degree

        self.mlp_color = tcnn.Network(
            n_input_dims=self.geo_feat_dim + direction_dim + self.appearance_embedding_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid" if rgb_output_parameterization == "sigmoid" else "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": color_num_layers,
            },
        )
        self.set_tcnn_network_jit(tcnn_network_jit, scope=tcnn_network_jit_scope)

    def _activate_rgb(self, raw_rgb: Tensor) -> Tensor:
        if self.rgb_output_parameterization == "sigmoid":
            # The default tiny-cuda-nn head already applied sigmoid.  Keeping
            # this branch as identity preserves historical checkpoints exactly.
            return raw_rgb
        return activate_hdr_rgb(
            raw_rgb,
            parameterization=self.rgb_output_parameterization,
            linear_scale=self.hdr_linear_scale,
            initial_radiance=self.hdr_initial_radiance,
            nits_per_scene_unit=self.pq_nits_per_scene_unit,
            black_nits=self.pq_black_nits,
            peak_nits=self.pq_peak_nits,
            softplus_beta=self.hdr_softplus_beta,
            pq_code_temperature=self.pq_code_temperature,
        )

    def _tcnn_networks_for_scope(self, scope: TCNNNetworkJITScope) -> Tuple[nn.Module, ...]:
        """Return the TCNN networks selected by a validated JIT scope."""

        if scope == "both":
            return self.mlp_geo, self.mlp_color
        if scope == "geometry":
            return (self.mlp_geo,)
        if scope == "color":
            return (self.mlp_color,)
        raise ValueError(f"Unsupported TCNN network JIT scope: {scope!r}")

    def set_tcnn_network_jit(self, enabled: bool, scope: TCNNNetworkJITScope = "both") -> None:
        """Toggle JIT on the selected TCNN network(s) without reconstruction."""

        networks = self._tcnn_networks_for_scope(scope)
        enabled = bool(enabled)
        if all(bool(network.jit_fusion) == enabled for network in networks):
            return
        if enabled and not tcnn.supports_jit_fusion():
            raise RuntimeError("tiny-cuda-nn JIT fusion was requested but is unsupported by this binding/GPU")
        for network in networks:
            network.jit_fusion = enabled

    def get_tcnn_network_jit(self, scope: TCNNNetworkJITScope = "both") -> bool:
        """Return selected JIT state, requiring all networks in the scope to agree."""

        states = tuple(bool(network.jit_fusion) for network in self._tcnn_networks_for_scope(scope))
        if any(state != states[0] for state in states[1:]):
            raise RuntimeError(f"LookCloser TCNN network JIT states disagree within scope {scope!r}")
        return states[0]

    def _normalize_positions(self, positions: Tensor) -> Tuple[Tensor, Tensor, Tuple[int, ...], Tensor]:
        """Normalizes world positions to the hash-grid unit cube."""
        world_positions_flat = positions.reshape(-1, 3)
        if self.spatial_distortion is not None:
            normalized = self.spatial_distortion(positions)
            normalized = (normalized + 2.0) / 4.0
        else:
            normalized = SceneBox.get_normalized_positions(positions, self.aabb)

        selector = ((normalized > 0.0) & (normalized < 1.0)).all(dim=-1)
        normalized = normalized * selector[..., None]
        prefix_shape = normalized.shape[:-1]
        return normalized.view(-1, 3), selector, prefix_shape, world_positions_flat

    def _encode_with_optional_reweighting(
        self,
        normalized_positions_flat: Tensor,
        query_positions_flat: Tensor,
        l_grid: Optional[Tensor] = None,
    ) -> Tensor:
        features = self.encoding(normalized_positions_flat)
        if not self.enable_feature_reweighting:
            return features
        queried_discrete_grid = l_grid is None and self.freq_grid.enabled
        if l_grid is None:
            l_grid = self.freq_grid.query(query_positions_flat)
        if queried_discrete_grid:
            weights = self._get_discrete_feature_weights(l_grid)
        else:
            # Preserve the analytical path for explicit/fractional levels and for
            # the disabled-grid fallback, whose configured level may be fractional.
            weights = self.get_weights(l_grid, batch_size=normalized_positions_flat.shape[0])
        return features * weights

    def _get_discrete_feature_weights(self, l_grid: Tensor) -> Tensor:
        """Look up exact analytical weights for the grid's integer levels."""
        strength = float(self.feature_reweighting_strength)
        if (
            self._feature_weight_lut.numel() == 0
            or self._feature_weight_lut.device != l_grid.device
            or self._feature_weight_lut.dtype != l_grid.dtype
            or self._feature_weight_lut_strength != strength
        ):
            levels = torch.arange(
                self.num_levels, device=l_grid.device, dtype=l_grid.dtype
            ).unsqueeze(-1)
            self._feature_weight_lut = self.get_weights(levels, batch_size=self.num_levels)
            self._feature_weight_lut_strength = strength
        indices = l_grid.reshape(-1).to(torch.long).clamp_(0, self.num_levels - 1)
        return self._feature_weight_lut[indices]

    def query_points(
        self,
        positions: Tensor,
        directions: Tensor,
        l_grid: Optional[Tensor] = None,
        camera_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Queries density and RGB for world-space points used by custom ray marchers."""
        positions_flat, selector, _, query_positions_flat = self._normalize_positions(positions)
        if l_grid is not None:
            l_grid = l_grid.reshape(-1, 1)
        features = self._encode_with_optional_reweighting(
            positions_flat,
            query_positions_flat=query_positions_flat,
            l_grid=l_grid,
        )
        h = self.mlp_geo(features)
        density = F.softplus(h[..., 0:1] + 1.0)
        density = density * selector.reshape(-1, 1)
        geo_feat = h[..., 1:]

        d_encoded = self.direction_encoding(directions.reshape(-1, 3))
        color_inputs = [geo_feat, d_encoded]
        if self.embedding_appearance is not None:
            if self.training and camera_indices is not None:
                embedded_appearance = self.embedding_appearance(camera_indices.reshape(-1).long())
            else:
                embedded_appearance = torch.zeros(
                    (geo_feat.shape[0], self.appearance_embedding_dim),
                    device=geo_feat.device,
                    dtype=geo_feat.dtype,
                )
            color_inputs.append(embedded_appearance)
        rgb = self._activate_rgb(self.mlp_color(torch.cat(color_inputs, dim=-1)))
        return density, rgb

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
        if self.feature_reweighting_strength != 1.0:
            final_weights = 1.0 + self.feature_reweighting_strength * (final_weights - 1.0)

        return final_weights.repeat_interleave(self.features_per_level, dim=1)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """
        Computes density and geometry features with frequency-aware re-weighting.
        """
        positions = ray_samples.frustums.get_positions()
        positions_flat, selector, prefix_shape, query_positions_flat = self._normalize_positions(positions)

        weighted_features = self._encode_with_optional_reweighting(
            positions_flat,
            query_positions_flat=query_positions_flat,
        )

        # 4. Geometry Decoding
        h = self.mlp_geo(weighted_features)

        # Split output
        density_before_activation = h[..., 0:1]
        geo_feat = h[..., 1:]

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

        color_inputs = [geo_feat_flat, d_encoded]
        if self.embedding_appearance is not None:
            camera_indices = ray_samples.camera_indices
            if self.training and camera_indices is not None:
                embedded_appearance = self.embedding_appearance(camera_indices.reshape(-1).long())
            else:
                embedded_appearance = torch.zeros(
                    (geo_feat_flat.shape[0], self.appearance_embedding_dim),
                    device=geo_feat_flat.device,
                    dtype=geo_feat_flat.dtype,
                )
            color_inputs.append(embedded_appearance)

        # Concatenate and Decode
        color_input = torch.cat(color_inputs, dim=-1)
        rgb = self._activate_rgb(self.mlp_color(color_input))

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
