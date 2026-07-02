# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Temporal field for bounded Instant-NGP.

Implements ``F(x, y, z, t) -> density, color`` (hypothesis H1): time is added as a
fourth coordinate of a multiresolution hash grid. This is a pure 4D-hash field; it
contains no deformation field, canonical space, or temporal regularization.
"""

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.utils.external import TCNN_EXISTS, tcnn


class TemporalHashMLP(nn.Module):
    """A multiresolution hash grid over ``in_dim`` coordinates followed by an MLP.

    This mirrors :class:`nerfstudio.field_components.mlp.MLPWithHashEncoding` but allows
    ``in_dim != 3`` (e.g. 4 for ``xyzt``). Only the tiny-cuda-nn backend is supported for
    ``in_dim != 3``; a torch fallback would require a 4D-aware interpolation kernel.
    """

    def __init__(
        self,
        in_dim: int = 4,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 512,
        log2_hashmap_size: int = 20,
        features_per_level: int = 2,
        num_layers: int = 2,
        layer_width: int = 64,
        out_dim: int = 16,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1.0

        if implementation == "torch" or (implementation == "tcnn" and not TCNN_EXISTS):
            raise NotImplementedError(
                "Temporal 4D hash grid currently supports only tiny-cuda-nn. "
                "Install tiny-cuda-nn and use implementation='tcnn'."
            )

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=in_dim,
            n_output_dims=out_dim,
            encoding_config=HashEncoding.get_tcnn_encoding_config(
                num_levels=num_levels,
                features_per_level=features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                min_res=min_res,
                growth_factor=growth_factor,
            ),
            network_config=MLP.get_tcnn_network_config(
                activation=activation,
                out_activation=out_activation,
                layer_width=layer_width,
                num_layers=num_layers,
            ),
        )

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        return self.model(in_tensor)


class TemporalNGPField(Field):
    """Bounded Instant-NGP field with a 4D (xyzt) hash grid.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers for the density/base MLP
        hidden_dim: dimension of hidden layers for the density/base MLP
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        features_per_level: number of features per level for the hashgrid
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
        rgb_output_activation: output activation for the RGB MLP
        average_init_density: average initial density scaling for the density activation
        implementation: tcnn or torch (only tcnn supported for the 4D grid)
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 512,
        log2_hashmap_size: int = 20,
        features_per_level: int = 2,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        rgb_output_activation: Literal["sigmoid", "none"] = "sigmoid",
        average_init_density: float = 1.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        hypothesis: Literal["H1", "H2", "H3", "H2D"] = "H1",
        static_num_levels: int = 16,
        static_max_res: int = 512,
        static_log2_hashmap_size: int = 19,
        num_images: int = 1,
        appearance_embedding_dim: int = 0,
        use_average_appearance_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.embedding_appearance = (
            Embedding(num_images, appearance_embedding_dim) if appearance_embedding_dim > 0 else None
        )
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))
        self.base_res = base_res
        self.average_init_density = average_init_density
        self.hypothesis = hypothesis
        self.step = 0

        # LookCloser feature-reweighting (FR): off by default (byte-identical behaviour). When enabled
        # by the model (and a frequency grid is available), ``freq_level_fn`` returns a per-point grid
        # frequency level used to dampen the high hash-grid levels of enc3/enc4 (Eq. 6).
        self.enable_feature_reweighting: bool = False
        self.feature_reweighting_strength: float = 1.0
        self.freq_level_fn = None
        self._fr_enc3_levels: int = int(static_num_levels)
        self._fr_enc4_levels: int = int(num_levels)
        self._fr_features_per_level: int = int(features_per_level)

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        if hypothesis == "H1":
            # Pure 4D hash grid: [x, y, z, t] -> density (1) + geo features (geo_feat_dim).
            self.mlp_base = TemporalHashMLP(
                in_dim=4,
                num_levels=num_levels,
                min_res=base_res,
                max_res=max_res,
                log2_hashmap_size=log2_hashmap_size,
                features_per_level=features_per_level,
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1 + self.geo_feat_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
        elif hypothesis == "H2D":
            # Mask-gated static/dynamic DECOMPOSITION.
            # Two SEPARATE sub-networks (separate hash encodings + separate MLPs):
            #   static : enc3 (3D hash) -> mlp_static -> sigma_s_raw(1) + geo_s(geo_feat_dim)
            #   dynamic: enc4 (4D hash on [xyz, t]) -> mlp_dyn -> sigma_d_raw(1) + geo_d(geo_feat_dim)
            # The two branches have INDEPENDENT color heads; the per-sample color is the
            # density-weighted blend of the two branch colors (see get_outputs).
            if implementation != "tcnn" or not TCNN_EXISTS:
                raise NotImplementedError("Temporal hypothesis H2D currently supports only tiny-cuda-nn.")
            self.enc3 = HashEncoding(
                num_levels=static_num_levels,
                min_res=base_res,
                max_res=static_max_res,
                log2_hashmap_size=static_log2_hashmap_size,
                features_per_level=features_per_level,
                implementation="tcnn",
                in_dim=3,
            )
            self.enc4 = HashEncoding(
                num_levels=num_levels,
                min_res=base_res,
                max_res=max_res,
                log2_hashmap_size=log2_hashmap_size,
                features_per_level=features_per_level,
                implementation="tcnn",
                in_dim=4,
            )
            self.mlp_static = MLP(
                in_dim=self.enc3.get_out_dim(),
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1 + self.geo_feat_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.mlp_dyn = MLP(
                in_dim=self.enc4.get_out_dim(),
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1 + self.geo_feat_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
        else:
            if implementation != "tcnn" or not TCNN_EXISTS:
                raise NotImplementedError(
                    "Temporal hypotheses H2/H3 currently support only tiny-cuda-nn."
                )
            # Static 3D hash branch (shared by H2 and H3).
            self.enc3 = HashEncoding(
                num_levels=static_num_levels,
                min_res=base_res,
                max_res=static_max_res,
                log2_hashmap_size=static_log2_hashmap_size,
                features_per_level=features_per_level,
                implementation="tcnn",
                in_dim=3,
            )
            feat_dim = self.enc3.get_out_dim()
            if hypothesis == "H2":
                # Additional 4D hash branch for dynamic content.
                self.enc4 = HashEncoding(
                    num_levels=num_levels,
                    min_res=base_res,
                    max_res=max_res,
                    log2_hashmap_size=log2_hashmap_size,
                    features_per_level=features_per_level,
                    implementation="tcnn",
                    in_dim=4,
                )
                feat_dim += self.enc4.get_out_dim()
            else:  # H3: weak temporal conditioning via scalar t appended to 3D features.
                feat_dim += 1
            self.mlp_base = MLP(
                in_dim=feat_dim,
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1 + self.geo_feat_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )

        # Color head(s). The decomposition variant uses TWO heads (static + dynamic);
        # all other hypotheses use the single shared head ``mlp_head``.
        color_head_in_dim = self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim
        if hypothesis == "H2D":
            self.color_head_s = MLP(
                in_dim=color_head_in_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid() if rgb_output_activation == "sigmoid" else None,
                implementation=implementation,
            )
            self.color_head_d = MLP(
                in_dim=color_head_in_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid() if rgb_output_activation == "sigmoid" else None,
                implementation=implementation,
            )

        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid() if rgb_output_activation == "sigmoid" else None,
            implementation=implementation,
        )

    def _get_times(self, ray_samples: RaySamples, positions: Tensor) -> Tensor:
        """Returns per-sample time broadcast to ``positions`` shape, clamped to [0, 1]."""
        times = ray_samples.times
        if times is None:
            # Fall back to t=0 (e.g. single-frame / static evaluation).
            times = torch.zeros_like(positions[..., :1])
        else:
            times = times.to(positions)
            # broadcast [..., 1] (or [..., 1, 1]) to match the sampled position shape.
            times = times.expand(*positions.shape[:-1], 1)
        return torch.clamp(times, 0.0, 1.0)

    def _reweight(self, features: Tensor, levels: Tensor, enc_num_levels: int) -> Tensor:
        """LookCloser Eq. 6 feature reweighting: keep hash levels <= grid level at weight 1, dampen the
        higher levels by w_l. ``features`` is (N, enc_num_levels * features_per_level); ``levels`` is
        (N,) the grid frequency level per point."""
        n = features.shape[0]
        device = features.device
        l_grid = levels.reshape(n, 1).to(device=device, dtype=torch.float32)
        feat_levels = torch.arange(enc_num_levels, device=device).float().expand(n, enc_num_levels)
        l_max = float(enc_num_levels - 1)
        range_sq = max((l_max - 0.0) ** 2, 1.0)
        denom = (l_max - l_grid.expand(n, enc_num_levels) + 1.0) ** 2
        denom = denom.clamp(min=1.0, max=range_sq)
        erf_arg_sq = range_sq / denom
        w_factor = torch.sqrt(torch.clamp(1.0 - torch.exp(-(4.0 / torch.pi) * erf_arg_sq), min=0.0))
        mask_keep = (feat_levels <= l_grid).float()
        mask_decay = (feat_levels > l_grid).float()
        weights = mask_keep + mask_decay * w_factor
        if self.feature_reweighting_strength != 1.0:
            weights = 1.0 + self.feature_reweighting_strength * (weights - 1.0)
        weights = weights.repeat_interleave(self._fr_features_per_level, dim=1)
        return features * weights.to(features.dtype)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities from the 4D (xyzt) hash grid."""
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets spatial inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]

        assert positions.numel() > 0, "positions is empty."

        times = self._get_times(ray_samples, positions)
        # Development asserts: ensure time plumbing is correct and normalized.
        assert torch.all(times >= 0.0) and torch.all(times <= 1.0), "ray times must lie in [0, 1]."

        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        pos_flat = positions.view(-1, 3)
        t_flat = times.reshape(-1, 1)
        assert pos_flat.numel() > 0, "positions_flat is empty."

        if self.hypothesis == "H2D":
            return self._get_density_decompose(ray_samples, pos_flat, t_flat, positions, selector)

        if self.hypothesis == "H1":
            xt_flat = torch.cat([pos_flat, t_flat], dim=-1)  # [N, 4]
            feats = self.mlp_base(xt_flat)
        elif self.hypothesis == "H2":
            f3 = self.enc3(pos_flat)
            f4 = self.enc4(torch.cat([pos_flat, t_flat], dim=-1))
            if self.enable_feature_reweighting and self.freq_level_fn is not None:
                # Frequency grid is queried at WORLD positions (its world_to_grid uses the aabb buffers).
                world_pos = ray_samples.frustums.get_positions().reshape(-1, 3)
                levels = self.freq_level_fn(world_pos).reshape(-1)
                f3 = self._reweight(f3, levels, self._fr_enc3_levels)
                f4 = self._reweight(f4, levels, self._fr_enc4_levels)
            feats = self.mlp_base(torch.cat([f3, f4], dim=-1))
        else:  # H3: 3D hash features + scalar time
            f3 = self.enc3(pos_flat)
            feats = self.mlp_base(torch.cat([f3, t_flat.to(f3.dtype)], dim=-1))

        h = feats.view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Exponential rectification, same convention as NerfactoField / Instant-NGP.
        density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def _get_density_decompose(self, ray_samples, pos_flat, t_flat, positions, selector):
        """H2D decomposition density.

        Computes the static and dynamic branch densities separately and returns the TOTAL
        density (sigma_s + sigma_d). Per-sample dynamic density and both geo embeddings are
        stashed so ``get_outputs`` can blend colors and expose the dynamic density. The
        returned ``density_embedding`` is the concatenation [geo_s, geo_d] (2 * geo_feat_dim).
        """
        f3 = self.enc3(pos_flat)
        f4 = self.enc4(torch.cat([pos_flat, t_flat], dim=-1))
        out_s = self.mlp_static(f3).view(*ray_samples.frustums.shape, -1)
        out_d = self.mlp_dyn(f4).view(*ray_samples.frustums.shape, -1)

        sigma_s_raw, geo_s = torch.split(out_s, [1, self.geo_feat_dim], dim=-1)
        sigma_d_raw, geo_d = torch.split(out_d, [1, self.geo_feat_dim], dim=-1)

        # Same trunc_exp / average_init_density convention as the single-branch path.
        sel = selector[..., None]
        sigma_s = self.average_init_density * trunc_exp(sigma_s_raw.to(positions)) * sel
        sigma_d = self.average_init_density * trunc_exp(sigma_d_raw.to(positions)) * sel
        density = sigma_s + sigma_d

        # Stash for get_outputs (per-sample, same leading shape as ray_samples).
        self._sigma_s = sigma_s
        self._sigma_d = sigma_d
        self._geo_s = geo_s
        self._geo_d = geo_d
        self._density_before_activation = sigma_s_raw + sigma_d_raw

        # Combined embedding so the base Field.forward plumbing has a valid tensor to pass.
        density_embedding = torch.cat([geo_s, geo_d], dim=-1)
        return density, density_embedding

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # Optional per-image appearance embedding (matches instant-ngp-bounded / NerfactoField).
        # Time already influences the geometry features via the 4D grid, so the color head needs
        # only the direction encoding + geo features (+ appearance embedding when enabled).
        embedded_appearance = None
        if self.embedding_appearance is not None:
            if self.training:
                camera_indices = ray_samples.camera_indices.squeeze()  # type: ignore
                embedded_appearance = self.embedding_appearance(camera_indices)
            elif self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*outputs_shape, self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*outputs_shape, self.appearance_embedding_dim), device=directions.device
                )

        if self.hypothesis == "H2D":
            # Two-branch color: blend by per-sample density contribution.
            geo_s = self._geo_s.reshape(-1, self.geo_feat_dim)
            geo_d = self._geo_d.reshape(-1, self.geo_feat_dim)

            def _head(head, geo):
                cat = [d, geo]
                if embedded_appearance is not None:
                    cat.append(embedded_appearance.view(-1, self.appearance_embedding_dim))
                return head(torch.cat(cat, dim=-1)).view(*outputs_shape, -1).to(directions)

            c_s = _head(self.color_head_s, geo_s)
            c_d = _head(self.color_head_d, geo_d)

            sigma_s = self._sigma_s
            sigma_d = self._sigma_d
            eps = 1e-6
            rgb = (sigma_s * c_s + sigma_d * c_d) / (sigma_s + sigma_d + eps)

            outputs[FieldHeadNames.RGB] = rgb
            # Per-sample dynamic density so the model can render a dynamic accumulation.
            outputs[FieldHeadNames.DYNAMIC_DENSITY] = sigma_d
            return outputs

        cat_list = [d, density_embedding.view(-1, self.geo_feat_dim)]
        if embedded_appearance is not None:
            cat_list.append(embedded_appearance.view(-1, self.appearance_embedding_dim))
        h = torch.cat(cat_list, dim=-1)
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})
        return outputs

    def density_fn(
        self, positions: Float[Tensor, "*bs 3"], times: Optional[Float[Tensor, "*bs 1"]] = None
    ) -> Float[Tensor, "*bs 1"]:
        """Returns only the density, conditioned on time. Used by the occupancy grid.

        Unlike the base implementation, this forwards ``times`` into the field so that the
        occupancy estimator can evaluate density at specific timestamps.
        """
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=times,
        )
        density, _ = self.get_density(ray_samples)
        return density
