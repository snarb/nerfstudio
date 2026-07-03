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

"""CPU unit tests for the mask-gated static/dynamic decomposition temporal NGP.

These tests run on CPU and never touch the GPU / tiny-cuda-nn. The tcnn-backed hash
encodings and MLPs are replaced with tiny torch stubs so the decomposition MATH (density
sum, blended color, dynamic-fraction accumulation), the loss gating, the method-config
registration and the datamanager mask-attach lookup are all exercised without CUDA.
"""

import types

import torch
from torch import nn

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.fields.temporal_ngp_field import TemporalNGPField

GEO = 4


def _make_decomp_field_stub() -> TemporalNGPField:
    """Build an H2D TemporalNGPField with torch stubs instead of tcnn modules."""
    field = TemporalNGPField.__new__(TemporalNGPField)
    nn.Module.__init__(field)
    field.register_buffer("aabb", torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    field.geo_feat_dim = GEO
    field.hypothesis = "H2D"
    field.average_init_density = 1.0
    field.appearance_embedding_dim = 0
    field.embedding_appearance = None
    field.use_average_appearance_embedding = True
    field.step = 0
    field.direction_encoding = SHEncoding(levels=4, implementation="torch")
    sh_dim = field.direction_encoding.get_out_dim()

    # Stub encodings: map xyz / xyzt -> feature vector.
    field.enc3 = nn.Linear(3, 8)
    field.enc4 = nn.Linear(4, 8)
    field.mlp_static = nn.Linear(8, 1 + GEO)
    field.mlp_dyn = nn.Linear(8, 1 + GEO)
    field.color_head_s = nn.Sequential(nn.Linear(sh_dim + GEO, 3), nn.Sigmoid())
    field.color_head_d = nn.Sequential(nn.Linear(sh_dim + GEO, 3), nn.Sigmoid())
    return field


def _ray_samples(n_rays=3, n_samples=5):
    origins = torch.rand(n_rays, n_samples, 3) * 0.2
    directions = torch.nn.functional.normalize(torch.rand(n_rays, n_samples, 3), dim=-1)
    starts = torch.rand(n_rays, n_samples, 1)
    ends = starts + 0.1
    frustums = Frustums(
        origins=origins,
        directions=directions,
        starts=starts,
        ends=ends,
        pixel_area=torch.ones(n_rays, n_samples, 1),
    )
    times = torch.rand(n_rays, n_samples, 1)
    return RaySamples(frustums=frustums, times=times)


def test_field_decomposition_forward():
    field = _make_decomp_field_stub()
    rs = _ray_samples()
    out = field.forward(rs)

    sigma = out[FieldHeadNames.DENSITY]
    sigma_d = out[FieldHeadNames.DYNAMIC_DENSITY]
    rgb = out[FieldHeadNames.RGB]

    assert torch.isfinite(sigma).all(), "total density must be finite"
    assert torch.isfinite(sigma_d).all(), "dynamic density must be finite"
    assert (sigma >= 0).all(), "total density nonneg"
    assert (sigma_d >= 0).all() and (sigma_d <= sigma + 1e-5).all(), "0 <= sigma_d <= sigma"
    assert FieldHeadNames.DYNAMIC_DENSITY in out, "dynamic_density present"
    assert rgb.shape[-1] == 3 and (rgb >= 0).all() and (rgb <= 1).all(), "rgb in [0,1]"

    # sigma == sigma_s + sigma_d (recompute sigma_s from the stash).
    sigma_s = field._sigma_s
    assert torch.allclose(sigma, sigma_s + sigma_d, atol=1e-5), "sigma == sigma_s + sigma_d"
    print("PASS test_field_decomposition_forward")


def test_loss_gating():
    from nerfstudio.models.temporal_decomp import TemporalDecompModel, TemporalDecompModelConfig

    model = TemporalDecompModel.__new__(TemporalDecompModel)
    nn.Module.__init__(model)
    model.config = TemporalDecompModelConfig(dynamic_sparsity_mult=0.5)

    # Stub parent get_loss_dict (avoids renderer / rgb_loss setup).
    def fake_parent(self, outputs, batch, metrics_dict=None):
        return {"rgb_loss": torch.tensor(1.0)}

    import nerfstudio.models.temporal_decomp as mod
    orig = mod.TemporalInstantNGPModel.get_loss_dict
    mod.TemporalInstantNGPModel.get_loss_dict = fake_parent
    try:
        dyn_acc = torch.tensor([[0.8], [0.6], [0.4], [0.2]])
        # Case 1: mixed labels -> only non-person rays (is_person=0) penalized.
        is_person = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        ld = model.get_loss_dict({"rgb_loss": None, "dynamic_accumulation": dyn_acc}, {"is_person": is_person})
        expected = 0.5 * ((0.6 + 0.2) / 4.0)  # mean over ALL rays of (1-is_person)*dyn_acc
        assert torch.isclose(ld["dynamic_sparsity_loss"], torch.tensor(expected)), ld["dynamic_sparsity_loss"]

        # Case 2: all person -> zero loss.
        ld2 = model.get_loss_dict(
            {"rgb_loss": None, "dynamic_accumulation": dyn_acc}, {"is_person": torch.ones(4, 1)}
        )
        assert torch.isclose(ld2["dynamic_sparsity_loss"], torch.tensor(0.0)), ld2["dynamic_sparsity_loss"]
    finally:
        mod.TemporalInstantNGPModel.get_loss_dict = orig
    print("PASS test_loss_gating")


def test_method_config_constructs():
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.data.datamanagers.temporal_decomp_datamanager import TemporalDecompDataManagerConfig
    from nerfstudio.models.temporal_decomp import TemporalDecompModelConfig
    from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

    cfg = method_configs["instant-ngp-time-decomp"]
    assert isinstance(cfg.pipeline, VanillaPipelineConfig)
    assert isinstance(cfg.pipeline.datamanager, TemporalDecompDataManagerConfig)
    assert isinstance(cfg.pipeline.model, TemporalDecompModelConfig)
    assert cfg.pipeline.datamanager.load_from_disk is True
    assert cfg.pipeline.model.decompose is True
    print(
        "PASS test_method_config_constructs:",
        type(cfg.pipeline).__name__,
        type(cfg.pipeline.datamanager).__name__,
        type(cfg.pipeline.model).__name__,
    )


def test_next_train_mask_attach():
    from nerfstudio.data.datamanagers.temporal_decomp_datamanager import TemporalDecompDataManager

    dm = TemporalDecompDataManager.__new__(TemporalDecompDataManager)
    dm.device = torch.device("cpu")
    # config stub with mask_downsample = 4.
    dm.config = types.SimpleNamespace(mask_downsample=4)
    # Stub mask: stem "frame_train_C0_000" -> [270,480] with a single person cell at (10, 20).
    mask = torch.zeros(270, 480)
    mask[10, 20] = 1.0
    dm._person_masks = {"frame_train_C0_000": mask}
    dm._idx_to_stem = {0: "frame_train_C0_000", 1: "frame_train_C0_001"}  # idx 1 has no mask

    # rays: (img,row,col). row//4, col//4 must hit (10,20) -> rows 40..43, cols 80..83.
    indices = torch.tensor(
        [
            [0, 40, 80],   # -> mask[10,20] = 1 (person)
            [0, 0, 0],     # -> mask[0,0] = 0 (non-person)
            [1, 40, 80],   # -> img 1 has no mask -> default 1 (unknown)
            [9, 0, 0],     # -> unknown img idx -> default 1
        ]
    )
    is_person = dm._lookup_is_person(indices)
    assert is_person.shape == (4, 1)
    assert is_person[0, 0] == 1.0, "person cell"
    assert is_person[1, 0] == 0.0, "background cell"
    assert is_person[2, 0] == 1.0, "missing mask -> unknown -> 1"
    assert is_person[3, 0] == 1.0, "missing image -> unknown -> 1"
    print("PASS test_next_train_mask_attach:", is_person.flatten().tolist())


if __name__ == "__main__":
    test_field_decomposition_forward()
    test_loss_gating()
    test_method_config_constructs()
    test_next_train_mask_attach()
    print("ALL TESTS PASSED")
