"""The cord controls preserve ordinary RGB rays and occupancy admissions."""
from types import SimpleNamespace

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.lookcloser_opacity import LookCloserOpacityModel, LookCloserOpacityModelConfig


def test_cord_occupancy_reopens_missing_cells_after_an_update():
    model=LookCloserOpacityModel.__new__(LookCloserOpacityModel)
    torch.nn.Module.__init__(model)
    model.config=LookCloserOpacityModelConfig(cord_occupancy_radius=.003)
    admission=torch.zeros(1,2,2,2,dtype=torch.bool)
    admission[0,0,0,1]=True
    model.cord_occupancy_admission=admission
    model.occupancy_grid=SimpleNamespace(binaries=torch.zeros_like(admission))
    model._admit_depth_occupancy()
    assert torch.equal(model.occupancy_grid.binaries,admission)
    model.occupancy_grid.binaries.zero_()  # emulate a subsequent EMA update
    model._admit_depth_occupancy()
    assert torch.equal(model.occupancy_grid.binaries,admission)


def test_cord_auxiliary_density_keeps_rgb_and_pixel_rng_unchanged():
    class Field(torch.nn.Module):
        def density_fn(self,positions):
            return torch.ones(len(positions),1,device=positions.device,requires_grad=True)

    model=LookCloserOpacityModel.__new__(LookCloserOpacityModel)
    torch.nn.Module.__init__(model)
    model.device_indicator_param=torch.nn.Parameter(torch.empty(0))
    model.current_train_step=30001
    model.field=Field()
    model.config=LookCloserOpacityModelConfig(cord_ray_regularization=.01,cord_rays_per_batch=2)
    model.cord_ray_origins=torch.zeros(3,3)
    model.cord_ray_directions=torch.tensor([[0.,0.,1.]]).expand(3,3)
    model.cord_ray_depths=torch.ones(3)
    ordinary=dict(rgb=torch.ones(2,3),packed_weights=torch.ones(2,1),packed_ray_indices=torch.arange(2),
                  packed_spacing_starts=torch.zeros(2,1),packed_spacing_ends=torch.ones(2,1))
    model._volume_outputs=lambda _:ordinary.copy()
    rays=RayBundle(origins=torch.zeros(2,3),directions=torch.ones(2,3),pixel_area=torch.ones(2,1))
    torch.manual_seed(99)
    state=torch.get_rng_state().clone()
    output=model.get_outputs(rays)
    assert torch.equal(state,torch.get_rng_state())
    assert torch.equal(output['rgb'],ordinary['rgb'])
    assert output['cord_band_sigma'].shape==(2,17)
    assert output['cord_free_sigma'].shape==(2,6)
