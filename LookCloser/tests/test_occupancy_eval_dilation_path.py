"""Exercise eval dilation through the constant-step traversal entry point."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.lookcloser import LookCloserModel


class TraversalCaptured(Exception):
    """Stop after sampling: these tests inspect traversal, not neural rendering."""


def _model(device, radius, *, native=False):
    model = LookCloserModel.__new__(LookCloserModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        occupancy_eval_dilation_radius=radius,
        render_step_size=0.01,
        near_plane=0.01,
        far_plane=4.0,
        transmittance_threshold=1e-4,
        cone_angle=0.0,
        alpha_thre=0.0,
    )
    original = torch.zeros((1, 8, 8, 8), dtype=torch.bool, device=device)
    original[0, 4, 4, 4] = True
    if native:
        import nerfacc

        grid = nerfacc.OccGridEstimator(
            roi_aabb=torch.tensor([-1., -1., -1., 1., 1., 1.], device=device),
            resolution=8,
            levels=1,
        ).to(device)
        grid.binaries.copy_(original)
        sample = grid.sampling
    else:
        grid = SimpleNamespace(binaries=original.clone())

        def sample(**_kwargs):
            # A ray parallel to X passes through the neighboring Y cell.
            count = int(grid.binaries[0, :, 5, 4].sum())
            return (
                torch.zeros(count, dtype=torch.long, device=device),
                torch.zeros(count, device=device),
                torch.ones(count, device=device),
            )

    model.occupancy_grid = grid
    model._eval_occupancy_backup = None
    captured = []

    def capture(**kwargs):
        indices, starts, ends = sample(**kwargs)
        captured.append((grid.binaries.clone(), indices.clone(), starts.clone(), ends.clone()))
        raise TraversalCaptured

    grid.sampling = capture
    ray = RayBundle(
        origins=torch.tensor([[-2., .375, .125]], device=device),
        directions=torch.tensor([[1., 0., 0.]], device=device),
        pixel_area=torch.ones((1, 1), device=device),
        nears=torch.tensor([[.01]], device=device),
        fars=torch.tensor([[4.]], device=device),
    )
    return model, ray, original, captured


def _trace(model, ray):
    with pytest.raises(TraversalCaptured):
        model.occupancy_ray_marching(ray)


@pytest.mark.parametrize("native", [False, True], ids=["cpu_sampler", "cuda_nerfacc"])
@pytest.mark.parametrize("radius", [0, 1])
def test_constant_step_eval_dilation_admits_neighbor_once_and_restores(radius, native):
    if native and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device("cuda" if native else "cpu")
    model, ray, original, captured = _model(device, radius, native=native)
    model.eval()
    _trace(model, ray)
    expected = original.clone()
    if radius:
        expected[0, 3:6, 3:6, 3:6] = True
    assert torch.equal(captured[-1][0], expected)
    assert bool(captured[-1][1].numel()) == bool(radius)
    first = captured[-1]
    _trace(model, ray)
    # Repeated render chunks cannot turn +1 into +2.
    assert torch.equal(captured[-1][0], first[0])
    assert torch.equal(captured[-1][2], first[2])
    assert torch.equal(captured[-1][3], first[3])
    assert not bool(model.occupancy_grid.binaries[0, 4, 6, 4])
    model.train(True)
    assert torch.equal(model.occupancy_grid.binaries, original)
    assert model._eval_occupancy_backup is None
    # Re-entering eval starts from the exact undilated training mask.
    model.eval()
    _trace(model, ray)
    assert torch.equal(captured[-1][0], expected)
    model.train(True)
    assert torch.equal(model.occupancy_grid.binaries, original)

