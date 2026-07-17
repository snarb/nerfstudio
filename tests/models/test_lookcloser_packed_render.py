"""Forward and gradient parity for the shared packed black renderer."""

import torch

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.models.lookcloser import LookCloserModel


def _samples() -> tuple[RaySamples, torch.Tensor]:
    ray_indices = torch.tensor([0, 0, 2, 2, 2], dtype=torch.long)
    starts = torch.tensor([[0.0], [1.0], [0.0], [1.0], [2.0]])
    ends = starts + 0.5
    ray_samples = RaySamples(
        frustums=Frustums(
            origins=torch.zeros((5, 3)),
            directions=torch.ones((5, 3)),
            starts=starts,
            ends=ends,
            pixel_area=torch.ones((5, 1)),
        ),
        camera_indices=None,
        deltas=ends - starts,
        spacing_starts=starts,
        spacing_ends=ends,
    )
    return ray_samples, ray_indices


def _reference(rgb, weights, ray_samples, ray_indices):
    return (
        RGBRenderer("black")(rgb, weights, ray_indices, 3),
        AccumulationRenderer()(weights, ray_indices, 3),
        DepthRenderer("expected")(weights, ray_samples, ray_indices, 3),
    )


def test_shared_packed_black_render_is_forward_and_gradient_exact() -> None:
    ray_samples, ray_indices = _samples()
    generator = torch.Generator().manual_seed(42)
    base_rgb = torch.rand((5, 3), generator=generator)
    base_weights = torch.rand((5, 1), generator=generator)
    gradients = []
    for fast in (False, True):
        rgb = base_rgb.clone().requires_grad_()
        weights = base_weights.clone().requires_grad_()
        outputs = (
            LookCloserModel._render_packed_black(rgb, weights, ray_samples, ray_indices, 3)
            if fast
            else _reference(rgb, weights, ray_samples, ray_indices)
        )
        if not fast:
            reference = tuple(output.detach().clone() for output in outputs)
        else:
            assert all(torch.equal(actual, expected) for actual, expected in zip(outputs, reference))
        loss = outputs[0].sum() + 0.3 * outputs[1].sum() + 0.7 * outputs[2].sum()
        loss.backward()
        gradients.append((rgb.grad, weights.grad))
    assert torch.equal(gradients[0][0], gradients[1][0])
    assert torch.equal(gradients[0][1], gradients[1][1])
