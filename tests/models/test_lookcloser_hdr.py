"""LookCloser scene-linear HDR output and reconstruction tests."""

from types import SimpleNamespace

import pytest
import torch
from torchmetrics.functional import structural_similarity_index_measure

from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.models.lookcloser import LookCloserModel
from nerfstudio.utils.hdr import activate_hdr_rgb


def _loss_config(loss_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        reconstruction_loss_type=loss_type,
        huber_delta=0.1,
        rawnerf_epsilon=1e-3,
        pq_black_nits=0.005,
        pq_linear_anchor_weight=0.0,
        eag_patch_size=3,
        eag_dssim_weight=0.2,
        eag_edge_weight=0.0,
        distortion_loss_mult=0.0,
        depth_loss_mult=0.0,
    )


class _LossHarness:
    device = torch.device("cpu")
    training = True
    hdr_linear_scale = 2.0
    pq_nits_per_scene_unit = 100.0
    ssim = staticmethod(structural_similarity_index_measure)


@pytest.mark.parametrize("loss_type", ["linear_l1", "rawnerf_weighted_l2", "linear_pq", "pq_l1"])
def test_hdr_losses_are_finite_and_differentiable(loss_type):
    harness = _LossHarness()
    harness.config = _loss_config(loss_type)
    prediction = torch.tensor([[0.0, 0.5, 4.0], [0.1, 1.0, 8.0]], requires_grad=True)
    target = torch.tensor([[-0.1, 0.25, 3.0], [0.2, 2.0, 9.0]])
    loss = LookCloserModel.get_loss_dict(harness, {"rgb": prediction}, {"image": target})["rgb_loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


def test_eag_loss_requires_patch_batches():
    harness = _LossHarness()
    harness.config = _loss_config("eag_pq_dssim")
    prediction = torch.ones((8, 3), requires_grad=True)
    with pytest.raises(ValueError, match="contiguous patch batches"):
        LookCloserModel.get_loss_dict(harness, {"rgb": prediction}, {"image": torch.ones_like(prediction)})


def test_eag_eval_loss_falls_back_to_pq_l1_for_unstructured_rays():
    harness = _LossHarness()
    harness.training = False
    harness.config = _loss_config("eag_pq_dssim")
    prediction = torch.ones((8, 3), requires_grad=True)
    loss = LookCloserModel.get_loss_dict(
        harness,
        {"rgb": prediction},
        {"image": torch.full_like(prediction, 0.75)},
    )["rgb_loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


def test_eag_edge_term_penalizes_a_broken_patch_edge():
    harness = _LossHarness()
    harness.config = _loss_config("eag_pq_dssim")
    harness.ssim = lambda prediction, target, data_range: prediction.new_tensor(1.0)
    target_patch = torch.zeros((3, 3, 3))
    target_patch[:, 1:, :] = 1.0
    target = target_patch.reshape(-1, 3)
    prediction = target.clone()
    prediction[4] = 0.0

    harness.config.eag_edge_weight = 0.0
    base = LookCloserModel.get_loss_dict(harness, {"rgb": prediction}, {"image": target})["rgb_loss"]
    harness.config.eag_edge_weight = 0.2
    edge = LookCloserModel.get_loss_dict(harness, {"rgb": prediction}, {"image": target})["rgb_loss"]

    assert edge > base


def test_hdr_output_parameterizations_are_positive_and_unbounded():
    raw = torch.tensor([[-10.0, 0.0, 20.0]], requires_grad=True)
    linear = activate_hdr_rgb(
        raw,
        parameterization="linear_softplus",
        linear_scale=0.1,
        initial_radiance=0.01,
        nits_per_scene_unit=100.0,
    )
    pq_code = activate_hdr_rgb(
        raw,
        parameterization="pq_code",
        linear_scale=0.1,
        initial_radiance=0.01,
        nits_per_scene_unit=100.0,
    )
    assert bool((linear >= 0).all())
    assert float(linear.max()) > 1.0
    assert bool((pq_code >= 0).all())
    (linear.sum() + pq_code.sum()).backward()
    assert torch.isfinite(raw.grad).all()


def test_rgb_renderer_can_preserve_hdr_during_eval():
    renderer = RGBRenderer(background_color="black", clamp_output=False).eval()
    rgb = torch.tensor([[[4.0, -0.25, 2.0]]])
    weights = torch.ones((1, 1, 1))
    rendered = renderer(rgb, weights)
    torch.testing.assert_close(rendered, rgb[:, 0])
