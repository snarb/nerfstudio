"""Verify actual exclusion of unwanted RGB, including perceptual footprints."""
import sys
from pathlib import Path

import pytest
import torch
from torchmetrics.functional.image.lpips import _NoTrainLpips

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from marshawn_actor_metrics import actor_image_metrics


@pytest.fixture(scope="module")
def lpips_model():
    torch.set_num_threads(2)
    return _NoTrainLpips(net="alex").eval()


def test_excluded_rgb_cannot_change_any_score(lpips_model):
    rng = torch.Generator().manual_seed(71)
    gt = torch.rand((1, 3, 320, 384), generator=rng)
    pred = (gt * .92 + .03).clamp(0, 1)
    mask = torch.zeros((1, 1, 320, 384), dtype=torch.bool)
    mask[:, :, 32:288, 40:344] = True
    before = actor_image_metrics(pred, gt, mask, lpips_model)
    outside = ~mask.expand_as(gt)
    altered_gt, altered_pred = gt.clone(), pred.clone()
    altered_gt[outside] = torch.rand(int(outside.sum()), generator=rng)
    altered_pred[outside] = torch.rand(int(outside.sum()), generator=rng)
    after = actor_image_metrics(altered_pred, altered_gt, mask, lpips_model)
    for key in ("psnr", "ssim", "lpips"):
        assert before[key] is not None
        assert after[key] == pytest.approx(before[key], abs=2e-7)
    assert before["actor_pixels"] == 256 * 304
    assert before["ssim_valid_centers"] < before["actor_pixels"]
    assert all(x["valid_features"] > 0 for x in before["lpips_feature_support"])
    # In-domain damage must remain observable despite excluding the background.
    damaged = pred.clone()
    damaged[:, :, 90:230, 100:280] = 0
    damage = actor_image_metrics(damaged, gt, mask, lpips_model)
    assert damage["psnr"] < before["psnr"] - 5
    assert damage["ssim"] < before["ssim"]
    assert damage["lpips"] > before["lpips"]


def test_full_domain_matches_standard_lpips(lpips_model):
    rng = torch.Generator().manual_seed(17)
    gt = torch.rand((1, 3, 192, 224), generator=rng)
    pred = gt * .8
    mask = torch.ones((1, 1, 192, 224), dtype=torch.bool)
    result = actor_image_metrics(pred, gt, mask, lpips_model)
    with torch.inference_mode():
        reference = float(lpips_model(pred, gt, normalize=True).mean())
    assert result["lpips"] == pytest.approx(reference, abs=2e-7)
    assert result["actor_pixels"] == result["ssim_valid_centers"]


def test_thin_domain_does_not_silently_drop_lpips_levels(lpips_model):
    gt = torch.ones((1, 3, 192, 224)) * .5
    mask = torch.zeros((1, 1, 192, 224), dtype=torch.bool)
    mask[:, :, :, 100:120] = True
    result = actor_image_metrics(gt, gt, mask, lpips_model)
    assert result["psnr"] == float("inf")
    assert result["ssim"] == pytest.approx(1)
    assert result["lpips"] is None
    with pytest.raises(ValueError, match="Empty actor"):
        actor_image_metrics(gt, gt, torch.zeros_like(mask), lpips_model)
