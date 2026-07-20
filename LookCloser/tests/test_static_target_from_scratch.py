from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_static_target_from_scratch as controller  # noqa: E402
import static_target_roi_protocol as roi_protocol  # noqa: E402
import build_chroma_normalized_frequency_maps as chroma_maps  # noqa: E402


def args(tmp_path: Path, *extra: str):
    return controller.parse_args(
        [
            "--campaign-name",
            "test_campaign",
            "--output-dir",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def candidate(step: int, psnr: float, ssim: float, lpips: float) -> dict:
    return {
        "step": step,
        "metrics": {"psnr": psnr, "ssim": ssim, "lpips": lpips},
        "roi": {"artifact": {"serious": False}},
        "full_view_serious_count": 0,
        "visual_gate": {"verdict": "pending"},
    }


def test_canonical_stage_a_is_from_scratch_and_stage_b_uses_only_own_parent(tmp_path: Path) -> None:
    parsed = args(tmp_path)
    parent = tmp_path / "out" / "test_campaign_A" / "lookcloser" / "stamp" / "nerfstudio_models" / "step-000075940.ckpt"
    stage_a = controller.stage_command(parsed, "test_campaign_A", "stamp", 75_940, 1.0, None)
    stage_b = controller.stage_command(parsed, "test_campaign_A_fw03", "stamp", 106_316, 0.3, parent)

    assert "--load-checkpoint" not in stage_a
    assert stage_a[stage_a.index("--max-num-iterations") + 1] == "75941"
    assert stage_a[stage_a.index("--feature-reweighting-strength") + 1] == "1.0"
    assert stage_a[stage_a.index("--step-interval") + 1] == "15188"
    assert stage_a[stage_a.index("--save-interval") + 1] == "15188"
    assert stage_b[stage_b.index("--load-checkpoint") + 1] == str(parent)
    assert stage_b[stage_b.index("--feature-reweighting-strength") + 1] == "0.3"
    assert str(controller.DEFAULT_LEADER_CHECKPOINT) not in stage_a
    assert str(controller.DEFAULT_LEADER_CHECKPOINT) not in stage_b


def test_variants_change_only_declared_capacity_or_fas_defaults(tmp_path: Path) -> None:
    canonical = args(tmp_path, "--variant", "canonical")
    fas075 = args(tmp_path, "--variant", "fas075")
    hash24 = args(tmp_path, "--variant", "hash24")
    assert (canonical.fas_strength, canonical.log2_hashmap_size) == (1.0, 23)
    assert (fas075.fas_strength, fas075.log2_hashmap_size) == (0.75, 23)
    assert (hash24.fas_strength, hash24.log2_hashmap_size) == (0.75, 24)


def test_parse_args_allows_explicit_seed_sweep(tmp_path: Path) -> None:
    parsed = args(tmp_path, "--seed", "43")
    assert parsed.seed == 43


def test_stage_b_feature_reweighting_is_explicit_recipe_coordinate(tmp_path: Path) -> None:
    parsed = args(tmp_path, "--stage-b-feature-reweighting", "0.2")
    assert parsed.stage_b_feature_reweighting == 0.2
    assert controller.feature_reweighting_tag(parsed.stage_b_feature_reweighting) == "fw02"

    with pytest.raises(SystemExit):
        args(tmp_path, "--stage-b-feature-reweighting", "1.01")


@pytest.mark.parametrize("seed", ["-1", str(2**32)])
def test_parse_args_rejects_seed_outside_uint32(tmp_path: Path, seed: str) -> None:
    with pytest.raises(SystemExit):
        args(tmp_path, "--seed", seed)


def test_selector_uses_psnr_window_then_lpips() -> None:
    rows = [
        candidate(60_752, 29.90, 0.67, 0.230),
        candidate(75_940, 29.85, 0.67, 0.215),
        candidate(91_128, 29.82, 0.68, 0.205),
    ]
    selected = controller.select_checkpoint(rows)
    assert selected["step"] == 75_940


def test_numeric_gate_requires_all_three_leader_metrics() -> None:
    passing = candidate(91_128, 29.840143, 0.669203, 0.219455)
    assert controller.numeric_pass(passing)
    for metric, value in (("psnr", 29.840142), ("ssim", 0.669202), ("lpips", 0.219456)):
        row = json.loads(json.dumps(passing))
        row["metrics"][metric] = value
        assert not controller.numeric_pass(row)


def test_plateau_requires_two_trailing_numeric_intervals() -> None:
    rows = [
        candidate(60_752, 29.80, 0.6700, 0.2300),
        candidate(75_940, 29.82, 0.6705, 0.2280),
        candidate(91_128, 29.83, 0.6708, 0.2260),
    ]
    summary = controller.plateau_summary(rows)
    assert summary["trailing_numeric_plateau_intervals"] == 2
    assert summary["visual_confirmation_required"] is True
    assert summary["confirmed"] is False

    visual = {
        "60752-75940": {"verdict": "no_improvement"},
        "75940-91128": {"verdict": "no_improvement"},
    }
    assert controller.plateau_summary(rows, visual)["confirmed"] is True


def test_visual_improvement_prevents_confirmed_plateau() -> None:
    rows = [
        candidate(60_752, 29.80, 0.6700, 0.2300),
        candidate(75_940, 29.82, 0.6705, 0.2280),
        candidate(91_128, 29.83, 0.6708, 0.2260),
    ]
    visual = {
        "60752-75940": {"verdict": "no_improvement"},
        "75940-91128": {"verdict": "improved"},
    }
    assert controller.plateau_summary(rows, visual)["confirmed"] is False


def test_metric_improvement_breaks_plateau_streak() -> None:
    rows = [
        candidate(60_752, 29.80, 0.6700, 0.2300),
        candidate(75_940, 29.82, 0.6705, 0.2280),
        candidate(91_128, 29.90, 0.6710, 0.2270),
    ]
    assert controller.plateau_summary(rows)["trailing_numeric_plateau_intervals"] == 0


def test_roi_split_and_fixed_box(tmp_path: Path) -> None:
    height, width = 1080, 1920
    gt = np.zeros((height, width, 3), dtype=np.uint8)
    render = np.full((height, width, 3), 127, dtype=np.uint8)
    pair = np.concatenate((gt, render), axis=1)
    path = tmp_path / "eval_img_0000.png"
    Image.fromarray(pair).save(path)

    loaded_gt, loaded_render = roi_protocol.split_render_pair(path)
    assert np.array_equal(loaded_gt, gt)
    assert np.array_equal(loaded_render, render)
    assert roi_protocol.crop(loaded_gt).shape == (380, 420, 3)
    assert roi_protocol.CONTACT_HANDS_CHAIN_BOX == (700, 100, 1120, 480)


def test_invalid_campaign_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        controller.parse_args(
            ["--campaign-name", "../unsafe", "--output-dir", str(tmp_path / "out")]
        )


def test_latest_completed_tail_allows_incremental_resume(tmp_path: Path) -> None:
    checkpoint_121504 = tmp_path / "step-000121504.ckpt"
    checkpoint_136692 = tmp_path / "step-000136692.ckpt"
    checkpoint_121504.touch()
    checkpoint_136692.touch()
    manifest = {
        "stages": {
            "stage_b": {
                "status": "complete",
                "target_step": 106_316,
                "checkpoint": str(tmp_path / "step-000106316.ckpt"),
            },
            "tail_121504": {
                "status": "complete",
                "target_step": 121_504,
                "checkpoint": str(checkpoint_121504),
            },
            "tail_136692": {
                "status": "complete",
                "target_step": 136_692,
                "checkpoint": str(checkpoint_136692),
            },
        }
    }

    latest = controller.latest_completed_tail(manifest)
    assert latest == checkpoint_136692
    assert controller.checkpoint_step(latest) + controller.CHECKPOINT_INTERVAL == 151_880


def test_chroma_422_normalization_preserves_luminance_and_reduces_horizontal_chroma() -> None:
    import torch

    image = torch.zeros((8, 8, 3), dtype=torch.float32)
    image[:, 0::2, 0] = 1.0
    image[:, 1::2, 2] = 1.0
    normalized = chroma_maps.normalize_chroma_422(image)

    old_y = chroma_maps.rgb_luminance(image)
    new_y = chroma_maps.rgb_luminance(normalized)
    old_horizontal = torch.mean(torch.abs(image[:, 1:] - image[:, :-1]))
    new_horizontal = torch.mean(torch.abs(normalized[:, 1:] - normalized[:, :-1]))
    assert torch.mean(torch.abs(new_y - old_y)).item() < 0.01
    assert new_horizontal < old_horizontal
