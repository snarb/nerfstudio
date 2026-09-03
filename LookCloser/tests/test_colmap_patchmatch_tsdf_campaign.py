from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load_script("colmap_patchmatch_tsdf_campaign_common.py")
SCORER = load_script("score_colmap_patchmatch_tsdf_face.py")
AUDIT = load_script("audit_colmap_patchmatch_tsdf_campaign.py")
CONTROLLER = load_script("run_colmap_patchmatch_tsdf_campaign.py")


def test_numeric_discovery_is_exact_ordered_prefix(tmp_path: Path) -> None:
    for name in ("000903", "000899", "000901", "notes", "12345"):
        (tmp_path / name).mkdir()

    result = COMMON.discover_frames(tmp_path, count=3)

    assert [path.name for path in result] == ["000899", "000901", "000903"]


def test_initial_thresholds_combine_floor_and_robust_spread() -> None:
    rows = [
        {"face_psnr": 20.0, "face_ssim": 0.80, "face_lpips": 0.20},
        {"face_psnr": 20.1, "face_ssim": 0.81, "face_lpips": 0.21},
        {"face_psnr": 20.2, "face_ssim": 0.82, "face_lpips": 0.22},
    ]

    thresholds = COMMON.robust_initial_thresholds(rows)

    assert thresholds["face_psnr"] == 1.0
    assert np.isclose(thresholds["face_ssim"], 0.044478)
    assert thresholds["face_lpips"] == 0.05


def test_manual_mask_is_bound_to_gt_hash_and_prediction_is_forbidden(tmp_path: Path) -> None:
    gt = tmp_path / "gt.png"
    Image.new("RGB", (32, 24), "white").save(gt)
    polygons = tmp_path / "face.json"
    polygons.write_text(
        json.dumps(
            {
                "selection_method": "manual_polygon_on_heldout_gt_only",
                "prediction_used_for_selection": False,
                "ground_truth_sha256": SCORER.sha256(gt),
                "include_polygons": [[[4, 4], [16, 4], [16, 16], [4, 16]]],
                "exclude_polygons": [[[10, 8], [12, 8], [12, 10], [10, 10]]],
            }
        ),
        encoding="utf-8",
    )

    mask, _ = SCORER.load_manual_face_mask(polygons, gt, (24, 32))

    assert mask[6, 6]
    assert not mask[9, 11]
    payload = json.loads(polygons.read_text())
    payload["prediction_used_for_selection"] = True
    polygons.write_text(json.dumps(payload), encoding="utf-8")
    try:
        SCORER.load_manual_face_mask(polygons, gt, (24, 32))
    except ValueError as error:
        assert "prediction_used_for_selection=false" in str(error)
    else:
        raise AssertionError("prediction-assisted face polygons must be rejected")


def test_masked_metrics_penalize_black_hole_inside_manual_face() -> None:
    gt = torch.ones((3, 16, 16), dtype=torch.float32)
    prediction = gt.clone()
    prediction[:, 4:8, 4:8] = 0.0
    mask = torch.zeros((16, 16), dtype=torch.bool)
    mask[2:14, 2:14] = True

    result = SCORER.masked_display_metrics(
        prediction, gt, mask, lambda left, right: torch.mean(torch.abs(left - right))
    )

    assert result["face_psnr"] < 20.0
    assert result["face_ssim"] < 1.0
    assert result["face_lpips"] > 0.0


def test_audit_rejects_non_face_metric_keys() -> None:
    AUDIT.assert_no_full_frame_metric_keys({"face_psnr": 20.0, "protocol": {"face": True}})
    try:
        AUDIT.assert_no_full_frame_metric_keys({"aggregate": {"psnr": 20.0}})
    except ValueError as error:
        assert "Forbidden non-face metric" in str(error)
    else:
        raise AssertionError("full-frame metric key must be rejected")


def test_remote_code_bundle_includes_exporter_dependency() -> None:
    assert "run_colmap_patchmatch_tsdf_campaign.py" in CONTROLLER.CAMPAIGN_SCRIPTS
    assert "export_nerfstudio_colmap_model.py" in CONTROLLER.CAMPAIGN_SCRIPTS
    assert "seed_colmap_from_nerfstudio.py" in CONTROLLER.CAMPAIGN_SCRIPTS


def test_campaign_recipe_enables_geometry_only_screen_component_fix() -> None:
    assert COMMON.RECIPE["target_depth_component_min_area"] == 1000
    assert COMMON.RECIPE["target_depth_component_max_log_jump"] == 0.0075
    assert COMMON.RECIPE["nearest_fill_color_continuity"] is True
    assert COMMON.RECIPE["nearest_fill_color_continuity_mode"] == "global"


def test_rsync_uses_shared_mount_safe_content_mode(monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(CONTROLLER, "run", lambda command: commands.append(command))

    CONTROLLER.rsync("source/", "host:destination/", delete=True)

    command = commands[0]
    assert "--inplace" in command
    assert "--no-times" in command
    assert "--no-perms" in command
    assert "--delete" in command
    assert "-a" not in command


def test_copy_tree_content_does_not_require_metadata_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "payload.bin").write_bytes(b"campaign")
    destination = tmp_path / "destination"

    CONTROLLER.copy_tree_content(source, destination)

    assert (destination / "nested" / "payload.bin").read_bytes() == b"campaign"
