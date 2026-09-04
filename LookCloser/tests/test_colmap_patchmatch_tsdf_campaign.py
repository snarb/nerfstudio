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
RESCORER = load_script("rescore_colmap_patchmatch_tsdf_face_rois.py")
RENDER_REVISION = load_script("revise_colmap_patchmatch_tsdf_renders.py")


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


def test_audit_validates_base_render_revision_provenance(tmp_path: Path) -> None:
    frame_id = "000899"
    correction_id = "hard_source_continuity_v1"
    base = tmp_path / ".diagnostics" / f"render_revision_{correction_id}"
    before = base / "before" / frame_id / "render"
    current = tmp_path / "frames" / frame_id / "render"
    before.mkdir(parents=True)
    current.mkdir(parents=True)
    (before / "eval_pred_0000.png").write_bytes(b"before")
    (current / "eval_pred_0000.png").write_bytes(b"after")
    COMMON.atomic_json(
        base / "correction_manifest.json",
        {"status": "complete", "correction_id": correction_id},
    )
    COMMON.atomic_json(
        base / "frame_receipts" / f"{frame_id}.json",
        {
            "frame_id": frame_id,
            "after_render_sha256": COMMON.sha256(current / "eval_pred_0000.png"),
            "mesh_unchanged": True,
        },
    )
    result = {
        "frame_id": frame_id,
        "render_sha256": COMMON.sha256(current / "eval_pred_0000.png"),
        "render_revision": {
            "correction_id": correction_id,
            "remote_render_preserved_in": str(before),
            "before_render_sha256": COMMON.sha256(before / "eval_pred_0000.png"),
            "after_render_sha256": COMMON.sha256(current / "eval_pred_0000.png"),
            "render_arguments": {
                "uses_eval_rgb_for_prediction": False,
                "uses_masks": False,
                "averages_sources": False,
                "primary_color_continuation": True,
            },
        },
    }

    AUDIT.verify_render_revision(tmp_path, result)


def test_remote_code_bundle_includes_exporter_dependency() -> None:
    assert "run_colmap_patchmatch_tsdf_campaign.py" in CONTROLLER.CAMPAIGN_SCRIPTS
    assert "export_nerfstudio_colmap_model.py" in CONTROLLER.CAMPAIGN_SCRIPTS
    assert "seed_colmap_from_nerfstudio.py" in CONTROLLER.CAMPAIGN_SCRIPTS
    assert "revise_colmap_patchmatch_tsdf_renders.py" in CONTROLLER.CAMPAIGN_SCRIPTS


def test_campaign_recipe_enables_geometry_only_screen_component_fix() -> None:
    assert COMMON.RECIPE["target_depth_component_min_area"] == 1000
    assert COMMON.RECIPE["target_depth_component_max_log_jump"] == 0.0075
    assert COMMON.RECIPE["nearest_fill_color_continuity"] is True
    assert COMMON.RECIPE["nearest_fill_color_continuity_mode"] == "global"
    assert COMMON.RECIPE["nearest_fill_primary_color_continuation"] is True
    assert COMMON.RECIPE["nearest_fill_primary_color_continuation_min_area"] == 20
    assert COMMON.RECIPE["nearest_fill_primary_color_continuation_max_area"] == 1000
    assert COMMON.RECIPE["nearest_fill_primary_color_continuation_min_median_l1"] == 0.1


def test_render_revision_requires_complete_published_prefix(tmp_path: Path) -> None:
    ordered = ["000899", "000901", "000903"]
    (tmp_path / "frames/000899").mkdir(parents=True)
    (tmp_path / "frames/000901").mkdir()
    request = {"ordered_frame_ids": ordered}

    assert RENDER_REVISION.selected_prefix(tmp_path, request, None) == ordered[:2]
    try:
        RENDER_REVISION.selected_prefix(tmp_path, request, ["000899"])
    except ValueError as error:
        assert "complete published ordered prefix" in str(error)
    else:
        raise AssertionError("partial render revision must be rejected")


def test_render_revision_extension_requires_exact_next_frame(tmp_path: Path) -> None:
    ordered = ["000899", "000901", "000903"]
    (tmp_path / "frames/000899").mkdir(parents=True)
    retained = tmp_path / ".work/frames/000901/retained"
    retained.mkdir(parents=True)
    (retained / "payload.bin").write_bytes(b"verified")
    COMMON.atomic_json(
        retained / "retained_manifest.json",
        {
            "schema_version": 1,
            "files": [
                {
                    "path": "payload.bin",
                    "bytes": len(b"verified"),
                    "sha256": COMMON.sha256(retained / "payload.bin"),
                }
            ],
        },
    )
    request = {"ordered_frame_ids": ordered}

    RENDER_REVISION.next_unpublished_frame(tmp_path, request, "000901")
    try:
        RENDER_REVISION.next_unpublished_frame(tmp_path, request, "000903")
    except ValueError as error:
        assert "must target next frame 000901" in str(error)
    else:
        raise AssertionError("render correction extension must not skip an unpublished frame")


def test_frozen_finalize_command_pins_immutable_request_values(tmp_path: Path) -> None:
    request = {
        "source_root": "/source",
        "calibration_template_source": "/calibration.json",
        "remote": {
            "host": "ubuntu@dev3",
            "scratch_root": "/scratch",
            "python": "/remote/python",
            "colmap": "/usr/local/bin/colmap",
            "gpu_index": "0",
        },
    }

    command = RENDER_REVISION.frozen_finalize_command(tmp_path, request, "000901")

    assert str(tmp_path / "config/code/run_colmap_patchmatch_tsdf_campaign.py") in command
    assert command[-2:] == ["--frames", "000901"]
    assert command[command.index("--source-root") + 1] == "/source"
    assert command[command.index("--remote-colmap") + 1] == "/usr/local/bin/colmap"


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


def test_roi_rescore_requires_contiguous_published_prefix(tmp_path: Path) -> None:
    ordered = ["000899", "000901", "000903"]
    (tmp_path / "frames" / "000899").mkdir(parents=True)
    (tmp_path / "frames" / "000903").mkdir()

    try:
        RESCORER.published_prefix(tmp_path, ordered)
    except ValueError as error:
        assert "contiguous ordered prefix" in str(error)
    else:
        raise AssertionError("a gapped finalized inventory must not be rescored")


def test_roi_rescore_removes_stale_controller_lock(tmp_path: Path) -> None:
    lock = tmp_path / ".campaign_controller.lock"
    lock.write_text("pid=999999999 started_at=old\n", encoding="utf-8")

    RESCORER.reject_live_campaign_controller(lock)

    assert not lock.exists()


def test_roi_rescore_metric_status_uses_correct_direction() -> None:
    thresholds = {"face_psnr": 1.0, "face_ssim": 0.03, "face_lpips": 0.05}
    previous = [
        {"face_psnr": 29.0, "face_ssim": 0.89, "face_lpips": 0.055},
        {"face_psnr": 28.8, "face_ssim": 0.88, "face_lpips": 0.060},
    ]

    status, reasons = RESCORER.metric_status(
        {"face_psnr": 27.0, "face_ssim": 0.84, "face_lpips": 0.12},
        thresholds,
        previous,
    )

    assert status == "regression_flag"
    assert reasons == ["face_psnr_drop", "face_ssim_drop", "face_lpips_rise"]


def test_roi_rescore_updates_only_metric_result_fields() -> None:
    old = {
        "frame_id": "000899",
        "face_psnr": 20.0,
        "face_ssim": 0.7,
        "face_lpips": 0.2,
        "metric_status": "regression_flag",
        "visual_status": "pass",
        "status": "pass",
        "mesh_sha256": "mesh",
        "render_sha256": "render",
    }
    metrics = {
        "face_psnr": 29.0,
        "face_ssim": 0.89,
        "face_lpips": 0.05,
        "metric_status": "pass",
    }

    result = RESCORER.updated_result(old, metrics, "roi_v2", "polygon")

    assert result["face_lpips"] == 0.05
    assert result["metric_status"] == "pass"
    assert result["mesh_sha256"] == "mesh"
    assert result["render_sha256"] == "render"
    assert result["metric_revision"]["face_polygons_sha256"] == "polygon"
