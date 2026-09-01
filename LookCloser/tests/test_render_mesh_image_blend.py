from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_mesh_image_blend.py"
SPEC = importlib.util.spec_from_file_location("render_mesh_image_blend_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_identity_camera_projection_preserves_pixels_and_depth() -> None:
    depth = torch.full((3, 4), 2.0)
    c2w = torch.eye(4)[:3]
    intrinsics = {"fx": 10.0, "fy": 12.0, "cx": 1.5, "cy": 1.0, "width": 4, "height": 3}
    u, v, z = MODULE.project_target_to_source(depth, c2w, intrinsics, c2w, intrinsics)
    yy, xx = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    torch.testing.assert_close(u, xx.float())
    torch.testing.assert_close(v, yy.float())
    torch.testing.assert_close(z, depth)


def test_grid_sample_identity() -> None:
    image = torch.arange(3 * 3 * 4, dtype=torch.float32).reshape(3, 3, 4)
    yy, xx = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    sampled = MODULE.grid_sample(image, xx.float(), yy.float())
    torch.testing.assert_close(sampled, image)


def test_camera_distance_power_must_be_non_negative(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    manifest = tmp_path / "depth.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--data",
            str(data),
            "--mesh-depth-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--camera-distance-power",
            "-1",
        ],
    )
    with pytest.raises(SystemExit):
        MODULE.parse_args()


def test_resolve_manifest_path_prefers_dataset_relative_file(tmp_path: Path) -> None:
    data = tmp_path / "dataset"
    expected = data / "depth" / "frame.npy.gz"
    expected.parent.mkdir(parents=True)
    expected.touch()
    manifest = tmp_path / "portable_manifest.json"
    manifest.touch()

    assert MODULE.resolve_manifest_path("depth/frame.npy.gz", data, manifest) == expected.resolve()


def test_surface_detail_transfer_preserves_base_when_source_matches() -> None:
    base = torch.rand(3, 9, 11)
    valid = torch.ones(1, 9, 11, dtype=torch.bool)

    result = MODULE.surface_detail_transfer(
        base=base,
        source=base,
        valid=valid,
        sigma=2.0,
        strength=1.0,
    )

    torch.testing.assert_close(result, base, atol=1e-6, rtol=1e-6)


def test_display_metrics_and_single_roi_are_diagnostic_only(tmp_path: Path) -> None:
    roi_path = tmp_path / "roi.json"
    roi_path.write_text('{"name": "face", "boxes_xyxy": [[2, 2, 14, 14]]}', encoding="utf-8")
    roi, name = MODULE.load_single_roi(roi_path)
    image = torch.rand(3, 16, 16)

    metrics = MODULE.display_metrics(
        image,
        image,
        lpips_model=lambda left, right: torch.mean(torch.abs(left - right)),
        roi=roi,
    )

    assert name == "face"
    assert metrics["psnr"] == pytest.approx(120.0)
    assert metrics["ssim"] == pytest.approx(1.0)
    assert metrics["lpips"] == 0.0
    assert metrics["roi_lpips"] == 0.0


def test_normalized_blur_does_not_darkens_constant_at_support_boundary() -> None:
    image = torch.full((3, 9, 11), 0.7)
    valid = torch.zeros(1, 9, 11, dtype=torch.bool)
    valid[:, 2:7, 3:9] = True

    blurred = MODULE.normalized_gaussian_blur(image, valid, sigma=1.5)

    torch.testing.assert_close(blurred[:, 2:7, 3:9], image[:, 2:7, 3:9], atol=1e-5, rtol=1e-5)
