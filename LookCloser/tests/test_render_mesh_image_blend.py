from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_mesh_image_blend.py"
SPEC = importlib.util.spec_from_file_location("render_mesh_image_blend_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_rgb_footprint_controls_are_opt_in(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir();manifest=tmp_path/'depth.json';manifest.write_text('{}')
    common=['render','--data',str(data),'--mesh-depth-manifest',str(manifest),'--output-dir',str(tmp_path/'out')]
    monkeypatch.setattr(sys,'argv',common)
    args=MODULE.parse_args()
    assert not args.source_rgb_footprint_visibility and not args.source_rgb_depth_aware_sampling
    assert not args.source_observed_free_space_veto and args.seam_cut_bandwidth_penalty==0
    assert not args.seam_cut_bandwidth_allow_primary
    assert not args.surface_texture_registration and args.pixel_center_offset==0
    monkeypatch.setattr(sys,'argv',common+['--source-rgb-depth-aware-sampling'])
    with pytest.raises(SystemExit):MODULE.parse_args()
    monkeypatch.setattr(sys,'argv',common+['--source-rgb-depth-aware-sampling','--exact-mesh-visibility','--pixel-center-offset','.5'])
    assert MODULE.parse_args().source_rgb_depth_aware_sampling
    monkeypatch.setattr(sys,'argv',sys.argv+['--source-rgb-footprint-visibility'])
    with pytest.raises(SystemExit):MODULE.parse_args()


def test_free_space_rgb_veto_requires_observed_depth_and_normalization(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir();manifest=tmp_path/'depth.json';manifest.write_text('{}')
    common=['render','--data',str(data),'--mesh-depth-manifest',str(manifest),'--output-dir',str(tmp_path/'out'),
            '--source-observed-free-space-veto']
    monkeypatch.setattr(sys,'argv',common)
    with pytest.raises(SystemExit):MODULE.parse_args()
    monkeypatch.setattr(sys,'argv',common+['--source-observed-depth-data',str(data)])
    with pytest.raises(SystemExit):MODULE.parse_args()
    monkeypatch.setattr(sys,'argv',sys.argv+['--source-observed-mesh-metadata',str(manifest)])
    assert MODULE.parse_args().source_observed_free_space_veto


def test_primary_bandwidth_penalty_requires_positive_hard_source_prior(tmp_path,monkeypatch):
    manifest=tmp_path/'depth.json';manifest.write_text('{}')
    common=['render','--data',str(tmp_path),'--mesh-depth-manifest',str(manifest),
            '--output-dir',str(tmp_path/'out'),'--seam-cut-bandwidth-allow-primary']
    monkeypatch.setattr(sys,'argv',common)
    with pytest.raises(SystemExit):MODULE.parse_args()
    monkeypatch.setattr(sys,'argv',common+['--aggregation-modes','seam-cut','--seam-cut-bandwidth-penalty','.003'])
    assert MODULE.parse_args().seam_cut_bandwidth_allow_primary


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


def test_half_pixel_identity_projection_preserves_array_indices() -> None:
    intrinsics={"fx":10.,"fy":12.,"cx":1.5,"cy":1.,"width":4,"height":3,"pixel_center_offset":.5}
    depth=torch.full((3,4),2.);pose=torch.eye(4)[:3]
    u,v,z=MODULE.project_target_to_source(depth,pose,intrinsics,pose,intrinsics)
    yy,xx=torch.meshgrid(torch.arange(3),torch.arange(4),indexing='ij')
    torch.testing.assert_close(u,xx.float());torch.testing.assert_close(v,yy.float());torch.testing.assert_close(z,depth)


def test_raycast_depth_unprojects_to_mesh_only_with_matching_pixel_center() -> None:
    o3d=pytest.importorskip('open3d')
    vertices=np.array([[-1,-1,-2.6],[1,-1,-1.4],[-1,1,-2.6],[1,1,-1.4]],np.float32)
    mesh=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices),o3d.core.Tensor(np.array([[0,1,2],[1,3,2]],np.int32)))
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(mesh)
    K=np.array([[10,0,2],[0,10,2],[0,0,1]],np.float32)
    ext=np.diag([1,-1,-1,1]).astype(np.float32)
    rays=scene.create_rays_pinhole(o3d.core.Tensor(K),o3d.core.Tensor(ext),4,4)
    t=scene.cast_rays(rays)['t_hit'].numpy();positions=rays.numpy()[...,:3]+rays.numpy()[...,3:]*t[...,None]
    depth=torch.tensor(-positions[...,2]);intrinsics={"fx":10.,"fy":10.,"cx":2.,"cy":2.,"width":4,"height":4}
    for offset in [0.,.5]:
        points=MODULE.target_depth_to_world(depth,torch.eye(4)[:3],dict(intrinsics,pixel_center_offset=offset))
        error=scene.compute_distance(o3d.core.Tensor(points.numpy().reshape(-1,3))).numpy()
        if offset==.5:assert error.max()<1e-6
        else:assert error.min()>.01


def test_nearest_fill_uses_later_source_only_for_holes() -> None:
    first = torch.full((3, 2, 3), 0.25)
    second = torch.full((3, 2, 3), 0.75)
    first_valid = torch.tensor([[True, False, True], [False, False, True]])
    second_valid = torch.tensor([[True, True, True], [True, False, False]])
    ones = [first_valid.float(), second_valid.float()]

    rgb, valid, selected = MODULE.aggregate_warped_sources(
        [first, second],
        [first_valid, second_valid],
        ones,
        ones,
        mode="nearest-fill",
    )

    assert selected.tolist() == [[0, 1, 0], [1, -1, 0]]
    assert valid[0].tolist() == [[True, True, True], [True, False, True]]
    torch.testing.assert_close(rgb[:, 0, 0], first[:, 0, 0])
    torch.testing.assert_close(rgb[:, 0, 1], second[:, 0, 1])


def test_nearest_fill_color_continuity_hard_selects_matching_fallback() -> None:
    first = torch.tensor([[[0.2, 0.0, 0.2]], [[0.2, 0.0, 0.2]], [[0.2, 0.0, 0.2]]])
    mismatched = torch.full((3, 1, 3), 0.9)
    matching = torch.full((3, 1, 3), 0.25)
    first_valid = torch.tensor([[True, False, True]])
    fallback_valid = torch.tensor([[False, True, True]])
    scores = [first_valid.float(), fallback_valid.float(), fallback_valid.float()]

    rgb, valid, selected = MODULE.aggregate_warped_sources(
        [first, mismatched, matching],
        [first_valid, fallback_valid, fallback_valid],
        scores,
        scores,
        mode="nearest-fill",
        nearest_fill_color_continuity=True,
    )

    assert valid[0].tolist() == [[True, True, True]]
    assert selected.tolist() == [[0, 2, 0]]
    torch.testing.assert_close(rgb[:, 0, 1], matching[:, 0, 1])


def test_nearest_fill_global_color_order_uses_one_photometrically_matching_fallback() -> None:
    first = torch.tensor([[[0.2, 0.0, 0.2]], [[0.2, 0.0, 0.2]], [[0.2, 0.0, 0.2]]])
    mismatched = torch.full((3, 1, 3), 0.9)
    matching = torch.full((3, 1, 3), 0.25)
    first_valid = torch.tensor([[True, False, True]])
    fallback_valid = torch.tensor([[False, True, True]])
    scores = [first_valid.float(), fallback_valid.float(), fallback_valid.float()]

    rgb, _, selected = MODULE.aggregate_warped_sources(
        [first, mismatched, matching],
        [first_valid, fallback_valid, fallback_valid],
        scores,
        scores,
        mode="nearest-fill",
        nearest_fill_color_continuity=True,
        nearest_fill_color_continuity_mode="global",
    )

    assert selected.tolist() == [[0, 2, 0]]
    torch.testing.assert_close(rgb[:, 0, 1], matching[:, 0, 1])


def test_primary_color_continuation_replaces_only_small_discontinuous_fallback() -> None:
    rgb = torch.full((3, 5, 7), 0.2)
    primary = rgb.clone()
    primary[:, 1:3, 2:4] = 0.3
    rgb[:, 1:3, 2:4] = 0.9
    selection = torch.zeros((5, 7), dtype=torch.long)
    selection[1:3, 2:4] = 2
    primary_valid = torch.ones((5, 7), dtype=torch.bool)
    primary_valid[1:3, 2:4] = False
    primary_projectable = torch.ones((5, 7), dtype=torch.bool)

    result, selected, stats = MODULE.continue_primary_color_across_small_fallback_components(
        rgb,
        selection,
        primary,
        primary_valid,
        primary_projectable,
        min_area=2,
        max_area=6,
        min_median_l1=0.1,
    )

    torch.testing.assert_close(result[:, 1:3, 2:4], primary[:, 1:3, 2:4])
    assert torch.all(selected[1:3, 2:4] == 0)
    assert stats["selected_components"] == 1
    assert stats["replaced_pixels"] == 4


def test_primary_color_continuation_preserves_continuous_or_unprojectable_fallback() -> None:
    primary = torch.full((3, 4, 8), 0.2)
    rgb = primary.clone()
    rgb[:, 1:3, 1:3] = 0.22
    rgb[:, 1:3, 5:7] = 0.9
    selection = torch.zeros((4, 8), dtype=torch.long)
    selection[1:3, 1:3] = 1
    selection[1:3, 5:7] = 2
    primary_valid = selection == 0
    primary_projectable = torch.ones((4, 8), dtype=torch.bool)
    primary_projectable[1:3, 5:7] = False

    result, selected, stats = MODULE.continue_primary_color_across_small_fallback_components(
        rgb,
        selection,
        primary,
        primary_valid,
        primary_projectable,
        min_area=2,
        max_area=6,
        min_median_l1=0.1,
    )

    torch.testing.assert_close(result, rgb)
    torch.testing.assert_close(selected, selection)
    assert stats["selected_components"] == 0
    assert stats["replaced_pixels"] == 0


def test_best_view_hard_selects_highest_score_without_averaging() -> None:
    first = torch.full((3, 2, 2), 0.2)
    second = torch.full((3, 2, 2), 0.8)
    valid = torch.ones(2, 2, dtype=torch.bool)
    best_scores = [
        torch.tensor([[0.9, 0.1], [0.8, 0.2]]),
        torch.tensor([[0.1, 0.9], [0.2, 0.8]]),
    ]

    rgb, _, selected = MODULE.aggregate_warped_sources(
        [first, second],
        [valid, valid],
        [valid.float(), valid.float()],
        best_scores,
        mode="best-view",
    )

    assert selected.tolist() == [[0, 1], [0, 1]]
    torch.testing.assert_close(rgb[:, 0, 0], first[:, 0, 0])
    torch.testing.assert_close(rgb[:, 0, 1], second[:, 0, 1])


def test_depth_hole_fill_recovers_small_locally_planar_hole() -> None:
    yy, xx = np.mgrid[:32, :40]
    depth = (3.0 + 0.01 * xx + 0.02 * yy).astype(np.float32)
    expected = depth.copy()
    depth[12:18, 15:23] = 0.0

    result, stats = MODULE.fill_small_consistent_depth_holes(
        depth,
        max_area=100,
        boundary_radius=3,
        max_relative_plane_rmse=0.001,
    )

    np.testing.assert_allclose(result, expected, atol=1e-5)
    assert stats["filled_holes"] == 1
    assert stats["filled_pixels"] == 48


def test_depth_hole_fill_leaves_large_hole_untouched() -> None:
    depth = np.ones((32, 40), dtype=np.float32)
    depth[8:24, 10:30] = 0.0

    result, stats = MODULE.fill_small_consistent_depth_holes(
        depth,
        max_area=100,
        boundary_radius=3,
        max_relative_plane_rmse=0.001,
    )

    np.testing.assert_array_equal(result, depth)
    assert stats["filled_holes"] == 0


def test_target_depth_filter_cuts_small_depth_discontinuous_lobe() -> None:
    depth = np.zeros((16, 24), dtype=np.float32)
    depth[2:14, 2:16] = 2.0
    depth[7, 16] = 2.05
    depth[6:9, 17:20] = 2.10

    result, stats = MODULE.filter_small_target_depth_components(
        depth,
        min_area=20,
        max_log_jump=0.012,
    )

    assert np.all(result[2:14, 2:16] == 2.0)
    assert np.all(result[6:9, 17:20] == 0.0)
    assert stats["components_before"] == 3
    assert stats["components_removed"] == 2
    assert stats["pixels_removed"] == 10


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


def test_metric_surface_manifest_is_optional_and_validated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    render_manifest = tmp_path / "render_depth.json"
    render_manifest.write_text("{}", encoding="utf-8")
    metric_manifest = tmp_path / "metric_depth.json"
    metric_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--data",
            str(data),
            "--mesh-depth-manifest",
            str(render_manifest),
            "--metric-surface-depth-manifest",
            str(metric_manifest),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    args = MODULE.parse_args()

    assert args.metric_surface_depth_manifest == metric_manifest.resolve()


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


def test_masked_display_metrics_ignore_pixels_outside_surface() -> None:
    ground_truth = torch.rand(3, 16, 16)
    prediction = ground_truth.clone()
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[3:13, 4:12] = True
    prediction[:, ~mask] = 1.0 - prediction[:, ~mask]

    metrics = MODULE.masked_display_metrics(
        prediction,
        ground_truth,
        mask=mask,
        lpips_model=lambda left, right: torch.mean(torch.abs(left - right)),
    )

    assert metrics["psnr"] == pytest.approx(120.0)
    assert metrics["ssim"] == pytest.approx(1.0)
    assert metrics["lpips"] == 0.0
    assert metrics["bbox_xyxy"] == [4, 3, 12, 13]


def test_normalized_blur_does_not_darkens_constant_at_support_boundary() -> None:
    image = torch.full((3, 9, 11), 0.7)
    valid = torch.zeros(1, 9, 11, dtype=torch.bool)
    valid[:, 2:7, 3:9] = True

    blurred = MODULE.normalized_gaussian_blur(image, valid, sigma=1.5)

    torch.testing.assert_close(blurred[:, 2:7, 3:9], image[:, 2:7, 3:9], atol=1e-5, rtol=1e-5)
