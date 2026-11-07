"""Stage sixteen reviewed source rays; no source model, GPU, or fit is loaded."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nerfstudio.cameras.cameras import Cameras, CameraType
from marshawn_actor_coordinates import raw_array_indices, raw_camera
from prepare_marshawn_color_variants import sha


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(campaign: Path, output: Path):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU staging must not initialize CUDA")
    output.mkdir(parents=True, exist_ok=False)
    (output / "queries").mkdir()
    positive = campaign / "native_actor_label_review_CPU_v1/qualified_review.json"
    floor = campaign / "native_actor_near_floor_review_CPU_v1/qualified_review.json"
    original = campaign / "frozen_native_semantic_CPU_v2/manifest.json"
    unknown = campaign / "frozen_native_semantic_head_technical_plan_v1/label_qualification.json"
    portable = campaign / "frozen_LC_semantic_source_silver_parity_plan_v1/source_execution_manifest.json"
    transforms = campaign / "data/transforms.json"
    paths = [positive, floor, original, unknown, portable, transforms, Path(__file__),
        Path(__file__).with_name("marshawn_actor_coordinates.py"), Path(inspect.getfile(Cameras))]
    bindings = {str(p): sha(p) for p in paths}
    for source_review in [positive, floor]:
        receipt = read(source_review)["root_manual_review"]
        assert sha(receipt["path"]) == receipt["sha256"]
        bindings[receipt["path"]] = receipt["sha256"]
    assert bindings[str(original)] == "1562bcfaf5ec5da00a840da210962ef136d521fbca0d1beb5bf30d4a9aeb30ba"
    old_queries = {r["camera"]: r for r in read(original)["records"]}
    for row in old_queries.values():
        p = Path(row["archive"])
        assert sha(p) == row["archive_sha256"]
        bindings[str(p)] = row["archive_sha256"]
    tf = read(transforms)
    frames = {f["physical_camera"]: f for f in tf["frames"]}
    approved_train = {f["physical_camera"] for f in tf["frames"] if f["file_path"] in tf["train_filenames"]}
    rows = []
    for issue in read(positive)["issues"]:
        if issue["qualification"] != "true_actor_positive_candidate":
            continue
        for i, point in enumerate(issue["pixels"], 1):
            rows.append(dict(query_id=f"extra16_{issue['id']}_{i:02d}", camera=issue["camera"],
                original_review_HD_pixel=point["HD_pixel"], capture_HD_array_xy=point["HD_pixel"],
                semantic_target=1., review_confidence=issue["confidence"], proposed_future_weight=1.,
                category="reviewed_actor_shoe" if issue["camera"] == "MD" else "hair_native_center_diagnostic",
                source_review=str(positive), source_issue=issue["id"], source_point_index=i-1,
                source_provenance={key: issue[key] for key in ["raw_source", "raw_source_sha256", "native_crop_source", "native_crop_source_sha256", "raw_archive", "raw_archive_sha256"]},
                source_RAW_footprint=point["RAW_geometric_corners"],
                original_nominal_HD_coverage=point["RAW_AREA_then_calibrated_remap_coverage"]["nominal"]))
    for point in read(floor)["records"]:
        rows.append(dict(query_id="extra16_"+point["id"], camera=point["camera"],
            original_review_HD_pixel=point["HD_pixel"], capture_HD_array_xy=point["HD_pixel"],
            semantic_target=0., review_confidence=point["confidence"], proposed_future_weight=1.,
            category="reviewed_visible_floor_class", source_review=str(floor), source_issue=point["id"],
            source_provenance={key: point[key] for key in ["raw_source", "raw_source_sha256", "native_crop_source", "native_crop_source_sha256", "raw_archive", "raw_archive_sha256"]},
            source_RAW_footprint=point["RAW_geometric_cell_corners"],
            true_actor_guard_margin_HD_pixels=point["verified_true_actor_margin_HD_pixels_conservative"]))
    assert len(rows) == 16 and sum(r["semantic_target"] for r in rows) == 4
    fd = next(r for r in rows if r["category"] == "hair_native_center_diagnostic")
    f = frames["FD"]
    K, dist = raw_camera("FD")
    target = np.array([[f["fl_x"], 0, f["cx"]-.5], [0, f["fl_y"], f["cy"]-.5], [0, 0, 1.]])
    native_xy = np.array([3637., 1629.])
    fd["capture_HD_array_xy"] = cv2.undistortPoints(native_xy.reshape(1, 1, 2), K, dist, P=target).reshape(2).tolist()
    fd["native_source_pixel_RAW_array_xy"] = native_xy.tolist()
    corners = native_xy + np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    fd["native_source_pixel_RAW_corners"] = corners.tolist()
    fd["native_source_pixel_HD_array_corners"] = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, dist, P=target).reshape(-1, 2).tolist()
    fd["proposed_future_weight"] = 0.
    fd["fit_exclusion"] = "Native-center membership does not qualify a pure HD footprint: original AREA coverage .23024653. Keep source HD camera pixel_area; FD remains diagnostic-only unless a separate native-footprint contract is reviewed."
    for i, row in enumerate(rows):
        n = row["camera"]
        assert n in approved_train and n not in {"DC", "HC", "KA"}
        f = frames[n]
        xy = np.array(row["capture_HD_array_xy"], np.float64)
        pose = np.array(f["transform_matrix"], np.float64)
        uv = xy+.5
        raw = raw_array_indices(uv[None], f)[0]
        if row is fd:
            np.testing.assert_allclose(raw, native_xy, atol=1e-6, rtol=0)
        footprint = raw_array_indices(xy + np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]), f)
        local = np.array([(uv[0]-f["cx"])/f["fl_x"], -(uv[1]-f["cy"])/f["fl_y"], -1.])
        direction = pose[:3, :3] @ local
        direction /= np.linalg.norm(direction)
        with np.load(old_queries[n]["archive"]) as old:
            assert not np.any(np.all(old["pixel_xy"] == xy, axis=1))
        row.update(standalone_row_index=i, namespace="native_actor_extra16_v1", split="TRAIN", frame=1660,
            capture_COLMAP_uv=uv.tolist(), capture_RAW_center=raw.tolist(),
            actual_HD_camera_RAW_geometric_footprint=footprint.tolist(), floor_camera_to_world=pose.tolist(),
            expected_floor_origin=pose[:3, 3].tolist(), expected_floor_direction_float64=direction.tolist(),
            capture_camera_raster=[f["w"], f["h"]], current_fit_weight=0., actual_LC_observability_qualified=False,
            semantic_class_only=True, alpha_target=None, depth_target=None, whole_ray_empty_target=False,
            camera_row=f, camera_pixel_area_policy="Unchanged source HD Cameras.generate_rays pixel_area, including FD; never silently replace it by a native RAW pixel area.")
    records = []
    max_direction_error = 0.
    for n in ["MD", "FD", "MC"]:
        selected = [r for r in rows if r["camera"] == n]
        f = frames[n]
        camera = Cameras(camera_to_worlds=torch.tensor(f["transform_matrix"], dtype=torch.float32)[None, :3],
            fx=f["fl_x"], fy=f["fl_y"], cx=f["cx"], cy=f["cy"], width=f["w"], height=f["h"], camera_type=CameraType.PERSPECTIVE)
        xy = np.array([r["capture_HD_array_xy"] for r in selected], np.float64)
        rays = camera.generate_rays(0, coords=torch.tensor(xy[:, [1, 0]]+.5, dtype=torch.float32), keep_shape=False)
        expected = np.array([r["expected_floor_direction_float64"] for r in selected])
        error = float(abs(rays.directions.numpy()-expected).max())
        assert error < 5e-7
        max_direction_error = max(max_direction_error, error)
        for j, row in enumerate(selected):
            row["CPU_native_camera_pixel_area"] = float(rays.pixel_area[j, 0])
            row["CPU_native_directions_norm"] = float(rays.metadata["directions_norm"][j, 0])
        path = output/"queries"/(n+".npz")
        np.savez_compressed(path, query_id=np.array([r["query_id"] for r in selected]),
            standalone_row_index=np.array([r["standalone_row_index"] for r in selected], np.int32),
            pixel_xy=xy, coords_yx_COLMAP=xy[:, [1, 0]]+.5,
            origins=rays.origins.numpy(), directions=rays.directions.numpy(), pixel_area=rays.pixel_area.numpy(),
            directions_norm=rays.metadata["directions_norm"].numpy(),
            semantic_target=np.array([r["semantic_target"] for r in selected], np.float32),
            label_weight=np.zeros(len(selected), np.float32),
            review_confidence=np.array([r["review_confidence"] for r in selected], np.float32),
            proposed_post_qualification_label_weight=np.array([r["proposed_future_weight"] for r in selected], np.float32),
            category=np.array([r["category"] for r in selected]), world_to_box=np.eye(4))
        records.append(dict(camera=n, rays=len(selected), archive=str(path), archive_sha256=sha(path)))
    source = read(portable)
    references = dict(source["original_source_FHD"])
    local_renders = campaign/"independent_occupancy/semantic_quarter_distortion_final_geometry/quarter_distortion_rawmin/renders"
    for name, expected in references["files_sha256"].items():
        assert sha(local_renders/name) == expected
    references["local_reference_renders"] = str(local_renders)
    observer = campaign/"frozen_LC_semantic_physical_plan_v3/code/marshawn_LC_semantic_physical_path.py"
    bindings[str(observer)] = sha(observer)
    manifest = dict(kind="STANDALONE_EXTRA16_LC_CAPTURE_CPU_PROPOSAL", authorization=False, phase="capture_only",
        source_step=126316, source_checkpoint_sha256=source["source_archive_sha256"]["checkpoint.ckpt"],
        source_pipeline_digest=source["expected_serialized_states_sha256"]["pipeline"],
        source_expected_serialized_states=source["expected_serialized_states_sha256"],
        world_to_box=np.eye(4).tolist(), bounds=source["bounds"], original_sigma=True,
        all_extra_semantic_sigma_selectors="disabled", native_max_steps_per_ray=1024, corrected_ARM_allocator=False,
        source_observer=dict(path=str(observer), sha256=sha(observer), API="capture_lc_path(model,ray_bundle,expected_cap=1024)", stable_FP64_proof=True),
        preferred_execution="Portable SG source after exact parity qualification; no SG parity is asserted by this CPU plan.",
        portable_source_manifest=dict(path=str(portable), sha256=sha(portable)),
        required_source_FHD_RGB_alpha_gate=references, records=records, rows=rows,
        original_69888_manifest=dict(path=str(original), sha256=sha(original), modified=False),
        original_two_IA_overrides=dict(path=str(unknown), sha256=sha(unknown), modified=False),
        standalone_capture_contract="Archive16 rays separately; capture query_id/source rows with original native sample arrays. Never overwrite or silently append to old69,888 archives or heads.",
        fit_policy="All current label_weight values0. Only15 reviewed class rows propose weight1 after per-ray actual-LC observability and root review; FD remains0. No fit is authorized.",
        required_outputs=["xyz", "starts", "ends", "sigma", "rgb", "weight", "ray_indices", "native_rgb", "native_alpha", "num_samples_per_ray", "cap_hit", "query_id", "pixel_area", "directions_norm", "model_digest", "world_to_box"],
        required_gates=dict(portable_source_parity_passed=False, all3_source_FHD_RGB_alpha_passed=False,
            source_states_RNG_LUT_unchanged=False, unwrapped_native_16ray_RGB_alpha_parity=False,
            native_weight_RGB_alpha_proof=False, actual_LC_per_row_observability_qualified=False, root_capture_release=False, root_fit_release=False),
        observability_review="Inspect actual native weight/depth/color/alpha/cap evidence per ray before class fitting. Visible source class is not alpha0/alpha1 or a hidden-body/whole-ray-empty label.",
        inputs_sha256=bindings, CPU_only=True, CUDA_initialized=torch.cuda.is_initialized(), GPU_launched=False,
        optimizer_created=False, original_queries_masks_heads_AABB_color_changed=False,
        CPU_validation=dict(ray_count=16, MD=7, FD=5, MC=4, targets_actor=4, targets_floor=12,
            current_nonzero_fit_weights=0, proposed_future_nonzero_weights=15, max_pinhole_direction_error=max_direction_error))
    assert manifest["source_checkpoint_sha256"] == "340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641"
    assert manifest["source_pipeline_digest"] == "c83bc7dbb1504063c35f0a2b14ae3fb329ff27ed2eb6b3661eb89b4918d46636"
    assert not torch.cuda.is_initialized()
    for path, expected in bindings.items():
        assert sha(path) == expected
    write(output/"manifest.json", manifest)
    write(output/"release_TEMPLATE.json", dict(authorization=False, capture_authorized=False, fitting_authorized=False,
        manifest_sha256=sha(output/"manifest.json"), reason="CPU proposal only; source parity and per-ray observability pending."))
    print(json.dumps(dict(output=str(output), manifest_sha256=sha(output/"manifest.json"), rays=16, fit_authorized=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.campaign, args.output)
