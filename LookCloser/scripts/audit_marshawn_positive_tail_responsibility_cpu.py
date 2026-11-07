"""Read-only 95% positive-visible-prefix responsibility prior, never native weights.

The view changes class-learning responsibility, not sigma, physical rendering,
labels, or source geometry. Removed tails stay unsupervised, never negative.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_marshawn_semantic_residual_exposure_cpu import nodes
from prepare_marshawn_color_variants import sha


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def positive_visible_prefix(weight, ray_indices, target, confidence, fraction=.95):
    """Fractional ordered prefix with per-ray A preserved; no sample filtering."""
    w = np.asarray(weight, np.float64)
    ids = np.asarray(ray_indices, np.int64)
    target, confidence = np.asarray(target), np.asarray(confidence)
    count = len(target)
    if not 0 < fraction <= 1 or len(w) != len(ids) or (w < 0).any():
        raise ValueError("Invalid fraction or weights")
    if ids.size and ((ids < 0).any() or (ids >= count).any() or (ids[1:] < ids[:-1]).any()):
        raise ValueError("Packed rays must be ordered")
    alpha = np.bincount(ids, weights=w, minlength=count)
    prefix = np.cumsum(w, dtype=np.float64)-w
    if ids.size:
        first = np.r_[True, ids[1:] != ids[:-1]]
        base = np.zeros(count)
        base[ids[first]] = prefix[first]
        prefix -= base[ids]
    active = (target == 1) & (confidence > 0) & (alpha > 0)
    retained = np.minimum(w, np.maximum(fraction*alpha[ids]-prefix, 0.))
    retained[~active[ids]] = w[~active[ids]]
    retained_alpha = np.bincount(ids, weights=retained, minlength=count)
    scale = np.ones(count)
    scale[active] = alpha[active]/retained_alpha[active]
    view = retained*scale[ids]
    view_alpha = np.bincount(ids, weights=view, minlength=count)
    if np.max(abs(view_alpha-alpha), initial=0) > 1e-11:
        raise ValueError("A preservation failed")
    if not np.array_equal(view[~active[ids]], w[~active[ids]]):
        raise ValueError("Negative or UNKNOWN responsibility changed")
    return view, retained, alpha, active, prefix


def self_test():
    ids = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    w = np.array([.6, .3, .1, .4, .2, .2, .3, 0.])
    target, confidence = np.array([1., 1., 0., 1.]), np.array([1., 0., 1., 1.])
    view, retained, alpha, active, _ = positive_visible_prefix(w, ids, target, confidence)
    np.testing.assert_allclose(retained[:3], [.6, .3, .05], atol=1e-15)
    np.testing.assert_allclose(view[:3], np.array([.6, .3, .05])/.95, atol=1e-15)
    assert np.array_equal(view[3:], w[3:]) and active.tolist() == [True, False, False, False]
    for probability in [0., .01, .37, 1.]:
        np.testing.assert_allclose(np.bincount(ids, weights=view*probability), alpha*probability, atol=1e-15)
    one = positive_visible_prefix(np.array([1.]), np.array([0]), [1], [1])[0]
    np.testing.assert_allclose(one, [1.], atol=2e-16, rtol=0)
    return dict(passed=True, fractional_boundary=True, zero_alpha=True, negative_UNKNOWN_unchanged=True,
        constant_probability_identity=True, single_sample_fractional_rescaling_identity=True)


def audit(campaign, output):
    output.mkdir(parents=True, exist_ok=False)
    tests = self_test()
    capture_path = campaign/"frozen_LC_semantic_capture_plan_v4/capture/complete.json"
    contract_path = campaign/"frozen_LC_semantic_head_plan_v1/head_input_contract.json"
    low_path = campaign/"low_island_source_support_CPU_v1/support.json"
    bag_path = campaign/"frozen_LC_semantic_physical_plan_v3/prior001/renders/DC_trace_1097_572.npz"
    witnesses_path = campaign/"semantic_cell_candidates_cpu_v1/prior001_fresh2064/hard_theta05/witnesses.json"
    cohort_path = campaign/"frozen_native_semantic_capacity_CPU_v1/capacity_and_density.json"
    qualification_path = campaign/"native_actor_bag_projection_review_CPU_v1/root_LA_two_UNKNOWN_qualification.json"
    capture, contract = read(capture_path), read(contract_path)
    bounds = np.asarray(contract["bounds"], np.float32)
    low = np.asarray(read(low_path)["point"], np.float32)
    with np.load(bag_path) as z:
        bag_index = int(z["hard_theta05_selected_weight"].argmax())
        bag = z["xyz"][bag_index].copy()
    witnesses = read(witnesses_path)
    witness_xyz = np.array([r["witness_xyz"] for r in witnesses["records"]])
    unique_xyz, inverse = np.unique(witness_xyz, axis=0, return_inverse=True)
    assert len(witness_xyz) == 109 and len(unique_xyz) == 41
    points = [low, bag]+[x for x in unique_xyz]
    kernel_indices, kernel_values = [], []
    for point in points:
        lookup = {}
        for index, basis in nodes(point[None], bounds):
            lookup[int(index[0])] = lookup.get(int(index[0]), 0.)+float(basis[0])
        kernel_indices.append(lookup)
    union = sorted(set().union(*(set(k) for k in kernel_indices)))
    lut = np.full(128**3, -1, np.int32)
    lut[union] = np.arange(len(union))
    kernels = np.array([[k.get(node, 0.) for k in kernel_indices] for node in union])
    point_totals = np.zeros((2, 2, len(points)))  # before/after, positive/negative, point
    source = {str(p): sha(p) for p in [capture_path, contract_path, low_path, bag_path,
        witnesses_path, cohort_path, qualification_path, Path(__file__)]}
    cohort = {(r["camera"], *r["pixel"]): r for r in read(cohort_path)["bad_positives"]}
    assert len(cohort) == 136
    capacity_rows, point_query_rows, camera_rows = [], [], []
    total = dict(rays=0, positive_active=0, positive_affected=0, UNKNOWN=0, samples=0,
        original_positive_mass=0., removed_tail_mass_before_rescale=0., rescaled_positive_mass=0.,
        fully_unsupervised_tail_samples=0, fractional_boundary_samples=0)
    offset = 0
    special_ids = {66502, 66665, 66496}
    special = []
    profiles = {}
    frames = {f["physical_camera"]: f for f in read(campaign/"data/transforms.json")["frames"]}
    for rec in capture["records"]:
        n = rec["rays"]
        if not n:
            camera_rows.append(dict(camera=rec["camera"], rays=0, unchanged_UNKNOWN_camera=True))
            continue
        path = Path(rec["archive"])
        assert sha(path) == rec["archive_sha256"]
        source[str(path)] = rec["archive_sha256"]
        with np.load(path) as z:
            data = {k: z[k] for k in ["xyz", "weight", "ray_indices", "semantic_target", "label_weight", "pixel_xy", "starts", "ends", "rgb", "native_rgb", "native_alpha", "cap_hit"]}
        xyz, w, ids = data["xyz"], data["weight"].astype(np.float64), data["ray_indices"]
        target, confidence = data["semantic_target"], data["label_weight"].copy()
        for unknown in read(qualification_path)["unknown_queries"]:
            if unknown["camera"] == rec["camera"]:
                q = unknown["camera_ray_index"]
                assert offset+q == unknown["global_query_index"] and data["pixel_xy"][q].tolist() == unknown["pixel_xy"]
                confidence[q] = 0  # authorized existing four-UNKNOWN analysis overlay only
        view, retained, alpha, active, prefix = positive_visible_prefix(w, ids, target, confidence)
        inside = ((xyz >= bounds[0]) & (xyz <= bounds[1])).all(-1)
        cap0 = np.bincount(ids, weights=w*inside, minlength=n)
        cap1 = np.bincount(ids, weights=view*inside, minlength=n)
        exp0, exp1 = np.zeros((n, len(points))), np.zeros((n, len(points)))
        overlaps = np.zeros((len(xyz), 2))
        for index, basis in nodes(xyz, bounds):
            match = lut[index]
            hit = (match >= 0) & inside
            selected = kernels[match[hit]]*basis[hit, None]
            np.add.at(exp0, ids[hit], selected*w[hit, None])
            np.add.at(exp1, ids[hit], selected*view[hit, None])
            overlaps[hit] += selected[:, :2]
        for kind in [0, 1]:
            category = target == (1-kind)
            factor = confidence*category/(60*n)
            point_totals[0, kind] += factor @ exp0
            point_totals[1, kind] += factor @ exp1
        for j, label in enumerate(["isolated_low_point", "DC_bag_sample22"]):
            for q in np.flatnonzero((exp0[:, j] > 1e-12) | (exp1[:, j] > 1e-12)):
                point_query_rows.append(dict(point=label, camera=rec["camera"], local_query=int(q),
                    global_query=offset+int(q), pixel_xy=data["pixel_xy"][q].tolist(), target=float(target[q]),
                    effective_confidence=float(confidence[q]), native_shared_node_exposure=float(exp0[q, j]),
                    prior_shared_node_exposure=float(exp1[q, j])))
        for q, pixel in enumerate(data["pixel_xy"]):
            key = (rec["camera"], *pixel)
            if key in cohort:
                capacity_rows.append(dict(camera=key[0], pixel=[int(v) for v in key[1:]], global_query=offset+q,
                    target=float(target[q]), effective_confidence=float(confidence[q]),
                    original_NGP_cohort_ratio=cohort[key]["ratio"], LC_native_alpha=float(alpha[q]),
                    LC_native_inside_capacity=float(cap0[q]), LC_prior_responsibility_inside_mass=float(cap1[q]),
                    LC_native_capacity_ratio=float(cap0[q]/max(alpha[q], 1e-12)),
                    LC_prior_responsibility_ratio=float(cap1[q]/max(alpha[q], 1e-12)),
                    actual_native_geometry_capacity_unchanged=True, cohort_filtered=False))
            if offset+q in special_ids:
                selected = ids == q
                ww, rr = w[selected], retained[selected]
                removed = ww-rr
                depth = .5*(data["starts"][selected]+data["ends"][selected])
                cumulative = np.cumsum(ww)/max(alpha[q], 1e-12)
                cut = int(np.searchsorted(cumulative, .95))
                rgb = data["rgb"][selected].astype(np.float64)
                source_rgb = data["native_rgb"][q]
                source_gt = np.asarray(Image.open(campaign/"data"/frames[key[0]]["file_path"]).convert("RGB"))[pixel[1], pixel[0]]/255.
                source[str(campaign/"data"/frames[key[0]]["file_path"])] = sha(campaign/"data"/frames[key[0]]["file_path"])
                def average(mass):
                    return (mass[:, None]*rgb).sum(0)/max(mass.sum(), 1e-12)
                special.append(dict(camera=key[0], global_query=offset+q, pixel=pixel.tolist(), target=float(target[q]),
                    original_confidence=float(data["label_weight"][q]), effective_confidence=float(confidence[q]),
                    native_alpha=float(alpha[q]), native_cap_hit=bool(data["cap_hit"][q]), samples=int(selected.sum()),
                    cut95_sample_index=min(cut, len(depth)-1), cut95_depth=float(depth[min(cut, len(depth)-1)]),
                    native_weight_depth_quantiles={str(t): float(depth[min(np.searchsorted(cumulative, t), len(depth)-1)]) for t in [.01, .05, .5, .9, .95, .99]},
                    first_last_sample_depth=[float(depth[0]), float(depth[-1])],
                    query_GT_RGB=source_gt.tolist(), actual_native_RGB=source_rgb.tolist(),
                    prefix_mean_sample_RGB=average(rr).tolist(), removed_tail_mean_sample_RGB=average(removed).tolist(),
                    lowpoint_exposure_before=float(exp0[q, 0]), lowpoint_exposure_after=float(exp1[q, 0]),
                    valid_source_boot_not_relabeled=True))
                profiles[str(offset+q)] = dict(depth=depth, weight=ww, responsibility_weight=view[selected],
                    cumulative_native_weight=cumulative, lowpoint_kernel_overlap=overlaps[selected, 0],
                    rgb=rgb, cut95_depth=depth[min(cut, len(depth)-1)])
        active_samples = active[ids]
        original_mass = float(alpha[active].sum())
        removed_mass = float((w-retained)[active_samples].sum())
        total["rays"] += n
        total["positive_active"] += int(active.sum())
        total["positive_affected"] += int(np.bincount(ids, weights=abs(view-w), minlength=n)[active].astype(bool).sum())
        total["UNKNOWN"] += int((confidence == 0).sum())
        total["samples"] += len(w)
        total["original_positive_mass"] += original_mass
        total["removed_tail_mass_before_rescale"] += removed_mass
        total["rescaled_positive_mass"] += float(np.bincount(ids, weights=view, minlength=n)[active].sum())
        total["fully_unsupervised_tail_samples"] += int((active_samples & (retained == 0) & (w > 0)).sum())
        total["fractional_boundary_samples"] += int((active_samples & (retained > 0) & (retained < w)).sum())
        camera_rows.append(dict(camera=rec["camera"], rays=n, active_positive_rays=int(active.sum()),
            removed_responsibility_mass=removed_mass, native_A_max_error=float(abs(alpha-data["native_alpha"]).max()),
            prior_A_max_error=float(abs(np.bincount(ids, weights=view, minlength=n)-alpha).max())))
        offset += n
        print(json.dumps(dict(camera=rec["camera"], rays=n, active_positive=int(active.sum()))), flush=True)
    assert offset == 69888 and len(capacity_rows) == 136 and {r["global_query"] for r in special} == special_ids
    assert total["UNKNOWN"] == 4
    assert abs(total["rescaled_positive_mass"]-total["original_positive_mass"]) < 1e-9
    witness_records = []
    for i, record in enumerate(witnesses["records"]):
        j = 2+int(inverse[i])
        before, after = point_totals[0, 0, j], point_totals[1, 0, j]
        witness_records.append(dict(record, witness_observation_index=i, unique_XYZ_index=int(inverse[i]),
            positive_objective_normalized_shared_node_exposure_before=float(before),
            positive_objective_normalized_shared_node_exposure_after=float(after),
            exposure_ratio=None if before <= 1e-15 else float(after/before),
            source_geometry_and_original_witness_record_unchanged=True, filtered=False))
    point_records = []
    for j, name in enumerate(["isolated_low_point", "DC_bag_sample22"]):
        point_records.append(dict(name=name, XYZ=points[j].tolist(),
            positive_exposure_before=float(point_totals[0, 0, j]), positive_exposure_after=float(point_totals[1, 0, j]),
            negative_exposure_before=float(point_totals[0, 1, j]), negative_exposure_after=float(point_totals[1, 1, j]),
            positive_ratio=float(point_totals[1, 0, j]/max(point_totals[0, 0, j], 1e-30))))
    for name, profile in profiles.items():
        np.savez_compressed(output/("ND_"+name+"_profile.npz"), **profile)
    for name, records in [("point_query_exposure", point_query_rows), ("capacity136", capacity_rows)]:
        with (output/(name+".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    for path, expected in source.items():
        assert sha(path) == expected
    result = dict(kind="POSITIVE_ONLY95_VISIBLE_PREFIX_RESPONSIBILITY_CPU_PROPOSAL", authorization=False,
        fraction=.95, class1_only=True, class0_UNKNOWN_unchanged=True, four_UNKNOWN_overlay_only=True,
        original_native_weights_sigma_RGB_physical_render_unchanged=True, original_A_fixedN_preserved=True,
        source_cache_immutable=True, no_new_labels=True, tails_are_unsupervised_not_negative=True,
        CPU_only=True, GPU_or_optimizer=False, tests=tests, totals=total, camera_checks=camera_rows,
        point_exposure=point_records, point_query_exposure=point_query_rows, ND_valid_boot_diagnostics=special,
        source_109_witness_observations=witness_records, unique_XYZ_41=len(unique_xyz),
        all136_cohort_records=capacity_rows, all136_original_geometry_capacities_unchanged=True,
        inputs_sha256=source, assumption="Explicit front-depth responsibility PRIOR. Source cumulative visible weight is not actor depth GT. Partial/multilayer hair and valid back surfaces may lose learning support; do not fit before root review.",
        constant_p_scope="Per-ray constant probability over all captured samples preserves class masses and fixedN objective. A box-restricted constant can change responsibility capacity; both native and derived inside masses are reported.")
    write(output/"report.json", result)
    print(json.dumps(dict(output=str(output), points=point_records, totals=total)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.campaign, args.output)
