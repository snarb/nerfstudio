"""Read-only CPU source/admission and saved-profile audit; no field queries."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from marshawn_signed_height_bins import partition_weights


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    c, out = args.campaign, args.output
    assert not torch.cuda.is_initialized()
    inputs = out / 'inputs'
    manifest_path = inputs / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert sha(inputs / manifest['archive']) == manifest['archive_sha256']
    assert sha(inputs / 'visibility_review.json') == manifest['bindings']['visibility_review_sha256']
    assert sha(c / 'data/transforms.json') == manifest['bindings']['transforms_sha256']
    assert sha(c / 'data/box.json') == manifest['bindings']['box_sha256']
    for camera, digest in manifest['source_semantic_hashes'].items():
        assert sha(c / 'data/semantics' / (camera + '.npz')) == digest
    with np.load(inputs / 'prefixes.npz') as z:
        arrays = {k: z[k].copy() for k in z.files}
    assert len(arrays['point_idx']) == 84 and len(np.unique(arrays['point_idx'])) == 31
    reserves = {63, 64, 71, 72, 2219, 5556, 5568}
    assert not reserves.intersection(map(int, arrays['point_idx']))
    review = json.loads((inputs / 'visibility_review.json').read_text())
    old = json.loads((inputs / 'old18_band_uncertainty.json').read_text())
    graphs = {r['point']: r['original_feature_graph'] for r in review['points']}
    graphs.update({r['point']: r['actual_feature_recovery'] for r in old['points']})
    data = json.loads((c / 'data/transforms.json').read_text())
    frames = {r['physical_camera']: r for r in data['frames'] if r['file_path'] in data['train_filenames']}
    audit = c / 'independent_occupancy/semantic_frequency_focus_15188'
    sources = json.loads((audit / 'selected_models.json').read_text())
    box = np.asarray(manifest['aabb'])
    o, d, depth = (arrays[k] for k in ['origins', 'directions', 'witness_depth'])
    half = np.maximum(.005, .01 * depth)
    inv = 1 / np.where(np.abs(d) < 1e-12, 1., d)
    left, right = (box[0] - o) * inv, (box[1] - o) * inv
    entry = np.where(np.abs(d) < 1e-12, -np.inf, np.minimum(left, right)).max(1)
    exit = np.where(np.abs(d) < 1e-12, np.inf, np.maximum(left, right)).min(1)
    band_start = np.maximum(np.maximum(depth - half, entry + 1e-5), .01)
    band_end = np.minimum(depth + half, exit - 1e-5)
    assert np.all(band_start < depth) and np.all(depth < band_end)
    rows = []
    for i, point in enumerate(arrays['point_idx']):
        point = int(point); camera = str(arrays['camera_names'][arrays['camera_idx'][i]])
        graph = graphs[point]
        observation = next(r for r in graph['observations'] if r['camera'] == camera)
        frame = frames[camera]; pose = np.asarray(frame['transform_matrix'])
        local = (arrays['witness_xyz'][i] - pose[:3, 3]) @ pose[:3, :3]
        projected = np.array([frame['fl_x'] * -local[0] / local[2] + frame['cx'],
                              frame['fl_y'] * local[1] / local[2] + frame['cy']])
        actual = np.asarray(observation['detected_uv'])
        source_direction = np.array([(actual[0] - frame['cx']) / frame['fl_x'],
                                     -(actual[1] - frame['cy']) / frame['fl_y'], -1.]) @ pose[:3, :3].T
        source_direction /= np.linalg.norm(source_direction)
        miss = np.linalg.norm(np.cross(arrays['witness_xyz'][i] - pose[:3, 3], source_direction))
        shifts = [r['xyz_shift_world'] for r in graph['leave_one_source_view_out']]
        assert not shifts or max(shifts) < min(depth[i] - band_start[i], band_end[i] - depth[i])
        rows.append(dict(index=i, point=point, camera=camera, projected_uv=projected.tolist(),
                         detected_uv=actual.tolist(), measured_fit_hd_px=float(np.linalg.norm(projected - actual)),
                         point_to_actual_feature_ray=float(miss), source_views=len(graph['observations']),
                         available_LOO_max_shift=max(shifts) if shifts else None,
                         band_start=float(band_start[i]), band_end=float(band_end[i]), depth=float(depth[i]),
                         prefix_end=float(arrays['t_end'][i]), unsupervised_gap=float(band_start[i] - arrays['t_end'][i])))
    field_helper = c / 'semantic_frequency_focus_candidate_20261107/code/LookCloser/scripts/marshawn_semantic_admission.py'
    sys.path.insert(0, str(field_helper.parent))
    spec = importlib.util.spec_from_file_location('frozen_semantic_lookup', field_helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    admission_results = {}; masks = []
    for source in sources:
        directory = Path(source['archive'])
        for name in ['grid.npz', 'config.yml', 'spec.json']:
            assert sha(directory / name) == source['files_sha256'][name]
        with np.load(directory / 'grid.npz') as z:
            mask = z['admission'][0].copy(); bounds = z['cascade_aabbs'][0].reshape(2, 3).copy()
        masks.append(mask)
        lookup = module.AdmissionLookup(torch.from_numpy(mask), torch.from_numpy(bounds))
        center = lookup(torch.from_numpy(arrays['witness_xyz'])).numpy()
        assert center.all()
        sample_checks = {}
        for kind, start, end, counts in [('prefix', arrays['t_start'], arrays['t_end'], [64, 256, 512]),
                                        ('band', band_start, band_end, [16, 64, 256])]:
            for count in counts:
                t = start[:, None] + (end - start)[:, None] * (np.arange(count)[None] + .5) / count
                xyz = o[:, None] + d[:, None] * t[..., None]
                retained = lookup(torch.from_numpy(xyz)).numpy()
                sample_checks[kind + str(count)] = dict(total=retained.size, admitted=int(retained.sum()),
                    per_ray_fraction=retained.mean(1).tolist(), all_samples_rejected_rays=np.flatnonzero(~retained.any(1)).tolist())
        admission_results[source['name']] = dict(grid_sha256=sha(directory / 'grid.npz'), witnesses_admitted=int(center.sum()), samples=sample_checks)
    assert np.array_equal(*masks)
    queries_path = c / 'independent_occupancy/queries/clothing_103_predeclared.json'
    queries = json.loads(queries_path.read_text())['pixels']
    profile_rows = []; conserved = 0.
    for source in sources:
        directory = Path(source['archive']) / 'clothing103_trace/rays'
        for index, query in enumerate(queries):
            t = float(query['query_witness_depth']); gap = max(.01, .05 * t); h = max(.005, .01 * t)
            assert gap > .02 > h
            path = directory / ('pixel%04d_native.npz' % index)
            with np.load(path) as z:
                n = len(z['actual_weights'])
                parts, check = partition_weights(z['t_start'], z['t_end'], z['sigma'], z['actual_weights'],
                                                np.zeros(n), np.ones(n), [t - gap, t - .02, t - h, t + h])
            conserved = max(conserved, check['max_sample_weight_reconstruction_error'])
            mass = parts.sum(0)
            profile_rows.append(dict(model=source['name'], index=index, point=query['witness_id'], camera=query['camera_name'],
                group=query['diagnostic_group'], material_opaque_eligible=query['witness_id'] != 5568,
                mass_before_prefix_gap=float(mass[0]), early_mass_in_unpenalized_gap=float(mass[1]),
                remaining_gap_to_band=float(mass[2]), band_mass=float(mass[3]), after_band_mass=float(mass[4]),
                original_accumulation=float(mass.sum()), source_profile_sha256=sha(path)))
    groups = []
    for name in admission_results:
        for group in sorted({r['group'] for r in profile_rows}):
            rr = [r for r in profile_rows if r['model'] == name and r['group'] == group and r['material_opaque_eligible']]
            if not rr: continue
            groups.append(dict(model=name, group=group, rays=len(rr), **{k: float(np.mean([r[k] for r in rr]))
                for k in ['mass_before_prefix_gap', 'early_mass_in_unpenalized_gap', 'remaining_gap_to_band', 'band_mass', 'after_band_mass', 'original_accumulation']}))
    assert not torch.cuda.is_initialized()
    result = dict(CPU_only=True, model_or_density_queried=False, new_annotations_or_training=False,
        manifest_sha256=sha(manifest_path), archive_sha256=sha(inputs / 'prefixes.npz'),
        helper_sha256=sha(Path(__file__)), actual_admission_helper_path=str(field_helper), actual_admission_helper_sha256=sha(field_helper),
        queries_sha256=sha(queries_path), supervised_rays=84, supervised_points=31, reserves_excluded=sorted(reserves),
        max_source_fit_hd_px=max(r['measured_fit_hd_px'] for r in rows),
        max_source_actual_ray_distance=max(r['point_to_actual_feature_ray'] for r in rows),
        band_halfwidth_range=[float(half.min()), float(half.max())],
        unsupervised_gap_range=[min(r['unsupervised_gap'] for r in rows), max(r['unsupervised_gap'] for r in rows)],
        same_actual_admission=True, admission=admission_results, source_rows=rows, saved_profile_groups=groups,
        weight_partition_max_error=conserved,
        profile_scope='Already saved integer pixel-center rays; policy-distance partition is NOT a new query on the projected84 prior rays',
        geometry_limit='Source residual/LOO consistency is not independent confidence; finite midpoint admission samples do not certify continuous surface volume')
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'saved_pixel_profile_partitions.json').write_text(json.dumps(profile_rows, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['max_source_fit_hd_px', 'max_source_actual_ray_distance', 'band_halfwidth_range', 'unsupervised_gap_range', 'weight_partition_max_error']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
