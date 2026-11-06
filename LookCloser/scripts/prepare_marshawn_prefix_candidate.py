"""Prepare the separately reviewed opaque-lettering prefix experiment; no training."""
import argparse
import json
from pathlib import Path

import numpy as np

from marshawn_softplus_prefix_candidate import sha256, validate_geometry


HOLDOUT = [63, 64, 71, 72]
DUPLICATE_EXCLUDED = [66]  # Same rounded AA/AB pixels as65, world gap.0002413.


def prepare(campaign, output):
    campaign, output = Path(campaign), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root, data = campaign.parent, campaign/'data'
    query = campaign/'independent_occupancy/queries/shirt_queries_with_depth.json'
    tracks_file = root/'geometry/strict_tracks.json'
    witnesses_file = root/'geometry/train_witnesses.npz'
    original = root/'native_png_radialK4_COLD_STAGED/geometry/scene_loose.npz'
    original_manifest = original.parent/'manifest.json'
    q, tracks, ready, box = [json.loads(p.read_text()) for p in
        [query, tracks_file, data/'ready.json', data/'box.json']]
    original_meta = json.loads(original_manifest.read_text())
    row = next(x for x in original_meta['variants'] if x['name'] == 'scene_loose')
    if sha256(original) != row['sha256']:
        raise ValueError('Original native geometry archive changed')
    for key, file in [('tracks_sha256', tracks_file), ('points_sha256', witnesses_file)]:
        if original_meta['bindings'][key] != sha256(file):
            raise ValueError('Original measured geometry provenance changed')
    with np.load(original, allow_pickle=False) as z:
        old = {k: z[k].copy() for k in z.files}
    names = old['camera_names'].astype(str).tolist()
    train = sorted(r['camera'] for r in ready['records'] if Path(r['image']).name.startswith('train_'))
    if names != train or set(names) & {'DC', 'HC', 'KA'}:
        raise ValueError('Original source must be exactly current63 TRAIN cameras')
    T, bounds = np.array(box['world_to_box']), np.array(box['aabb'])
    if not np.allclose(T[:3, :3].T@T[:3, :3], np.eye(3), atol=1e-10) or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError('Only rigid world-to-floor conversion preserves distance policy')
    records = {r['camera']: r for r in ready['records']}
    transforms = json.loads((data/'transforms.json').read_text())
    camera_rows = {Path(f['file_path']).stem.split('_')[-1]: f for f in transforms['frames']}
    maximum_origin_error, maximum_projection_error = 0., 0.
    indices, supervised, observations = [], [], []
    for p in q['pixels']:
        point, cam = p['witness_id'], p['camera_name']
        ci = names.index(cam)
        if ci not in tracks['camera_ids'][point] or len(set(tracks['camera_ids'][point])) < 3:
            raise ValueError('Observation absent from original >=3view track graph')
        found = np.flatnonzero((old['point_idx'] == point) & (old['camera_idx'] == ci))
        if len(found) != 1:
            raise ValueError('Expected exact original point-camera observation')
        i = int(found[0]); xyz = T[:3, :3]@old['witness_xyz'][i]+T[:3, 3]
        if not np.allclose(xyz, p['witness_xyz'], rtol=0, atol=1e-7):
            raise ValueError('Accepted floor witness chain mismatch')
        current = camera_rows[cam]; pose = np.array(current['transform_matrix'])
        origin_error = float(np.max(np.abs(pose[:3, 3]-(T[:3, :3]@old['origins'][i]+T[:3, 3]))))
        local = pose[:3, :3].T@(xyz-pose[:3, 3])
        uv = np.array([current['fl_x']*local[0]/(-local[2])+current['cx'],
                       -current['fl_y']*local[1]/(-local[2])+current['cy']])
        projection_error = float(np.max(np.abs(uv-old['uv'][i])))
        if origin_error > 1e-6 or projection_error > 1e-3 or local[2] >= 0:
            raise ValueError('Current TRAIN camera origin/projection differs from original visible ray')
        maximum_origin_error = max(maximum_origin_error, origin_error)
        maximum_projection_error = max(maximum_projection_error, projection_error)
        row = records[cam]; image = data/row['image']; semantic = data/'semantics'/f'{cam}.npz'
        if sha256(image) != row['rgb_sha256'] or sha256(semantic) != row['semantic_sha256']:
            raise ValueError('Reviewed current source changed')
        with np.load(semantic) as z:
            if not z['black'][p['y'], p['x']] or z['grey'][p['y'], p['x']]:
                raise ValueError('Reviewed shirt ray no longer selects only black actor')
        status = 'reserved_local_diagnostic' if point in HOLDOUT else (
                 'duplicate_excluded_not_independent' if point in DUPLICATE_EXCLUDED else 'prefix_supervision')
        observations.append(dict(point=point, camera=cam, original_index=i, status=status,
            source_track_views=len(set(tracks['camera_ids'][point])),
            original_projected_uv=old['uv'][i].tolist(), diagnostic_pixel=[p['x'], p['y']]))
        if status == 'prefix_supervision':
            indices.append(i); supervised.append(p)
    ids = np.array(indices)
    origins = old['origins'][ids]@T[:3, :3].T+T[:3, 3]
    directions = old['directions'][ids]@T[:3, :3].T
    depth = old['witness_depth'][ids]
    parallel = np.abs(directions) < 1e-12
    inv = 1./np.where(parallel, 1., directions)
    a, b = (bounds[0]-origins)*inv, (bounds[1]-origins)*inv
    enter = np.where(parallel, -np.inf, np.minimum(a, b)).max(1)
    leave = np.where(parallel, np.inf, np.maximum(a, b)).min(1)
    # Small inward distance protects strict native field selectors after FP32 cast.
    start = np.maximum(np.maximum(enter+1e-5, old['t_start'][ids]), .01)
    end = np.minimum(depth-np.maximum(.01, .05*depth), leave-1e-5)
    arrays = dict(origins=origins, directions=directions,
        witness_xyz=origins+directions*depth[:, None], witness_depth=depth,
        t_start=start, t_end=end, point_idx=old['point_idx'][ids],
        camera_idx=old['camera_idx'][ids], camera_names=np.array(names),
        source_track_view_count=np.array([len(set(tracks['camera_ids'][int(p)]))
                                         for p in old['point_idx'][ids]], dtype=np.int64))
    validate_geometry(arrays, bounds, names)
    # Do not double-weight a source pixel shared by separate near-duplicate tracks.
    pixel_keys = [(p['camera_name'], p['x'], p['y']) for p in supervised]
    if len(set(pixel_keys)) != len(pixel_keys):
        raise ValueError('Duplicate supervised image pixels')
    np.savez_compressed(output/'prefixes.npz', **arrays)
    evidence = [campaign/'independent_occupancy/full_baseline_r0_006000/missed_upper_back_witnesses.png',
                campaign/'independent_occupancy/queries/AD_lettering_review.png']
    review = dict(kind='MANUAL_OPAQUE_TRAIN_LETTERING_VISIBILITY_REVIEW',
        evidence=[dict(path=str(p), sha256=sha256(p)) for p in evidence],
        observations=observations, prior_points=sorted(set(arrays['point_idx'].tolist())),
        reserved_points=HOLDOUT, duplicate_excluded=DUPLICATE_EXCLUDED,
        findings=['All28 original diagnostic pixels lie on opaque lettering of black upper-back shirt.',
            'Hair is above the reviewed lettering; no floor, bag, silhouette or person overlap labels.',
            '63/64 and71/72 are spatially correlated groups kept entirely outside prior.',
            '65/66 share rounded pixels; exclude66 instead of doubling its supervision.',
            'Reserved points remain nearby correlated TRAIN evidence, not independent scene depth truth.'],
        additional_healthy_control_clothing_rays='None newly admitted without manual qualification',
        original_archive_sha256=sha256(original), original_query_sha256=sha256(query))
    review.update(maximum_current_camera_origin_error=maximum_origin_error,
                  maximum_original_projected_uv_error=maximum_projection_error)
    (output/'visibility_review.json').write_text(json.dumps(review, indent=2)+'\n')
    semantic_hashes = {cam: records[cam]['semantic_sha256'] for cam in sorted({p['camera_name'] for p in supervised})}
    import hashlib
    bindings = dict(transforms_sha256=sha256(data/'transforms.json'), box_sha256=sha256(data/'box.json'),
        tracks_sha256=sha256(tracks_file), witnesses_sha256=sha256(witnesses_file),
        semantic_source_sha256=hashlib.sha256(json.dumps(semantic_hashes, sort_keys=True).encode()).hexdigest(),
        visibility_review_sha256=sha256(output/'visibility_review.json'))
    manifest = dict(kind='reviewed_train_clothing_visible_prefixes', coordinate_system='accepted_floor_world',
        uses_eval_supervision=False, original_observed_track_graph_only=True,
        manual_clothing_visibility_review=True, distance_scale=1., aabb=bounds.tolist(),
        bindings=bindings, archive='prefixes.npz', archive_sha256=sha256(output/'prefixes.npz'),
        supervised_rays=len(ids), supervised_points=len(set(arrays['point_idx'].tolist())),
        reserved_rays=8, reserved_points=HOLDOUT, duplicate_excluded=DUPLICATE_EXCLUDED,
        source_semantic_hashes=semantic_hashes, sources=dict(query=str(query), tracks=str(tracks_file),
        witnesses=str(witnesses_file), original_archive=str(original)), builder_sha256=sha256(__file__),
        approved_for_training=False, pending=['native AMP zero-update gate', 'root review'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), supervised_rays=len(ids), supervised_points=manifest['supervised_points'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); prepare(a.campaign, a.output)
