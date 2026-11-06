"""CPU-only provenance/domain audit of the saved floor frame; no field queries or carving."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import ConvexHull


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    sys.path.insert(0, str(a.geometry_helper.parent))
    from actor_box_geometry import fit_floor_frame
    root = a.scene
    frame_path = root / 'two_actor_isolation/temporal_2sec_floor_v1/floor_frame.json'
    frame = json.loads(frame_path.read_text())
    witnesses = Path(frame['floor_points_source'])
    assert sha(witnesses) == frame['floor_points_source_sha256']
    points = np.asarray(frame['floor_points_prepared_world'])
    with np.load(witnesses) as data:
        assert np.array_equal(points, data['xyz'][frame['floor_ids']])
        cameras = data['cameras'].tolist()
    assert not set(cameras) & {'DC', 'HC', 'KA'}
    transform, residual = fit_floor_frame(points)
    assert np.max(np.abs(transform - np.asarray(frame['world_to_box']))) < 1e-12
    assert np.max(np.abs(residual - np.asarray(frame['fit_residuals']))) < 1e-12
    floor = points @ transform[:3, :3].T + transform[:3, 3]
    hull = ConvexHull(floor[:, :2])
    loo = []
    for index, point_id in enumerate(frame['floor_ids']):
        other, _ = fit_floor_frame(np.delete(points, index, axis=0))
        loo.append(dict(point_id=point_id, heldout_signed_residual=float(points[index] @ other[2, :3] + other[2, 3])))
    base = root / 'tuning_20261106/native_semantic_grid_20000'
    upper_path = base / 'corrected_v2/train_shoe_feature_depth.json'
    sole_path = base / 'mat_geometry_TRAIN/qualified_material_points_v2.json'
    upper = json.loads(upper_path.read_text())['points'][0]
    sole = next(row for row in json.loads(sole_path.read_text())['records'] if row['source_point_id'] == 6862)
    assert sole['label'] == 'black_shoe_sole_material'
    anchors = []
    for name, xyz in [('toe_upper', upper['xyz_floor']), ('sole6862', sole['xyz_floor'])]:
        xyz = np.asarray(xyz)
        anchors.append(dict(name=name, floor_xyz=xyz.tolist(),
                            inside_broad_floor_XY_hull=bool((hull.equations[:, :2] @ xyz[:2] + hull.equations[:, 2] <= 1e-12).all()),
                            nearest_floor_witness_XY_distance=float(np.linalg.norm(floor[:, :2] - xyz[:2], axis=1).min())))
    old_path = root / 'floor_band_prior_PROPOSAL/proposal.json'
    old = json.loads(old_path.read_text())
    raw_path = root / 'native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/raw8k_floor_sfm/manual_review_geometry.json'
    raw = json.loads(raw_path.read_text())
    raw_xyz = np.asarray(raw['safe_floor_track']['xyz'])
    raw_floor = transform[:3, :3] @ raw_xyz + transform[:3, 3]
    cell = .7 / 128
    cells = [dict(index=i, lower=-.07 + i * cell, center=-.07 + (i + .5) * cell, upper=-.07 + (i + 1) * cell) for i in [10, 12, 13]]
    out = dict(kind='FLOOR_REFERENCE_DOMAIN_DIAGNOSTIC_ONLY', units='prepared world; physical scale unspecified',
        source_bindings={str(p): sha(p) for p in [frame_path, witnesses, old_path, raw_path, a.geometry_helper, upper_path, sole_path]},
        stored_historical_helper_sha256=frame['floor_fit_helper_sha256'],
        current_fit_helper_reproduces_saved_frame_to_1e_minus12=True,
        TRAIN_cameras=cameras, floor_ids=frame['floor_ids'], excluded_ids=frame['excluded_floor_ids'],
        floor_points=floor.tolist(), fit_max_abs=float(np.abs(residual).max()),
        leave_one_witness_out=loo, leave_one_witness_out_max_abs=max(abs(row['heldout_signed_residual']) for row in loo),
        XY_support_hull=floor[hull.vertices, :2].tolist(), anchors=anchors,
        support_domain_note='Sparse TRAIN floor-covering features only. Convex hull is an interpolation footprint, not certification of continuous opaque floor or local height.',
        raw80=dict(source=raw['safe_floor_track'], floor_xyz=raw_floor.tolist(),
                   old8290_distance=raw['old8290_to_new80_3D_distance'],
                   note='Corroborates the same floor-covering fleck, not a new local spatial floor anchor.'),
        historical_finite_domain=dict(triangle_file=old['domain_triangles_file'],
            triangle_sha256=old['domain_triangles_sha256'], count=old['domain_triangles'],
            floor_ids=old['source_train_floor_ids'], domain_rule=old['domain_rule'],
            measured_points_in_domain=old['all9022_continuous_domain_audit']['count'],
            physical_depth_certificate=False, inherited_as_current_truth=False),
        relevant_128_cells=cells,
        suggested_signed_Z_bin_edges=[-.02, -.015, -.01, -frame['max_fit_residual'], 0., frame['max_fit_residual'], .01, .015, .02],
        bin_note='Coordinate sensitivity cuts, not calibrated confidence or empty-space thresholds. Split sample intervals at cuts without changing their original transmittance.',
        no_GPU=True, no_field_queries=True, no_masks_or_model_or_grid_mutation=True)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'reference_domain.json').write_text(json.dumps(out, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    polygon = floor[np.r_[hull.vertices, hull.vertices[0]], :2]
    ax.plot(polygon[:, 0], polygon[:, 1], '--', color='grey', label='support hull; not surface certificate')
    ax.scatter(floor[:, 0], floor[:, 1], c=floor[:, 2], cmap='coolwarm', s=35)
    for index, point_id in enumerate(frame['floor_ids']):
        ax.annotate(str(point_id), floor[index, :2], fontsize=7)
    ax.plot([-.84, .15, .15, -.84, -.84], [-.21, -.21, .42, .42, -.21], 'r:', label='accepted actor AABB XY')
    for item in anchors:
        x, y = item['floor_xyz'][:2]
        ax.scatter([x], [y], marker='x', color='black')
        ax.annotate(item['name'], (x, y), xytext=(8, 8 if item['name'] == 'toe_upper' else -12), textcoords='offset points', fontsize=8)
    ax.set(xlabel='floor-frame X', ylabel='floor-frame Y', title='Sparse floor support is far from both shoe anchors')
    ax.set_aspect('equal'); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(a.output / 'support_domain.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(fit_max=out['fit_max_abs'], leave_one_out_max=out['leave_one_witness_out_max_abs'], anchors=anchors)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--geometry-helper', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
