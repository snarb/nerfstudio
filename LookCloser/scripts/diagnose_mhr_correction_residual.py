"""Posthoc 001193 residual veto attribution; never supplies fitting targets."""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

import fit_mhr_silhouette_conformance as fit
from joint_temporal_texture import project
from study_multiview_face_prior import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prior, review, output = (p.resolve() for p in (args.prior, args.review, args.output))
    assert not output.exists() and len(output.parts) >= 4
    for source in (prior, review):
        assert source != output and source not in output.parents and output not in source.parents
    runtime = read(review / 'runtime_config.json')
    assert Path(runtime['arms']['candidate']) == prior
    result = read(review / 'result.json')
    for p, h in result['input_hashes'].items():
        assert sha(p) == h, p
    for p, h in result['hashes'].items():
        assert sha(review / p) == h, p
    _, rows, masks, names, evidence, validation = fit.prepare()
    data = np.load(prior / 'fit.npz')
    rays = np.load(review / 'evidence.npz')
    points = rays['candidate_points']
    assert len(points) == len(rays['portrait_xy']) == 30
    parents = np.unique(rays['candidate_faces'])
    vertices = np.unique(data['triangles'][parents])
    maximum = read(prior / 'protocol.json')['settings']['maximum_displacement']
    displacement = np.linalg.norm(data['vertices'] - data['baseline'], axis=1)
    grids, outside, records = [], [], []
    for ci, row in enumerate(rows):
        mask = masks[names.index(row['physical_camera'])].astype(bool)
        sdf = (distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float32)
        uv, z = project(points, [row]); uv, z = uv[0], z[0]
        available = (z > 0) & (uv[:, 0] > 2) & (uv[:, 0] < 1917) & (uv[:, 1] > 2) & (uv[:, 1] < 1077)
        assert available.all(), 'This small residual diagnostic expects all points in the rig field of view'
        xy = np.rint(uv).astype(int)
        bad = ~mask[xy[:, 1], xy[:, 0]]
        exact_uv, _, _ = fit.project_jacobian(points, row)
        values, _ = fit.sample_sdf(sdf, exact_uv)
        grids.append(values); outside.append(bad)
        if bad.any() or (values > 0).any():
            records.append(dict(camera=row['physical_camera'], reserved=bool(validation[ci]),
                binary_veto_count=int(bad.sum()), binary_veto_ray_ids=np.flatnonzero(bad).tolist(),
                positive_sdf_count=int((values > 0).sum()), maximum_positive_sdf=float(np.maximum(values, 0).max()),
                binary_vs_sdf_disagreement=int((bad != (values > 0)).sum())))
    outside = np.array(outside)
    np.testing.assert_array_equal(outside.sum(0), rays['candidate_point_mask_outside'])
    output.mkdir()
    np.savez_compressed(output / 'evidence.npz', portrait_xy=rays['portrait_xy'], points=points,
        camera_sdf=np.array(grids), camera_binary_veto=outside, validation=validation,
        residual_parent_ids=parents, residual_vertex_ids=vertices,
        residual_vertex_displacement=displacement[vertices])
    paths = [prior / 'fit.npz', prior / 'protocol.json', review / 'result.json',
             review / 'runtime_config.json', review / 'evidence.npz', Path(__file__),
             Path(fit.__file__), Path(__file__).with_name('joint_temporal_texture.py')]
    save(output / 'result.json', dict(cameras=records, mask_evidence=evidence,
        fitting_camera_vetoed_rays=int(outside[~validation].any(0).sum()),
        reserved_camera_vetoed_rays=int(outside[validation].any(0).sum()),
        residual_parent_ids=parents.tolist(), residual_vertex_count=len(vertices),
        residual_vertex_max_displacement=float(displacement[vertices].max()),
        residual_vertices_at_99_percent_bound=int((displacement[vertices] >= .99 * maximum).sum()),
        all_active_vertices_at_99_percent_bound=int((displacement[data['active']] >= .99 * maximum).sum()),
        posthoc_only=True, not_input_to_fit=True, no_admission_or_rgb_performed=True,
        input_hashes={str(p): sha(p) for p in paths}, evidence_sha256=sha(output / 'evidence.npz')))
    print(read(output / 'result.json'), flush=True)


if __name__ == '__main__':
    main()
