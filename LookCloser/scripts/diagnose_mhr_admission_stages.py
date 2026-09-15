"""Post-hoc stage attribution on fixed rays; never supplies fitting inputs.

Compare terminal, independently audited candidates without editing any mesh or
admission threshold. First-hit proposal explanations describe that particular
triangle, not all possible surfaces along a ray.
"""
import argparse
from pathlib import Path
import numpy as np
from study_multiview_face_prior import read, save, sha


def stage_ids(count, semantic, strict, interpolated, strict_final, interpolated_final):
    semantic = np.asarray(semantic, dtype=int)
    assert len(np.unique(semantic)) == len(semantic)
    assert ((semantic >= 0) & (semantic < count)).all()
    assert np.asarray(strict).dtype == np.asarray(interpolated).dtype == bool
    assert np.shape(strict) == np.shape(interpolated) == semantic.shape
    assert (~strict | interpolated).all()
    result = dict(raw=np.arange(count), semantic=semantic,
                  strict_initial=semantic[strict], interpolated_initial=semantic[interpolated],
                  strict_final=np.asarray(strict_final), interpolated_final=np.asarray(interpolated_final))
    for name in ['strict', 'interpolated']:
        final = result[name + '_final']
        assert np.issubdtype(final.dtype, np.integer)
        assert len(np.unique(final)) == len(final)
        assert np.isin(final, result[name + '_initial']).all()
    return result


def trace(root, camera, xy, bindings):
    import open3d as o3d
    from admit_mhr_local_patch_depth import Scene2
    root = root.resolve(); admission = root / 'admission'; folder = admission / 'certified_conic'
    audit = read(admission / 'audit.json'); assert audit['status'] == 'passed'
    for rel, digest in audit['inventory'].items():
        path = admission / rel; assert sha(path) == digest, path
    bindings[str(admission / 'audit.json')] = sha(admission / 'audit.json')
    source = root / 'candidates/certified_conic'; meta = read(source / 'result.json')
    for rel, digest in meta['hashes'].items():
        path = source / rel; assert sha(path) == digest, path; bindings[str(path)] = digest
    bindings[str(source / 'result.json')] = sha(source / 'result.json')
    mesh = o3d.io.read_triangle_mesh(str(source / 'local_raw.ply'))
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    nt = meta['original_triangles']; old = t[:nt]
    proposals = np.load(source / 'proposal_evidence.npz')['proposals']
    np.testing.assert_array_equal(t[nt:], proposals)
    a = np.load(folder / 'admission.npz'); semantic = a['semantic_ids']
    stages = stage_ids(len(proposals), semantic, a['strict'], a['interpolated'],
        np.load(folder / 'strict/evidence.npz')['retained_proposal_ids'],
        np.load(folder / 'interpolated/evidence.npz')['retained_proposal_ids'])
    pose = np.asarray(camera['transform_matrix'])
    ext = np.linalg.inv(pose @ np.diag([1., -1., -1., 1.])).astype(np.float32)
    k = np.array([[camera['fl_x'], 0, camera['cx']], [0, camera['fl_y'], camera['cy']], [0, 0, 1]], np.float32)
    scene = Scene2(v, old)
    rays = scene.create_rays_pinhole(o3d.core.Tensor(k), o3d.core.Tensor(ext), camera['w'], camera['h']).numpy()
    rays = o3d.core.Tensor(np.ascontiguousarray(np.rot90(rays)[xy[:, 1], xy[:, 0]]))
    records = {}; arrays = {}
    for name, ids in dict(base=np.zeros(0, int), **stages).items():
        hit = Scene2(v, np.concatenate((old, proposals[ids]))).cast_rays(rays)
        d = hit['t_hit'].numpy(); triangle = hit['primitive_ids'].numpy().astype(np.int64)
        valid = np.isfinite(d) & (d > 0); added = valid & (triangle >= nt)
        mapped = np.full(len(xy), -1, int); mapped[added] = ids[triangle[added] - nt]
        records[name] = dict(hits=int(valid.sum()), ray_ids=np.flatnonzero(valid).tolist())
        arrays[name + '_depth'] = d; arrays[name + '_proposal'] = mapped
    probe = read(root / 'posthoc_probe/result.json')
    for name, oldname in [('base', 'production_base'), ('raw', 'raw'), ('semantic', 'semantic_only')]:
        assert records[name]['hits'] == probe['records'][oldname]['residual_hits']
    rgb = next(r for r in read(admission / 'rgb_review/result.json')['records'] if r['view'] == 'F004_E')
    assert records['interpolated_final']['hits'] == rgb['fixed_residual']['corrected_hits']
    for path in [root / 'posthoc_probe/result.json', admission / 'rgb_review/result.json']:
        bindings[str(path)] = sha(path)
    reverse = np.full(len(proposals), -1, int); reverse[semantic] = np.arange(len(semantic))
    free = a['trusted_free']; votes = a['votes']; strict = a['strict']; certified = a['certified_prior']
    certificates = np.load(folder / 'certificates.npz')
    cert_index = {int(q): i for i, q in enumerate(certificates['query_ids'])}
    cert_notes = read(folder / 'certificates.json')['notes']
    first_hit = []
    for ray, proposal in enumerate(arrays['semantic_proposal']):
        if proposal < 0: continue
        i = reverse[proposal]; assert i >= 0
        vertices = proposals[proposal]
        first_hit.append(dict(ray=ray, portrait_xy=xy[ray].tolist(), proposal=int(proposal),
            sample_votes=votes[i].tolist(), footprint_veto_cameras=np.flatnonzero(free[:, i].any(1)).tolist(),
            strict=bool(strict[i]), certified_prior=bool(certified[i]),
            survives_interpolated_initial=bool(a['interpolated'][i]),
            survives_interpolated_final=bool(np.isin(proposal, stages['interpolated_final'])),
            vertex_certificates=[dict(vertex=int(q), accepted=bool(certificates['certificate'][cert_index[int(q)]]),
                note=cert_notes[cert_index[int(q)]]) for q in vertices]))
    return dict(candidate_root=str(root), records=records, semantic_first_hit=first_hit), arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', action='append', required=True, help='LABEL=absolute_root')
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    assert not args.output.exists()
    base = Path('/mnt/data/dec5_mhr_production_patch_001193')
    ep = base / 'residual_hole/evidence.npz'
    cp = base / 'admission/rgb/F004_E/baseline/frames/001193/result.json'
    xy = np.load(ep)['portrait_xy']; camera = read(cp)['camera']
    bindings = {str(path): sha(path) for path in [ep, cp, Path(__file__),
        Path(__file__).with_name('admit_mhr_local_patch_depth.py')]}
    records = {}; arrays = dict(portrait_xy=xy)
    args.output.mkdir(parents=True)
    for item in args.candidate:
        label, root = item.split('=', 1); assert label not in records
        records[label], evidence = trace(Path(root), camera, xy, bindings)
        arrays.update({label + '_' + k: value for k, value in evidence.items()})
        print(label, {k: v['hits'] for k, v in records[label]['records'].items()}, flush=True)
    np.savez_compressed(args.output / 'evidence.npz', **arrays)
    save(args.output / 'result.json', dict(records=records, input_hashes=bindings,
        evidence_sha256=sha(args.output / 'evidence.npz'), production_accepted=False,
        posthoc_only=True, thresholds_changed=False, no_image_metrics=True))


if __name__ == '__main__': main()
