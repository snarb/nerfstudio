"""Fixed surface quadrature diagnostics for terminal, topology-matched 001193 priors."""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
import fit_mhr_silhouette_conformance as fit
import mhr_dense_surface_silhouette as dense
from study_multiview_face_prior import read, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior', action='append', required=True, help='LABEL=/absolute/prior/root')
    p.add_argument('--order', type=int, choices=[2,4,8], default=8)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    roots = {}; output = args.output.resolve()
    assert not output.exists() and len(output.parts) >= 4
    for item in args.prior:
        label, path = item.split('=', 1)
        assert label.isidentifier() and label not in roots and Path(path).is_absolute()
        roots[label] = Path(path).resolve()
        assert output != roots[label] and output not in roots[label].parents and roots[label] not in output.parents
    assert len(roots) >= 2
    datasets, bindings = {}, {}
    for label, root in roots.items():
        r = read(root / 'result.json'); q = read(root / 'protocol.json')
        assert q['frame'] == '001193' and r['protocol_sha256'] == sha(root / 'protocol.json')
        assert sha(root / 'fit.npz') == r['hashes']['fit.npz']
        datasets[label] = np.load(root / 'fit.npz')
        for name in ['result.json', 'protocol.json', 'fit.npz']:
            bindings[str(root / name)] = sha(root / name)
    first = next(iter(datasets.values()))
    for d in datasets.values():
        for key in ['triangles','neutral','active']: np.testing.assert_array_equal(d[key], first[key])
    mapping = dense.association(first['triangles'], first['active'], args.order)
    movable = np.asarray(mapping[:,first['active']].sum(1)).ravel() > 0
    _, rows, masks, names, evidence, validation = fit.prepare()
    sdfs = []
    for row in rows:
        mask = masks[names.index(row['physical_camera'])].astype(bool)
        sdfs.append((distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float32))
    grids = {}
    for name, data in datasets.items():
        grid = np.full((len(rows), mapping.shape[0]), np.nan)
        for ci, (ids, values, _) in enumerate(fit.silhouette_samples(mapping @ data['vertices'], rows, sdfs)):
            grid[ci,ids] = values
        grids[name] = grid
    common = np.logical_and.reduce([np.isfinite(g) for g in grids.values()]); records = {}
    for scope, selection in [('all', np.ones(mapping.shape[0], bool)), ('movable', movable)]:
        records[scope] = {}
        for split, selected in [('fit',~validation), ('reserved',validation)]:
            take = common & selected[:,None] & selection[None]
            assert take.any()
            records[scope][split] = dict(samples=int(take.sum()), arms={
                name: dict(outside=int((g[take]>0).sum()), mean_positive_sdf=float(np.maximum(g[take],0).mean()),
                           maximum_positive_sdf=float(np.maximum(g[take],0).max())) for name,g in grids.items()})
    output.mkdir()
    np.savez_compressed(output / 'evidence.npz', **grids, common=common, validation=validation, movable=movable)
    for path in [Path(__file__),Path(dense.__file__),Path(dense.base.__file__),Path(fit.__file__)]:
        bindings[str(path)] = sha(path)
    save(output / 'result.json', dict(records=records, order=args.order, samples_per_face=len(dense.barycentric_lattice(args.order)),
        surface_samples=mapping.shape[0], fixed_only_samples=int((~movable).sum()), roots={k:str(v) for k,v in roots.items()},
        input_hashes=bindings, mask_evidence=evidence, evidence_sha256=sha(output / 'evidence.npz'),
        posthoc_only=True, not_an_image_metric=True, not_an_admission_or_video_quality_gate=True))
    print(records, flush=True)


if __name__ == '__main__': main()
