"""Posthoc fixed barycentric-cohort comparison of terminal 001193 controls."""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
import fit_mhr_silhouette_conformance as fit
from mhr_surface_silhouette import association
from study_multiview_face_prior import read, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    output = args.output.resolve()
    assert not output.exists() and len(output.parts) >= 4
    roots = dict(vertex16=Path('/mnt/data/dec5_mhr_silhouette_weight16'),
                 vertex32=Path('/mnt/data/dec5_mhr_sampling_vertex32'),
                 surface=Path('/mnt/data/dec5_mhr_sampling_surface'))
    data, bindings = {}, {}
    for arm, root in roots.items():
        r = read(root / 'result.json')
        assert r['protocol_sha256'] == sha(root / 'protocol.json')
        assert read(root / 'protocol.json')['frame'] == '001193'
        assert sha(root / 'fit.npz') == r['hashes']['fit.npz']
        data[arm] = np.load(root / 'fit.npz')
        for name in ['result.json', 'protocol.json', 'fit.npz']:
            bindings[str(root / name)] = sha(root / name)
    for d in data.values():
        for key in ['triangles', 'neutral', 'active']:
            np.testing.assert_array_equal(d[key], data['vertex16'][key])
    mapping = association(data['vertex16']['triangles'], data['vertex16']['active'])
    _, rows, masks, names, evidence, validation = fit.prepare()
    sdfs = []
    for row in rows:
        mask = masks[names.index(row['physical_camera'])].astype(bool)
        sdfs.append((distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float32))
    arrays = dict(validation=validation)
    grids = {}
    for arm, d in data.items():
        grid = np.full((len(rows), mapping.shape[0]), np.nan)
        for ci, (ids, value, _) in enumerate(fit.silhouette_samples(mapping @ d['vertices'], rows, sdfs)):
            grid[ci, ids] = value
        grids[arm] = grid; arrays[arm] = grid
    common = np.logical_and.reduce([np.isfinite(g) for g in grids.values()])
    arrays['common'] = common
    records = {}
    for split, selected in [('fit', ~validation), ('reserved', validation)]:
        take = common & selected[:, None]
        assert take.any()
        records[split] = dict(samples=int(take.sum()), arms={
            arm: dict(outside=int((g[take] > 0).sum()),
                      mean_positive_sdf=float(np.maximum(g[take], 0).mean()),
                      maximum_positive_sdf=float(np.maximum(g[take], 0).max()))
            for arm, g in grids.items()})
    output.mkdir()
    np.savez_compressed(output / 'evidence.npz', **arrays)
    for path in [Path(__file__), Path(fit.__file__), Path(__file__).with_name('mhr_surface_silhouette.py')]:
        bindings[str(path)] = sha(path)
    save(output / 'result.json', dict(records=records, input_hashes=bindings, mask_evidence=evidence,
        surface_samples=mapping.shape[0], fixed_cohort=True, posthoc_not_input_to_fit=True,
        no_image_metrics_or_production_admission=True, evidence_sha256=sha(output / 'evidence.npz')))
    print(records, flush=True)


if __name__ == '__main__':
    main()
