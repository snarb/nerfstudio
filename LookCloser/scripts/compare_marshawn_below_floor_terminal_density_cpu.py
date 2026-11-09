"""Compare actual native point densities without treating them as opacity GT."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    import numpy as np
    names = ['source126316', 'production_control0', 'production_prior1e4', 'production_prior1e3']
    data = {}
    receipts = {}
    for name in names:
        folder = root / 'outputs' / name
        r = json.loads((folder / 'complete.json').read_text())
        assert r['passed'] and r['state_unchanged'] and r['actual_native_density_fn']
        assert not r['optimizer_created'] and r['optimizer_updates'] == 0
        assert sha(folder / 'native_density.npz') == r['results_sha256']
        with np.load(folder / 'native_density.npz', allow_pickle=False) as z:
            data[name] = dict(xyz=z['xyz_float32'].copy(), sigma=z['sigma'].copy())
        receipts[name] = dict(sha256=sha(folder / 'complete.json'), step=r['source_step'],
            checkpoint_sha256=r['checkpoint_sha256'], pipeline_digest=r['pipeline_digest'])
    xyz = data[names[0]]['xyz']
    assert xyz.shape == (116, 3) and all(np.array_equal(v['xyz'], xyz) for v in data.values())
    actor_indices = np.unique(xyz[7:], axis=0, return_index=True)[1] + 7
    assert len(actor_indices) == 41
    provenance = json.loads((root / 'point_provenance.json').read_text())
    rows = []
    for i in range(116):
        values = {k: float(v['sigma'][i]) for k, v in data.items()}
        rows.append(dict(index=i, cohort='fixed_low_cell' if i < 7 else 'actor_observation',
            identity=provenance['fixed_low_cells'][i]['cell'] if i < 7 else provenance['actor_observations_original'][i-7],
            xyz_float32=xyz[i].tolist(), actual_native_sigma=values,
            ratio_to_matched_control={k: v / values['production_control0'] if values['production_control0'] else None
                for k, v in values.items()}))
    summaries = {}
    for name, z in data.items():
        summaries[name] = {}
        for cohort, idx in [('seven_low_cells', np.arange(7)), ('actor_unique41', actor_indices)]:
            v = z['sigma'][idx].astype(np.float64)
            c = data['production_control0']['sigma'][idx].astype(np.float64)
            summaries[name][cohort] = dict(count=len(idx), minimum=float(v.min()), median=float(np.median(v)),
                maximum=float(v.max()), mean=float(v.mean()), mean_signed_delta_vs_control=float((v-c).mean()),
                max_abs_delta_vs_control=float(np.abs(v-c).max()), exactly_zero=int((v == 0).sum()))
    report = dict(CPU_only=True, receipts=receipts, rows=rows, summary=summaries,
        source_provenance_sha256=sha(root / 'point_provenance.json'),
        all_xyz_exactly_equal=True, actor_observations=109, unique_actor_positions=41,
        terminal_steps_equal=len({receipts[n]['step'] for n in names[1:]}) == 1,
        quality_promotion=False, physical_free_space_claim=False,
        limitations=['Point sigma is not rendered opacity, visibility, or optical alpha ground truth.',
            'Sparse actor witness coordinates do not exhaust hair/shoe/body geometry and may have depth uncertainty.',
            'Different terminal steps are explicit; no equal-iteration causal claim.',
            'Actual fields are different from the immutable126316 source; original semantic certificates are not transferred.',
            'Measured training repeatability noise remains relevant even with exact restore starts.'])
    out = root / 'CPU_comparison'
    out.mkdir(exist_ok=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), summary=summaries, quality_promotion=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
