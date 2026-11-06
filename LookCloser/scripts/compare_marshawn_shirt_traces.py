"""Compare fixed trained-field shirt traces, with prefix fit/reserve separation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RESERVED = {63, 64, 71, 72}
EXCLUDED = {66}


def masses(folder, row, profile):
    source = row['profiles'][profile]
    with np.load(folder/source['trace']) as z:
        mid = (z['t_start'].reshape(-1)+z['t_end'].reshape(-1))*.5
        weight = z['weights'].reshape(-1)
    point = np.asarray(row['pixel']['witness_xyz'])
    origin, direction = np.asarray(row['origin']), np.asarray(row['direction'])
    depth = float((point-origin)@direction)
    value = [float(weight[mid < depth-.02].sum()),
             float(weight[np.abs(mid-depth) <= .02].sum()),
             float(weight[mid > depth+.02].sum()), max(0., 1.-float(weight.sum()))]
    if abs(sum(value)-1.) > 1e-5:
        raise ValueError('Termination mass/transmission do not conserve probability')
    return value, mid, np.cumsum(weight), depth


def compare(traces, output):
    output.mkdir(parents=True, exist_ok=False)
    reports, plotted, identifiers = {}, {}, None
    matched_step = None
    for name, directory in traces.items():
        folder = Path(directory)
        audit = json.loads((folder/'audit.json').read_text())
        if matched_step is None:
            matched_step = int(audit['step'])
        if int(audit['step']) != matched_step:
            raise ValueError('Checkpoint steps differ; matched-step comparison required')
        ids = [(x['pixel']['witness_id'], x['pixel']['camera_name'], x['pixel']['x'], x['pixel']['y'])
               for x in audit['rows']]
        if identifiers is None:
            identifiers = ids
        if ids != identifiers or len(ids) != 28:
            raise ValueError('Matched28 original point-camera diagnostic rays required')
        rows = []
        for row in audit['rows']:
            pixel = row['pixel']; point = pixel['witness_id']
            record = dict(point=point, camera=pixel['camera_name'],
                group='reserved_correlated_train' if point in RESERVED else
                      'excluded_duplicate' if point in EXCLUDED else 'prefix_fitted',
                masses={mode:masses(folder, row, mode)[0] for mode in ('native','uniform','uniform2x')},
                original_uniform_convergence=row['uniform_convergence'])
            record['max_mass_component_8192_to_16384_delta'] = float(np.max(np.abs(
                np.array(record['masses']['uniform'])-record['masses']['uniform2x'])))
            rows.append(record)
            if pixel['camera_name'] == 'AA' and point in (60, 65, 63, 71):
                plotted[(name, point)] = masses(folder, row, 'uniform2x')
        groups = {}
        for group in ('all', 'prefix_fitted', 'reserved_correlated_train', 'excluded_duplicate'):
            take = rows if group == 'all' else [r for r in rows if r['group'] == group]
            groups[group] = dict(rays=len(take),
                mean_mass={mode:np.mean([r['masses'][mode] for r in take], axis=0).tolist()
                           for mode in ('native', 'uniform', 'uniform2x')},
                max_mass_component_convergence_delta=max(r['max_mass_component_8192_to_16384_delta'] for r in take))
        summary = json.loads((folder/'diagnostic_summary.json').read_text())
        reports[name] = dict(groups=groups, rows=rows, source=str(folder),
            audit_sha256=hashlib.sha256((folder/'audit.json').read_bytes()).hexdigest(),
            converged_rgb_opacity_rays=summary['converged_rays'],
            coordinate_chain=json.loads((folder.parent/'coordinate_chain.json').read_text()),
            provenance=json.loads((folder.parent/'subcell_density.json').read_text()))
    result = dict(kind='MATCHED_STEP_TRAIN_SHIRT_DIAGNOSTIC', step=matched_step, models=reports,
        mass_order=['before_shirt_band', 'in_shirt_band', 'after_shirt_band', 'transmission'],
        band_halfwidth=.02, units='prepared world, not certified meters/uncertainty',
        prefix_reserved_points=sorted(RESERVED), prefix_excluded_duplicate=sorted(EXCLUDED),
        caveat='All are TRAIN RGB rays. Reserve is local correlated evidence excluded from prefix labels only; native20k supervised all28 via geometry.',
        no_training_or_production_mutation=True)
    (output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, point in zip(axes.flat, (60, 65, 63, 71)):
        depth = None
        for name in traces:
            _, mid, cdf, depth = plotted[(name, point)]
            ax.plot(mid, cdf, label=name, linewidth=1.6)
        ax.axvspan(depth-.02, depth+.02, color='magenta', alpha=.16)
        ax.axvline(depth, color='magenta', linestyle='--', linewidth=1)
        ax.set(title=f"AA point{point} / {'reserved' if point in RESERVED else 'original prefix subset'}",
               xlabel='World-distance along actual unit camera ray', ylabel='Cumulative termination mass', ylim=(0,1.02))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'Matched step {matched_step}: dense integration with 16384 samples\nMagenta: triangulated shirt ±.02; final deficit is transmission')
    fig.savefig(output/'comparison.png', dpi=150);plt.close(fig)
    print(json.dumps({k:v['groups'] for k,v in reports.items()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', action='append', required=True, help='Name=/path/to/trace/rays')
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args();compare(dict(item.split('=', 1) for item in a.trace), a.output)
