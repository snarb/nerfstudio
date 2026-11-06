"""Measure saved ray weight lying outside original semantic point support.

This is not a rerender: removing density would change transmittance and later
weights. It only qualifies a possible independent point-support experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from marshawn_semantic_cell_admission import WholeCellAdmission


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(source, trace_folder):
    with np.load(source, allow_pickle=False) as z:
        gate = WholeCellAdmission(z['semantic_gate_pad1'], z['aabb'])
    path = trace_folder/'audit.json'
    audit = json.loads(path.read_text())
    rows = []
    for row in audit['rows']:
        p = row['pixel']
        o, d = np.array(row['origin']), np.array(row['direction'])
        trace = trace_folder/row['profiles']['native']['trace']
        with np.load(trace, allow_pickle=False) as z:
            start, end, weights = [z[k].reshape(-1) for k in ('t_start', 't_end', 'weights')]
        mid = (start+end)*.5
        positions = o+mid[:,None]*d
        outside = ~gate.point_admitted(positions)
        result = dict(pixel={k:v for k,v in p.items() if k!='witness_xyz'}, trace_sha256=sha(trace), original_opacity=float(weights.sum()), original_weight_outside_source=float(weights[outside].sum()))
        if 'witness_xyz' in p:
            xyz = np.array(p['witness_xyz'])
            depth = float((xyz-o)@d/(d@d))
            band = np.abs(mid-depth) <= .02
            result.update(witness_retained=bool(gate.point_admitted(xyz[None])[0]), ray_at_witness_depth_retained=bool(gate.point_admitted((o+depth*d)[None])[0]), band_weight_outside_source=float(weights[outside & band].sum()), front_weight_outside_source=float(weights[outside & (mid<depth-.02)].sum()))
        rows.append(result)
    groups = {}
    for name in sorted({r['pixel'].get('diagnostic_group','unspecified') for r in rows}):
        selected = [r for r in rows if r['pixel'].get('diagnostic_group','unspecified') == name]
        groups[name] = {'rays':len(selected)}
        for key in ('original_weight_outside_source','band_weight_outside_source','front_weight_outside_source'):
            if key in selected[0]:
                groups[name][key+'_mean'] = float(np.mean([r[key] for r in selected]))
                groups[name][key+'_max'] = float(np.max([r[key] for r in selected]))
    return dict(kind='READ_ONLY_ORIGINAL_WEIGHT_OUTSIDE_SOURCE_POINT_SUPPORT', source_sha256=sha(source), trace_audit_sha256=sha(path), checkpoint_sha256=audit['checkpoint_sha256'], groups=groups, rows=rows, model_loaded=False, production_changed=False, rerender_prediction=False, physical_empty_space_certified=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = audit(a.source, a.traces)
    r['script_sha256'] = sha(__file__)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2)+'\n')
    print(json.dumps({'groups':r['groups'],'all_witnesses_retained':all(x.get('witness_retained',True) and x.get('ray_at_witness_depth_retained',True) for x in r['rows'])}, indent=2))
