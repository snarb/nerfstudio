"""Post-hoc nearest-24 versus all-radius measured-seed diagnostic.

No fitting, mesh publication, changed radius or relaxed certificate. Diagnostic
query vertices are explicitly selected from the stage-attribution first hits;
this is not a target-independent production repair or a coverage guarantee.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from local_surface_certificate import certify
from study_multiview_face_prior import read, save, sha


def main():
    import open3d as o3d
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    assert not args.output.exists()
    previous = read(args.diagnosis / 'result.json')
    for path, digest in previous['input_hashes'].items(): assert sha(path) == digest, path
    assert sha(args.diagnosis / 'evidence.npz') == previous['evidence_sha256']
    bindings = {str(path): sha(path) for path in [args.diagnosis / 'result.json', Path(__file__),
        Path(__file__).with_name('local_surface_certificate.py')]}
    results = {}
    for name, arm in previous['records'].items():
        root = Path(arm['candidate_root']); folder = root / 'admission/certified_conic'
        audit = read(root / 'admission/audit.json')
        for rel in ['certified_conic/certificates.npz', 'certified_conic/certificates.json']:
            path = root / 'admission' / rel
            assert sha(path) == audit['inventory'][rel]; bindings[str(path)] = sha(path)
        cp = folder / 'certificates.npz'; c = np.load(cp)
        v = np.asarray(o3d.io.read_triangle_mesh(str(root / 'candidates/certified_conic/local_raw.ply')).vertices)
        valid = c['valid_seed_mask']; seeds = c['observed_seed_points'][valid]
        normals = c['seed_normals'][valid]; tree = cKDTree(seeds)
        lookup = {int(q): i for i, q in enumerate(c['query_ids'])}
        queries = sorted({y['vertex'] for x in arm['semantic_first_hit'] for y in x['vertex_certificates']})
        rows = []
        for qid in queries:
            i = lookup[qid]; q = v[qid]; n = c['query_normals'][i]
            dist, nearest = tree.query(q, k=min(24, len(seeds)))
            original = nearest[(dist <= .003) & ((normals[nearest] @ n) >= .5)]
            accepted, note = certify(q, n, seeds[original], tolerance=.0005)
            assert accepted == bool(c['certificate'][i]), (name, qid)
            nearby = np.asarray(sorted(tree.query_ball_point(q, .003)), int)
            nearby = nearby[(normals[nearby] @ n) >= .5]
            after, why = certify(q, n, seeds[nearby], tolerance=.0005)
            rows.append(dict(vertex=qid, nearest24_count=len(original), all_radius_count=len(nearby),
                baseline_accepted=accepted, all_radius_accepted=after, baseline_note=note, all_radius_note=why))
        results[name] = dict(query_vertices=len(rows),
            baseline_accepted=sum(r['baseline_accepted'] for r in rows),
            all_radius_accepted=sum(r['all_radius_accepted'] for r in rows),
            recovered=sum(not r['baseline_accepted'] and r['all_radius_accepted'] for r in rows),
            lost=sum(r['baseline_accepted'] and not r['all_radius_accepted'] for r in rows), rows=rows)
        print(name, {k: v for k, v in results[name].items() if k != 'rows'}, flush=True)
    args.output.mkdir(parents=True)
    save(args.output / 'result.json', dict(records=results, input_hashes=bindings,
        posthoc_selected_queries=True, production_accepted=False, radius=.003, tolerance=.0005,
        geometry_or_render_modified=False))


if __name__ == '__main__': main()
