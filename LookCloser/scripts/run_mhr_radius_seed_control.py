"""Opt-in whole-candidate all-radius seed control; frozen prior/depth/masks.

The only certificate-policy change is all eligible measured seeds within .003
instead of nearest24 then normal/radius filtering. No target selects queries.
Actual native free-space admission still runs before any RGB rendering.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from local_surface_certificate import certify
from study_multiview_face_prior import read, save, sha


def main():
    import open3d as o3d
    import probe_mhr_conic_candidates as probe
    import admit_mhr_local_patch_depth as admission
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    source = args.candidate_root.resolve(); out = args.output.resolve()
    assert not out.exists() and len(out.parts) >= 4
    assert source != out and source not in out.parents and out not in source.parents
    audit = read(source / 'admission/audit.json'); assert audit['status'] == 'passed'
    bindings = {str(source / 'admission/audit.json'): sha(source / 'admission/audit.json')}
    for rel, digest in audit['inventory'].items():
        path = source / 'admission' / rel; assert sha(path) == digest, path; bindings[str(path)] = digest
    # This audit records a count, not a path->digest dictionary. Its output
    # inventory is checked above; inputs/helpers are revalidated below.
    assert isinstance(audit['checked_bindings'], int) and audit['checked_bindings'] > 0
    prior = Path(read(source / 'candidates/request.json')['final_prior_root'])
    probe.PRIOR = prior; probe.ROOT = source; probe.configure(); probe.control.configure()
    proof = probe.control.binding(); assert read(source / 'request.json') == proof
    cq, rows, depths, masks, names, real_binding = admission.inputs(); del masks, names
    folder = source / 'admission/certified_conic'; c = np.load(folder / 'certificates.npz')
    meta = read(source / 'candidates/certified_conic/result.json')
    for rel, digest in meta['hashes'].items():
        path = source / 'candidates/certified_conic' / rel
        assert sha(path) == digest, path; bindings[str(path)] = digest
    mesh = o3d.io.read_triangle_mesh(str(source / 'candidates/certified_conic/local_raw.ply'))
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles); nt = meta['original_triangles']
    old = o3d.io.read_triangle_mesh(cq['source_mesh'])
    np.testing.assert_array_equal(np.asarray(old.triangles), t[:nt])
    np.testing.assert_array_equal(np.asarray(old.vertices), v[:len(old.vertices)])
    proposals = t[nt:]; a = np.load(folder / 'admission.npz'); semantic = a['semantic_ids']
    ids = c['query_ids']; normals = c['query_normals']; valid = c['valid_seed_mask']
    seeds = c['observed_seed_points'][valid]; seed_normals = c['seed_normals'][valid]
    tree = cKDTree(seeds); lookup = np.zeros(len(v), bool); notes = []; counts = []
    dest = out / 'admission/certified_conic'; dest.mkdir(parents=True)
    request = dict(source_candidate_root=str(source), production_base_binding=proof,
        real_binding=real_binding, frame='001193', source_inputs=bindings,
        policy='all_radius_then_normal_filter', radius=.003, normal_dot=.5, tolerance=.0005,
        fixed_seed_pool=True, seed_requalification_changed=False, old_surface_unchanged=True,
        target_used=False, prior_fit_changed=False, production_accepted=False,
        script_sha256=sha(__file__), certificate_helper_sha256=sha(admission.certify.__code__.co_filename),
        admission_helper_sha256=sha(admission.__file__))
    save(out / 'request.json', request); started = time.monotonic()
    for i, qid in enumerate(ids):
        candidates = np.asarray(sorted(tree.query_ball_point(v[qid], .003)), int)
        selected = candidates[(seed_normals[candidates] @ normals[i]) >= .5]
        lookup[qid], note = certify(v[qid], normals[i], seeds[selected], tolerance=.0005)
        notes.append(note); counts.append(len(selected))
        if (i + 1) % 10000 == 0:
            save(out / 'progress.json', dict(stage='certificate', queries=i+1, total=len(ids),
                seconds=time.monotonic()-started, unix_time=time.time()))
            print('certificates', i+1, '/', len(ids), flush=True)
    free = a['trusted_free']
    interpolated, certified = admission.interpolation_admission(a['strict'], lookup, proposals[semantic],
        free, a['mask_support'][semantic], a['mask_outside'][semantic])
    np.savez_compressed(dest / 'radius_certificates.npz', query_ids=ids, certificate=lookup[ids],
        seed_counts=np.asarray(counts), semantic_ids=semantic, interpolated=interpolated, certified_prior=certified)
    save(dest / 'radius_certificates.json', dict(notes=notes))
    old_c = c['certificate']; new_c = lookup[ids]
    summary = dict(queries=len(ids), old_certified=int(old_c.sum()), new_certified=int(new_c.sum()),
        newly_certified=int((new_c & ~old_c).sum()), lost_certificate=int((old_c & ~new_c).sum()),
        old_interpolated=int(a['interpolated'].sum()), new_interpolated=int(interpolated.sum()))
    print(summary, flush=True); save(out / 'progress.json', dict(stage='native_guard', **summary))
    branch = admission.native_guard(dest / 'interpolated', v, t[:nt], proposals,
        semantic[interpolated], rows, depths)
    save(out / 'admission/result.json', dict(summary=summary, branch=branch,
        request_sha256=sha(out / 'request.json'), seconds=time.monotonic()-started,
        hashes={str(path.relative_to(out)): sha(path) for path in dest.rglob('*') if path.is_file()},
        production_accepted=False))
    save(out / 'progress.json', dict(stage='terminal', unix_time=time.time(), seconds=time.monotonic()-started))
    print('terminal', time.monotonic()-started, flush=True)


if __name__ == '__main__': main()
