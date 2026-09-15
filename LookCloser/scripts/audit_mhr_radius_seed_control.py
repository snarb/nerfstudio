"""Recheck all-radius certificates, admission and final62x2 native visibility.

Seed selection uses exhaustive distances, independent of the worker's KD-tree.
The established quadratic certificate and native veto mathematics are replayed,
not claimed as independently implemented. Existing depth/seed provenance is
inherited only after rehashing its completed audit inputs and inventory.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from study_multiview_face_prior import read, save, sha


def main():
    import open3d as o3d
    import probe_mhr_conic_candidates as probe
    import admit_mhr_local_patch_depth as admission
    from local_surface_certificate import certify
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); args = p.parse_args()
    root = args.root.resolve(); assert not (root / 'audit.json').exists()
    started = time.monotonic(); q = read(root / 'request.json'); result = read(root / 'admission/result.json')
    assert result['request_sha256'] == sha(root / 'request.json')
    assert q['radius'] == .003 and q['normal_dot'] == .5 and q['tolerance'] == .0005
    assert q['policy'] == 'all_radius_then_normal_filter' and not q['target_used']
    for path, digest in q['source_inputs'].items(): assert sha(path) == digest, path
    for rel, digest in result['hashes'].items(): assert sha(root / rel) == digest, rel
    assert sha(Path(__file__).with_name('run_mhr_radius_seed_control.py')) == q['script_sha256']
    assert sha(admission.__file__) == q['admission_helper_sha256']
    assert sha(Path(__file__).with_name('local_surface_certificate.py')) == q['certificate_helper_sha256']
    source = Path(q['source_candidate_root']); prior = Path(read(source / 'candidates/request.json')['final_prior_root'])
    probe.PRIOR = prior; probe.ROOT = source; probe.configure(); probe.control.configure()
    assert probe.control.binding() == q['production_base_binding']
    cq, rows, depths, masks, names, binding = admission.inputs(); del masks, names
    assert binding == q['real_binding']
    raw = o3d.io.read_triangle_mesh(str(source / 'candidates/certified_conic/local_raw.ply'))
    v, t = np.asarray(raw.vertices), np.asarray(raw.triangles)
    nt = read(source / 'candidates/certified_conic/result.json')['original_triangles']; proposals = t[nt:]
    c = np.load(source / 'admission/certified_conic/certificates.npz'); valid = c['valid_seed_mask']
    seeds = c['observed_seed_points'][valid]; normals = c['seed_normals'][valid]
    folder = root / 'admission/certified_conic'; evidence = np.load(folder / 'radius_certificates.npz')
    notes = read(folder / 'radius_certificates.json')['notes']; ids = c['query_ids']
    # NpzFile indexes decompress the entire named array. Materialize once,
    # not twice per vertex during this full-domain replay.
    query_normals = c['query_normals']
    seed_counts = evidence['seed_counts']; expected_certificate = evidence['certificate']
    np.testing.assert_array_equal(ids, evidence['query_ids']); lookup = np.zeros(len(v), bool)
    for i, qid in enumerate(ids):
        distances = np.linalg.norm(seeds - v[qid], axis=1)
        take = (distances <= .003) & ((normals @ query_normals[i]) >= .5)
        assert int(take.sum()) == int(seed_counts[i])
        accepted, note = certify(v[qid], query_normals[i], seeds[take], tolerance=.0005)
        assert accepted == bool(expected_certificate[i]) and note == notes[i], (i, qid)
        lookup[qid] = accepted
        if (i+1) % 20000 == 0: print('audit certificates', i+1, '/', len(ids), flush=True)
    old = np.load(source / 'admission/certified_conic/admission.npz'); semantic = old['semantic_ids']
    np.testing.assert_array_equal(semantic, evidence['semantic_ids'])
    keep, cert = admission.interpolation_admission(old['strict'], lookup, proposals[semantic],
        old['trusted_free'], old['mask_support'][semantic], old['mask_outside'][semantic])
    np.testing.assert_array_equal(keep, evidence['interpolated'])
    np.testing.assert_array_equal(cert, evidence['certified_prior'])
    branch = folder / 'interpolated'; retained = np.load(branch / 'evidence.npz')['retained_proposal_ids']
    assert len(np.unique(retained)) == len(retained) and np.isin(retained, semantic[keep]).all()
    mesh = o3d.io.read_triangle_mesh(str(branch / 'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(mesh.vertices), v)
    np.testing.assert_array_equal(np.asarray(mesh.triangles), np.concatenate((t[:nt], proposals[retained])))
    scene = admission.Scene2(v, np.asarray(mesh.triangles)); checked = []
    for i, (camera, depth) in enumerate(zip(rows, depths)):
        for offset in [0, .5]:
            implicated, count, raw_count = admission.measured_pixel_veto(scene, camera, depth,
                rows, depths, nt, len(mesh.triangles), offset)
            assert len(implicated) == count == 0
            checked.append(dict(camera=camera['physical_camera'], offset=offset, raw_far=raw_count))
        if (i+1) % 10 == 0: print('audit native', i+1, '/62', flush=True)
    save(root / 'audit.json', dict(status='passed', request_sha256=sha(root / 'request.json'),
        result_sha256=sha(root / 'admission/result.json'), script_sha256=sha(__file__),
        certificates_replayed=len(ids), native_checks=checked, original_prefix_exact=True,
        independent_exhaustive_seed_selection=True, certificate_and_veto_math_reused=True,
        production_accepted=False, seconds=time.monotonic()-started))
    print('audit terminal', time.monotonic()-started, flush=True)


if __name__ == '__main__': main()
