"""Replay protected field changes against native depths and verify the matched pair."""
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json
from study_protected_field_veto import ROOT
from fuse_depth_tsdf_mesh import load_depth
from prune_measured_free_surface import near_tap_evidence
from carve_patchmatch_mesh_free_space import free_space_evidence
from study_confidence_depth_prior import load_real
from review_full_block_transfer import ROOT as DEPTH_ROOT


def audit():
    raw_rows, raw_depths, raw_receipt = load_real(DEPTH_ROOT, '000995')
    raw = {r['physical_camera']: d for r, d in zip(raw_rows, raw_depths)}
    dataset = ROOT/'depth_dataset'; source = read(dataset/'transforms.json')
    physical = {str((dataset/f['depth_file_path']).resolve()): f['physical_camera']
                for f in source['frames'] if f.get('depth_file_path')}
    arrays = {}; bindings = {}; summaries = {}
    for arm in ['control', 'protected']:
        folder = ROOT/arm; q = read(folder/'request.json'); done = read(folder/'complete.json')
        assert sha(folder/'request.json') == done['request_sha256']
        for name, digest in done['hashes'].items():
            assert sha(folder/name) == digest; bindings[str(folder/name)] = digest
        for name, digest in q['geometry_inputs'].items():
            assert sha(dataset/name) == digest; bindings[str(dataset/name)] = digest
        for name, digest in q['scripts'].items(): assert sha(name) == digest; bindings[name] = digest
        meta = read(folder/'mesh.json'); r = read(folder/'field_result.json')
        a = np.load(folder/'field_evidence.npz'); arrays[arm] = a
        ids = np.flatnonzero(a['eligible']); p = a['points'][ids].astype(np.float64)
        actual = (a['tsdf_before'] < 0)&(a['weights'] > 0)&(a['near_by_camera'].sum(0) == 0)&(a['far_by_camera'].sum(0) >= 6)
        np.testing.assert_array_equal(actual, a['eligible'])
        expected = a['tsdf_before'].copy()
        if arm == 'protected': expected[ids] = 1.
        np.testing.assert_array_equal(expected, a['tsdf_after'])
        near = np.zeros(len(ids), np.uint8); far = np.zeros_like(near)
        for ci, row in enumerate(r['observations']):
            path = Path(row['depth']); assert sha(path) == row['depth_sha256']
            depth = load_depth(path, scale_factor=meta['dataparser_scale'])
            np.testing.assert_array_equal(depth, raw[physical[str(path.resolve())]])
            K = np.asarray(row['intrinsic']); E = np.asarray(row['extrinsic'])
            c = p @ E[:3, :3].T+E[:3, 3]; z = c[:, 2]
            uv = c[:, :2]/z[:, None]*np.array([K[0,0], K[1,1]])+K[:2,2]
            n = near_tap_evidence(depth, uv, z, radius=0)
            f, _ = free_space_evidence(depth, uv[:,0], uv[:,1], z)
            # Independent float64 projection and NumPy native footprint logic.
            np.testing.assert_array_equal(n, a['near_by_camera'][ci, ids])
            np.testing.assert_array_equal(f, a['far_by_camera'][ci, ids])
            near += n; far += f
        assert (near == 0).all() and (far >= 6).all()
        mesh = o3d.io.read_triangle_mesh(str(folder/'mesh.ply'))
        assert len(mesh.vertices) == meta['vertices'] and len(mesh.triangles) == meta['triangles']
        assert len(mesh.triangles) > 0 and np.isfinite(np.asarray(mesh.vertices)).all()
        _, components, _ = mesh.cluster_connected_triangles()
        summaries[arm] = dict(vertices=len(mesh.vertices), triangles=len(mesh.triangles),
            components=sorted(map(int, components), reverse=True), changed_field=len(ids) if arm=='protected' else 0,
            all_eligible_native_evidence_replayed=True, all62_imported_depth_arrays_equal_raw=True)
    # Hash-map allocation order need not agree; compare the same physical voxels.
    orders = {arm: np.lexsort(a['points'].T[::-1]) for arm, a in arrays.items()}
    a, b = arrays['control'], arrays['protected']; i, j = orders['control'], orders['protected']
    for key in ['points', 'weights', 'eligible']:
        np.testing.assert_array_equal(a[key][i], b[key][j])
    np.testing.assert_allclose(a['tsdf_before'][i], b['tsdf_before'][j], rtol=0, atol=1e-6)
    error = float(np.abs(a['tsdf_before'][i]-b['tsdf_before'][j]).max())
    atomic_json(ROOT/'field_audit.json', dict(arms=summaries, matched_pre_field_max_error=error,
        raw_receipt=raw_receipt, hashes=bindings, field_only_change_verified=True,
        production_promoted=False, raw_volume_serialized=False, visual_quality_approved=False))
    print('Both field arms replayed; max pre-field error', error, flush=True)


def seal():
    from review_measured_free_surface import VIEWS
    from review_jaw_repair_transfer import verified_image
    from render_smooth_temporal_mesh_video import verify_request
    verdict = read(ROOT/'visual_review.json')
    assert verdict['status'] == 'fail_no_material_lipstick_geometry_repair'
    assert [r['view'] for r in verdict['views']] == VIEWS
    assert all(r['status'] == 'fail' for r in verdict['views'])
    bindings = dict(read(ROOT/'field_audit.json')['hashes'])
    for arm in ['control', 'protected']:
        for view in VIEWS:
            folder = ROOT/arm/'rgb'/view; q = verify_request(folder)
            image, result = verified_image(folder, '000995')
            assert image.shape == (1920,1080,3)
            assert len(result['source_cameras']) == 62
            entry = q['inventory'][0]
            for key in ['mesh', 'metadata']:
                assert sha(entry[key]) == entry[key+'_sha256']; bindings[entry[key]] = entry[key+'_sha256']
            for name, digest in q['script_hashes'].items():
                bindings[str(Path(__file__).with_name(name))] = digest
    for row in read(ROOT/'review_result.json')['records']:
        for name, digest in row['files'].items(): assert sha(name) == digest
    for view in verdict['views']:
        for name in view['inspected']: assert (ROOT/name).is_file()
    # Retain current outputs plus the original failed pre-render attempt.
    for p in ROOT.rglob('*'):
        if p.is_file() and p.name not in ['artifact_manifest.json', 'seal.log']:
            bindings[str(p)] = sha(p)
    for name in ['audit_protected_field_veto.py', 'study_protected_field_veto.py',
                 'review_protected_field_veto.py', 'prune_measured_free_surface.py',
                 'study_confidence_depth_prior.py', 'carve_patchmatch_mesh_free_space.py']:
        p = Path(__file__).with_name(name); bindings[str(p)] = sha(p)
    atomic_json(ROOT/'artifact_manifest.json', dict(status=verdict['status'], hashes=bindings,
        production_promoted=False, complete_render_count=6, visual_fail_views=3,
        raw_volume_serialized=False))
    for name, digest in bindings.items(): assert sha(name) == digest
    print('Sealed and rechecked', len(bindings), 'SHA-256 bindings', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--seal', action='store_true')
    if p.parse_args().seal: seal()
    else: audit()
