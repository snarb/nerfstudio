"""Seal a completed production-base diagnostic, never promote a video implicitly."""
from pathlib import Path
import numpy as np
from PIL import Image
from run_mhr_production_patch_control import ROOT, OUT, CANDIDATES, ARM, PARENT, FRAME, binding, configure, builder
from review_mhr_production_patch_control import check_recipe
from study_multiview_face_prior import read, save, sha


def main():
    import open3d as o3d
    proof = binding()
    checked = {}
    def check(path, digest):
        path = Path(path)
        assert sha(path) == digest, str(path)
        checked[str(path)] = digest
    assert read(ROOT/'request.json') == proof
    replay = read(ROOT/'audit.json')
    assert replay['candidate_arrays_replayed'] and replay['candidate_ply_byte_exact']
    assert replay['production_base_binding'] == proof
    check(OUT/'audit.json', replay['admission_audit_sha256'])
    depth_audit = read(OUT/'audit.json')
    assert depth_audit['status'] == 'passed' and depth_audit['native_ray_checks_replayed'] == 248
    assert depth_audit['original_prefix_exact'] and depth_audit['source_geometric_depth_hashes'] == 62
    # That audit ran concurrently with RGB. Its geometry inventory is immutable;
    # renderer progress/results were a point-in-time snapshot, not a seal.
    for name, digest in depth_audit['inventory'].items():
        if not name.startswith('rgb/'):
            check(OUT/name, digest)
    request = read(OUT/'request.json')
    for name, digest in request['helpers'].items():
        check(Path(__file__).with_name(name), digest)
    check(Path(__file__).with_name('admit_mhr_local_patch_depth.py'), request['script_sha256'])
    parent = read(PARENT/'request.json')
    check_recipe(parent)
    production = next(r for r in parent['inventory'] if r['frame_id'] == FRAME)
    old = o3d.io.read_triangle_mesh(proof['production_mesh'])
    ov, ot = np.asarray(old.vertices), np.asarray(old.triangles)
    geometry = {}
    for branch in ['strict', 'interpolated']:
        folder = OUT/ARM/branch
        result = read(folder/'result.json')
        for name, digest in result['hashes'].items():
            check(folder/name, digest)
        mesh = o3d.io.read_triangle_mesh(str(folder/'mesh.ply'))
        v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        assert np.isfinite(v).all() and t.min() >= 0 and t.max() < len(v)
        np.testing.assert_array_equal(v[:len(ov)], ov)
        np.testing.assert_array_equal(t[:len(ot)], ot)
        assert len(t) - len(ot) == result['added']
        checks = depth_audit['details'][0]['branches'][branch]['native_checks']
        assert len(checks) == len(set((r['camera'], r['offset']) for r in checks)) == 124
        assert all(r['trusted_free_pixels'] == 0 for r in checks)
        geometry[branch] = dict(vertices=len(v), triangles=len(t), added=result['added'])
    views = ['old_moving', 'F004_E', 'M004_B', 'C004_E']
    from joint_temporal_texture import HELD_CAMERAS
    import render_smooth_temporal_mesh_video as renderer
    rows, _, _ = renderer.cameras(FRAME)
    check(renderer.ROOT/'parameters.npz', parent['profiles_sha256'])
    check(renderer.ROOT/'exposure.json', parent['exposure_sha256'])
    moving_path = Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')
    moving = next(r['camera'] for r in read(moving_path)['inventory'] if r['frame_id'] == FRAME)
    rgb_records = []
    for view in views:
        camera = moving if view == 'old_moving' else next(r for r in rows if r['physical_camera'].startswith(view))
        for variant in ['baseline', 'strict', 'interpolated']:
            folder = OUT/'rgb'/view/variant
            q = read(folder/'request.json')
            check_recipe(q)
            assert q['recipe'] == parent['recipe'] and q['production_base_binding'] == proof
            assert q['inventory'][0]['camera'] == camera
            assert q['inventory'][0]['source_masks'] == production['source_masks']
            assert q['source_quality_implementation_sha256'] == parent['source_quality_implementation_sha256']
            for key in ['profiles_sha256', 'exposure_sha256']:
                assert q[key] == parent[key]
            check(Path(__file__).with_name('review_mhr_production_patch_control.py'), q['script_sha256'])
            for name, digest in q['helpers'].items():
                check(Path(__file__).with_name(name), digest)
            retained = folder/'frames'/FRAME
            receipt = read(retained/'complete.json')
            check(folder/'request.json', receipt['request_sha256'])
            for name, digest in receipt['hashes'].items():
                check(retained/name, digest)
            result = read(retained/'result.json')
            assert len(result['source_cameras']) == len(set(result['source_cameras'])) == 62
            assert not set(result['source_cameras']) & HELD_CAMERAS
            assert result['source_cameras'] == [r['physical_camera'] for r in rows]
            assert result['camera'] == camera and result['mesh_sha256'] == q['inventory'][0]['mesh_sha256']
            assert Image.open(retained/'frame.png').size == (1080, 1920)
            depth = np.load(retained/'target_depth.npz')['depth']
            assert depth.shape == (1080, 1920) and np.isfinite(depth).all() and (depth >= 0).all()
            rgb_records.append(dict(view=view, variant=variant, complete_sha256=sha(retained/'complete.json')))
        review = read(OUT/'rgb_review'/(view+'.json'))
        check(review['panel_path'], review['panel_sha256'])
        for path, digest in review['input_hashes'].items():
            check(path, digest)
    verdict = read(ROOT/'visual_review.json')
    assert verdict['reviewer'] == 'LLM' and verdict['status'] == 'local_improvement_with_remaining_global_defects'
    assert verdict['production_promoted'] is False
    for path, digest in verdict['viewed_images'].items():
        check(path, digest)
    occlusion = read(OUT/'occlusion_review/result.json')
    for item in occlusion['files']:
        check(item['path'], item['sha256'])
    for path, digest in occlusion['input_hashes'].items():
        check(path, digest)
    black = read(OUT/'black_pixel_review/result.json')
    for item in black['files']:
        check(item['path'], item['sha256'])
    for path, digest in black['input_hashes'].items():
        check(path, digest)
    check(Path(__file__).with_name('localize_mhr_production_patch_side_effects.py'), black['script_sha256'])
    residual = read(ROOT/'residual_hole/result.json')
    check(Path(__file__).with_name('diagnose_mhr_production_residual_hole.py'), residual['script_sha256'])
    check(ROOT/'residual_hole/evidence.npz', residual['evidence_sha256'])
    for path, digest in residual['input_hashes'].items():
        check(path, digest)
    attribution_root = ROOT/'residual_hole/mask_attribution'
    attribution = read(attribution_root/'result.json')
    for item in attribution['files']:
        check(item['path'], item['sha256'])
    check(attribution_root/'projections.npz', attribution['projections_sha256'])
    for item in attribution['records']:
        if item['outside_samples']:
            check(item['source_path'], item['source_sha256'])
    from scipy.ndimage import distance_transform_edt, map_coordinates
    import admit_mhr_local_patch_depth as admission
    configure()
    _, _, _, masks, names, inputs = admission.inputs()
    assert inputs == attribution['input_binding'] == request['inputs']
    projections = np.load(attribution_root/'projections.npz')
    prior = read(builder.PRIOR/'protocol.json')
    margin = []
    for item in attribution['records']:
        if not item['outside_samples']:
            continue
        name = item['camera']; mask = masks[names.index(name)] > 0
        signed = distance_transform_edt(~mask) - distance_transform_edt(mask)
        uv = projections[name]
        values = map_coordinates(signed, [uv[:, 1], uv[:, 0]], order=1)
        margin.append(dict(camera=name, minimum=float(values.min()), maximum=float(values.max()),
            outside_zero=int((values > 0).sum()), outside_two=int((values > 2).sum()),
            camera_in_fit=name in prior['fit_cameras']))
    save(ROOT/'residual_hole/margin_audit.json', dict(records=margin,
        prior_protocol_sha256=sha(builder.PRIOR/'protocol.json'),
        prior_boundary_tolerance_pixels=prior['recipe']['boundary_tolerance_pixels'],
        admission_outside_mask_tolerance_pixels=0, distance_method='bilinear signed pixel EDT',
        note='Bilinear SDF and nearest-pixel mask votes differ at subpixel boundaries.',
        projections_sha256=attribution['projections_sha256'], input_binding=inputs,
        script_sha256=sha(__file__), posthoc_only=True))
    inventory = {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*'))
                 if p.is_file() and p.name != 'final_seal.json'}
    save(ROOT/'final_seal.json', dict(status='passed', script_sha256=sha(__file__),
        checked_bindings=checked, inventory=inventory, geometry=geometry, rgb_records=rgb_records,
        production_modified=False, full_sequence_accepted=False, artifact_free_approval=False,
        matched_cpu_not_cuda_equivalence=True, diagnostic_rgb_dimensions=[1080, 1920],
        diagnostic_not_6k_delivery=True, historical_admission_audit_rgb_snapshot_not_a_seal=True))
    print('production-base diagnostic sealed', len(checked), 'bindings', len(inventory), 'files', flush=True)


if __name__ == '__main__':
    main()
