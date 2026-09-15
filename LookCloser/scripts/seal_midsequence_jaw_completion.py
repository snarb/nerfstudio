"""Seal the completed local transfer and the independent dark-source diagnosis.

Visual observations are the main agent's completed native-jaw-panel review,
not an automatically inferred image-quality pass.
"""
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json
from render_midsequence_jaw_completion import ROOT, FRAMES, VIEWS, geometry, baseline
from review_jaw_repair_transfer import verified_image
from render_smooth_temporal_mesh_video import verify_request


def seal():
    bindings = {}; inspected = []; results = []; new_renders = 0
    for frame in FRAMES:
        geo = geometry(frame); audit = read(geo/'audit.json')
        assert audit['original_prefix_exact'] and audit['reconstruction_is_an_inferred_prior']
        assert audit['mesh_sha256'] == sha(geo/'mesh.ply')
        summary = read(ROOT/frame/'review_result.json')
        assert [r['view'] for r in summary['records']] == VIEWS
        for row in summary['records']:
            view = row['view']; candidate = ROOT/frame/'rgb'/view/'completed'
            images = []
            for folder in [baseline(frame, view), candidate]:
                request = verify_request(folder); image, receipt = verified_image(folder, frame)
                assert image.shape == (1920, 1080, 3) and len(receipt['source_cameras']) == 62
                images.append(image)
                for name, digest in request['script_hashes'].items():
                    path = Path(__file__).with_name(name); assert sha(path) == digest
                    bindings[str(path)] = digest
                for name in ['request.json', f'frames/{frame}/complete.json']:
                    bindings[str(folder/name)] = sha(folder/name)
                for name, digest in read(folder/'frames'/frame/'complete.json')['hashes'].items():
                    path = folder/'frames'/frame/name; assert sha(path) == digest
                    bindings[str(path)] = digest
            assert np.any(images[0] != images[1], axis=2).sum() == row['changed_rgb']
            assert row['lost_depth'] == 0 and row['new_black'] == 0
            request = read(candidate/'request.json'); source = request['inventory'][0]
            old = o3d.io.read_triangle_mesh(read(geo/'request.json')['source_mesh'])
            new = o3d.io.read_triangle_mesh(source['mesh'])
            np.testing.assert_array_equal(np.asarray(old.vertices), np.asarray(new.vertices)[:len(old.vertices)])
            np.testing.assert_array_equal(np.asarray(old.triangles), np.asarray(new.triangles)[:len(old.triangles)])
            for name, digest in row['images'].items(): assert sha(name) == digest
            panel = str(ROOT/frame/'review'/view/'jaw_native.png'); inspected.append(panel)
            note = 'No material jaw improvement in this view; only tiny or zero pixel changes.'
            if view == VIEWS[-1] and frame != '000995':
                note = 'Some isolated holes filled, but the visible thin dark under-jaw seam remains.'
            row['visual_status'] = 'fail_material_repair_goal'
            results.append(dict(frame=frame, view=view, status=row['visual_status'], notes=note, inspected=[panel]))
            new_renders += 1 + int(frame != '000995' and view != 'moving')
        atomic_json(ROOT/frame/'review_result.json', summary)
        for name in ['request.json', 'result.json', 'audit.json', 'mesh.ply']:
            bindings[str(geo/name)] = sha(geo/name)
    for frame, count in [('001193', 110), ('001195', 67)]:
        folder = ROOT/frame/'seam_diagnosis'; classification = read(folder/'classification.json')
        for record in classification['records']:
            assert record['roi_pixels'] == 8611
            assert all(record[key] == 0 for key in ['black_rgb', 'geometry_misses', 'valid_geometry_no_source', 'valid_source_black_rgb'])
        trace = read(folder/'source_trace/result.json')
        assert trace['selected_dark_ridge_pixels'] == count and trace['max_uint8_rgb_replay_error'] <= 1
        assert sha(folder/'source_trace/evidence.npz') == trace['evidence_sha256']
        for row in trace['examples']:
            assert sha(row['path']) == row['sha256']; inspected.append(row['path'])
    assert new_renders == 13
    atomic_json(ROOT/'visual_review.json', dict(status='fail_material_repair_goal', records=results,
        inspected_native_jaw_panels=9, inspected_source_witnesses=4, inspected=inspected,
        main_agent_visual_review=True, full_head_inventory_approved=False,
        production_promoted=False, no_full_video_or_heldout_metric_claim=True,
        conclusion='Tiny observed-neighborhood gap filling transfers, but not material broad repair. Late K/B seam has valid geometry and actual dark J/C RGB; source trace replays within one uint8 level.'))
    for name in ['transfer_close_boundary_midsequence.py', 'render_midsequence_jaw_completion.py',
                 'supervise_midsequence_jaw_completion.py', 'diagnose_midsequence_jaw_seam.py',
                 'classify_midsequence_jaw_seam.py', 'trace_midsequence_jaw_rgb.py', Path(__file__).name]:
        path = Path(__file__).with_name(name); bindings[str(path)] = sha(path)
    for path in ROOT.rglob('*'):
        if path.is_file() and path.name not in ['artifact_manifest.json', 'seal.log']:
            bindings[str(path)] = sha(path)
    atomic_json(ROOT/'artifact_manifest.json', dict(status='fail_material_repair_goal', hashes=bindings,
        new_completed_renders=13, comparison_pairs=9, production_promoted=False,
        raw_volume_serialized=False, quality_metrics=False))
    for name, digest in bindings.items(): assert sha(name) == digest
    print('Sealed', len(bindings), 'bindings; 13 completed renders, nine comparisons; not promoted', flush=True)


if __name__ == '__main__': seal()
