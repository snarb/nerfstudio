"""Matched three-view hard-source RGB evaluation of a protected TSDF field veto."""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json
from study_protected_field_veto import ROOT
from review_measured_free_surface import VIEWS, baseline
from review_full_block_transfer import ROOT as DEPTH_ROOT
from review_temporal_full_block_control import transfer_mesh_gauge
from review_jaw_repair_transfer import verified_image, panel

FRAME = '000995'
ARMS = ['control', 'protected']
BOXES = {'moving': (380,1400,680,1870), 'H004_C005_1210SZ': (230,1080,530,1550),
         'K004_B005_1210DS': (40,1120,340,1590)}
HEADS = {'moving': (150,900,850,1490), 'H004_C005_1210SZ': (100,500,1000,1200),
         'K004_B005_1210DS': (100,450,950,1210)}


def prepare():
    for arm in ARMS:
        root = ROOT/arm; receipt = read(root/'complete.json')
        assert receipt['request_sha256'] == sha(root/'request.json')
        for name, digest in receipt['hashes'].items(): assert sha(root/name) == digest
        for view in VIEWS:
            q = deepcopy(read(baseline(FRAME, view)/'request.json'))
            q['inventory'] = [r for r in q['inventory'] if r['frame_id'] == FRAME]
            entry = q['inventory'][0]; meta = read(entry['metadata'])
            mesh = o3d.io.read_triangle_mesh(str(root/'mesh.ply'))
            before = np.asarray(mesh.vertices).copy(); source_meta = read(root/'mesh.json')
            points = transfer_mesh_gauge(before, source_meta, meta)
            np.testing.assert_allclose(transfer_mesh_gauge(points, meta, source_meta), before, rtol=0, atol=1e-12)
            out = root/'rgb'/view; out.mkdir(parents=True, exist_ok=False)
            (out/'frames').mkdir()
            mesh.vertices = o3d.utility.Vector3dVector(points)
            assert o3d.io.write_triangle_mesh(str(out/'mesh.ply'), mesh)
            entry.update(mesh=str(out/'mesh.ply'), mesh_sha256=sha(out/'mesh.ply'))
            q['source_rows'] = [r for r in q['source_rows'] if Path(r['source_dataset']).name == FRAME]
            q['ordered_frame_ids'] = [FRAME]
            q.update(partial_diagnostic_only=True, full_video_candidate=False, geometry_changed=True,
                artifact_free_approval=False, protected_field_arm=arm, raw_fusion_pair=True,
                production_head_repairs_reapplied=False, texture_source_masks_unchanged=True,
                fusion_completion_sha256=sha(root/'complete.json'),
                gauge_roundtrip_passed=True, max_gauge_displacement=float(np.abs(points-before).max()))
            q['script_hashes'][Path(__file__).name] = sha(__file__)
            atomic_json(out/'request.json', q)


def render(view):
    from run_view_consistent_dynamic_video import install
    import render_smooth_temporal_mesh_video as engine
    implementation = install(); engine.torch.set_num_threads(2)
    for arm in ARMS:
        out = ROOT/arm/'rgb'/view
        assert implementation == read(out/'request.json')['source_quality_implementation_sha256']
        engine.render(out, [FRAME])


def review():
    records = []
    for view in VIEWS:
        folders = {'production': baseline(FRAME, view), **{a: ROOT/a/'rgb'/view for a in ARMS}}
        images = {}; receipts = {}; depths = {}
        for name, folder in folders.items():
            images[name], receipts[name] = verified_image(folder, FRAME)
            depths[name] = np.load(folder/'frames'/FRAME/'target_depth.npz')['depth']
            assert np.isfinite(depths[name]).all()
            for key in ['camera', 'source_cameras', 'fixed_exposure']:
                assert receipts[name][key] == receipts['production'][key]
            for key in ['profiles_sha256', 'exposure_sha256', 'calibration_sha256']:
                assert read(folder/'request.json')[key] == read(folders['production']/'request.json')[key]
        ims = list(images.values()); labels = ['production (later repairs)', 'raw full-block control', 'protected field veto']
        if view != 'moving':
            ims.insert(0, np.asarray(Image.open(DEPTH_ROOT/FRAME/'review'/view/'train_gt.png')))
            labels.insert(0, 'real train GT')
        out = ROOT/'review'/view
        panel(out/'lipstick_native.png', ims, labels, BOXES[view])
        panel(out/'head_native.png', ims, labels, HEADS[view])
        small = [np.asarray(Image.fromarray(im).resize((324,576))) for im in ims]
        panel(out/'actor.png', small, labels, (0,0,324,576))
        a, b = depths['control'], depths['protected']
        records.append(dict(view=view, changed_rgb=int(np.any(images['control'] != images['protected'], axis=2).sum()),
            lost_depth=int(((a > 0)&(b == 0)).sum()), added_depth=int(((a == 0)&(b > 0)).sum()),
            farther_depth=int(((a > 0)&(b > a+1e-6)).sum()),
            nearer_depth=int(((a > 0)&(b > 0)&(b < a-1e-6)).sum()),
            files={str(p): sha(p) for p in out.iterdir() if p.is_file()}, visual_status='pending'))
    atomic_json(ROOT/'review_result.json', dict(records=records, counts_not_quality_metrics=True,
        production_promoted=False, raw_fusion_pair_does_not_reapply_production_repairs=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['prepare', 'render', 'review'])
    p.add_argument('--view', choices=VIEWS); a = p.parse_args()
    if a.action == 'render':
        if not a.view: p.error('render requires --view')
        render(a.view)
    else: globals()[a.action]()
