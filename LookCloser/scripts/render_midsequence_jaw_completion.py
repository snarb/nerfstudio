"""Validate identical jaw-completion thresholds at three times on wider poses."""
from pathlib import Path
from copy import deepcopy
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json, cameras
from transfer_close_boundary_midsequence import ROOT, VIDEO
from review_full_block_transfer import ROOT as DEPTH_ROOT
from review_jaw_repair_transfer import verified_image, panel

FRAMES = ['000995','001193','001195']
VIEWS = ['moving','H004_C005_1210SZ','K004_B005_1210DS']


def geometry(frame):
    if frame == '000995': return ROOT/frame/'geometry/interpolated'/frame
    return Path('/mnt/data/dec5_close_boundary_completion')/frame/'interpolated'/frame


def baseline(frame, view):
    if view == 'moving': return VIDEO
    if frame == '000995': return DEPTH_ROOT/frame/'rgb'/view/'production'
    return ROOT/frame/'rgb'/view/'production'


def prepare():
    parent = read(VIDEO/'request.json')
    for frame in FRAMES:
        source = next(r for r in parent['inventory'] if r['frame_id'] == frame)
        g = geometry(frame); result = read(g/'result.json'); request = read(g/'request.json')
        assert result['observed_guard_passed'] and result['original_prefix_exact']
        assert request['source_mesh_sha256'] == source['mesh_sha256'] == sha(source['mesh'])
        for name, digest in result['hashes'].items(): assert sha(g/name) == digest
        old = o3d.io.read_triangle_mesh(source['mesh']); new = o3d.io.read_triangle_mesh(str(g/'mesh.ply'))
        np.testing.assert_array_equal(np.asarray(old.vertices), np.asarray(new.vertices)[:len(old.vertices)])
        np.testing.assert_array_equal(np.asarray(old.triangles), np.asarray(new.triangles)[:len(old.triangles)])
        rows, _, _ = cameras(frame)
        for view in VIEWS:
            for arm in ['production','completed']:
                if arm == 'production' and (view == 'moving' or frame == '000995'): continue
                q = deepcopy(parent); entry = deepcopy(source)
                if view != 'moving':
                    entry['camera'] = deepcopy(next(r for r in rows if r['physical_camera'] == view))
                    entry['camera']['physical_camera'] = 'diagnostic_unmasked_target_'+view
                    entry['camera']['reference_physical_camera'] = view
                if arm == 'completed': entry.update(mesh=str(g/'mesh.ply'), mesh_sha256=sha(g/'mesh.ply'))
                q['inventory'] = [entry]; q['ordered_frame_ids'] = [frame]
                q['source_rows'] = [r for r in q['source_rows'] if Path(r['source_dataset']).name == frame]
                q.update(partial_diagnostic_only=True, full_video_candidate=False,
                    artifact_free_approval=False, geometry_changed=arm=='completed',
                    jaw_completion_arm=arm, geometry_result_sha256=sha(g/'result.json'),
                    geometry_audit_sha256=sha(g/'audit.json'), source_masks_unchanged=True,
                    native_target_mask_disabled=view!='moving', original_geometry_prefix_exact=True)
                q['script_hashes'][Path(__file__).name] = sha(__file__)
                out = ROOT/frame/'rgb'/view/arm; out.mkdir(parents=True, exist_ok=False)
                (out/'frames').mkdir(); atomic_json(out/'request.json', q)


def render(frame, view):
    from run_view_consistent_dynamic_video import install
    import render_smooth_temporal_mesh_video as engine
    implementation = install(); engine.torch.set_num_threads(2)
    for arm in ['production','completed']:
        if arm == 'production' and (view == 'moving' or frame == '000995'): continue
        out = ROOT/frame/'rgb'/view/arm
        assert implementation == read(out/'request.json')['source_quality_implementation_sha256']
        engine.render(out,[frame])


def review(frame):
    from calibrated_depth_witness import load_images
    gt, _, gt_receipt = load_images(frame); records = []
    for view in VIEWS:
        old = baseline(frame,view); new = ROOT/frame/'rgb'/view/'completed'
        a, ar = verified_image(old,frame); b, br = verified_image(new,frame)
        for key in ['camera','source_cameras','fixed_exposure']: assert ar[key] == br[key]
        for key in ['profiles_sha256','exposure_sha256','calibration_sha256']:
            assert read(old/'request.json')[key] == read(new/'request.json')[key]
        ad = np.rot90(np.load(old/'frames'/frame/'target_depth.npz')['depth'])
        bd = np.rot90(np.load(new/'frames'/frame/'target_depth.npz')['depth'])
        assert np.isfinite(ad).all() and np.isfinite(bd).all()
        assert not ((ad > 0)&(bd == 0)).any()
        assert not ((ad > 0)&(bd > ad+1e-6)).any()
        ims = [a,b]; labels = ['production','observed-neighborhood completion']
        if view != 'moving': ims.insert(0,np.rot90(gt[view])); labels.insert(0,'real train GT')
        if view == 'moving':
            head = (150,900,850,1490) if frame == '000995' else (180,750,760,1280)
            jaw = (280,1220,720,1530) if frame == '000995' else (330,1040,650,1220)
        else:
            head = (100,450,1000,1250); jaw = (300,960,820,1250)
        out = ROOT/frame/'review'/view
        panel(out/'head.png', ims, labels, head); panel(out/'jaw_native.png', ims, labels, jaw)
        small = [np.asarray(Image.fromarray(im).resize((324,576))) for im in ims]
        panel(out/'actor.png', small, labels, (0,0,324,576))
        records.append(dict(view=view, gained_depth=int(((ad == 0)&(bd > 0)).sum()),
            nearer_depth=int(((ad > 0)&(bd > 0)&(bd < ad-1e-6)).sum()), lost_depth=0,
            changed_rgb=int(np.any(a != b,axis=2).sum()), new_black=int(((a.max(2)>0)&(b.max(2)==0)).sum()),
            removed_black=int(((a.max(2)==0)&(b.max(2)>0)).sum()),
            images={str(p):sha(p) for p in out.iterdir() if p.is_file()}, visual_status='pending'))
    atomic_json(ROOT/frame/'review_result.json',dict(records=records, gt_receipt=gt_receipt,
        counts_not_face_metrics=True, production_promoted=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['prepare','render','review'])
    p.add_argument('--frame',choices=FRAMES); p.add_argument('--view',choices=VIEWS); a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='render':
        if not a.frame or not a.view:p.error('render requires frame and view')
        render(a.frame,a.view)
    else:
        if not a.frame:p.error('review requires frame')
        review(a.frame)
