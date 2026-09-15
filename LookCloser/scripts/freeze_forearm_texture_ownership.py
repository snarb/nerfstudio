"""Verify the nine fixed-geometry controls and preserve their negative gate."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json
from build_multiview_forearm_admission import ROOT as BASE, FRAME
from render_visible_patch_owner import ROOT
from render_component_coherent_forearm import ROOT as AREA
from diagnose_forearm_texture_seam import ROOT as DIAG
from render_forearm_layer_qualified_guard import VIEWS
from review_jaw_repair_transfer import verified_image
from nearby_texture_regions import texture_regions


def freeze():
    meshpath = BASE / 'mesh.ply';q = read(BASE / 'request.json')
    old = o3d.io.read_triangle_mesh(q['source_mesh']);mesh = o3d.io.read_triangle_mesh(str(meshpath))
    old_count = len(old.triangles);records = [];inputs = {str(meshpath): sha(meshpath)}
    for mode in ['area', 'added', 'joined']:
        for view in VIEWS:
            dest = AREA / view if mode == 'area' else ROOT / mode / view
            before, ar = verified_image(BASE / 'rgb' / view, FRAME);after, br = verified_image(dest, FRAME)
            request = read(dest / 'request.json');selection = read(dest / 'owner_selection.json')
            assert selection['request_sha256'] == sha(dest / 'request.json')
            assert not request['artifact_free_approval'] and not br['rgb_averaging'] and not br['target_rgb_read']
            for key in ['camera', 'source_cameras', 'fixed_exposure', 'mesh_sha256']:
                assert ar[key] == br[key]
            da = np.load(BASE / 'rgb' / view / 'frames' / FRAME / 'target_depth.npz')['depth']
            db = np.load(dest / 'frames' / FRAME / 'target_depth.npz')['depth']
            np.testing.assert_array_equal(da, db)
            assert after.shape == (1920, 1080, 3) and np.isfinite(after).all()
            assert np.array_equal(before[:1200], after[:1200])
            labels_a = np.load(BASE / 'rgb' / view / 'frames' / FRAME / 'face_source_labels.npy')
            labels_b = np.load(dest / 'frames' / FRAME / 'face_source_labels.npy')
            if mode != 'joined':
                np.testing.assert_array_equal(labels_a[:old_count], labels_b[:old_count])
            else:
                weights = np.load(dest / 'region_weights.npz')
                regions = texture_regions(np.asarray(mesh.vertices), np.asarray(mesh.triangles), old_count, join_original=True)
                np.testing.assert_array_equal(regions, weights['region'])
                np.testing.assert_array_equal(labels_a[regions < 0], labels_b[regions < 0])
            largest = max(selection['regions'], key=lambda r:r['faces'])
            records.append(dict(mode=mode, view=view, depth_identical=True,
                old_face_labels_changed=int((labels_a[:old_count] != labels_b[:old_count]).sum()),
                largest_owner=ar['source_cameras'][largest['owner']], largest_coverage=largest['owner_coverage'],
                changed_rgb_pixels=int(np.any(before != after, axis=2).sum()),
                new_black_pixels=int(((before.max(2) > 0) & (after.max(2) == 0)).sum()),
                upper_1200_rows_unchanged=True, status='fail'))
            for name, digest in request['script_hashes'].items():
                path = Path(__file__).resolve().with_name(name);assert sha(path) == digest;inputs[str(path)] = digest
    attribution = read(DIAG / 'result.json')
    assert attribution['script_sha256'] == sha(Path(__file__).with_name('diagnose_forearm_texture_seam.py'))
    assert attribution['mesh_sha256'] == sha(meshpath)
    inspected = [DIAG / (v+'_'+suffix+'.png') for v in ['H004_A005_1210M6', 'moving'] for suffix in ['maps', 'legend']]
    inspected += [AREA / 'review' / (v+'_detail.png') for v in VIEWS]
    inspected += [ROOT / 'review' / (v+'_detail.png') for v in VIEWS] + [ROOT / 'review/moving_overview.png']
    atomic_json(ROOT / 'audit.json', dict(records=records, geometry_fixed=True, frame_count=1, new_rgb_renders=9,
        full_frame_quality_metrics=False, production_updated=False))
    atomic_json(ROOT / 'visual_review.json', dict(status='reject_all_three_ownership_variants',
        inspected_images={str(p): sha(p) for p in inspected},
        notes='Lower forearm source band is reduced, but moving view gains a conspicuous light wrist patch. Joining nearby old faces moves the seam and adds mottling; native controls do not establish a general benefit.',
        geometry_unchanged=True, no_new_exposure_or_color_fit=True, rgb_averaging=False,
        whole_frame_status='fail', full_video_approved=False, production_updated=False,
        diagnostic_erratum='First fresh train-depth check omitted the existing renderer target mask and failed before writing images. Corrected replay masks only native train targets and matches saved depth. Failed log retained.'))
    names = ['diagnose_forearm_texture_seam.py', 'render_component_coherent_forearm.py', 'render_visible_patch_owner.py']
    ps = subprocess.check_output(['ps', '-eo', 'pid,etime,args'], text=True)
    live = [s for s in ps.splitlines() if any('python scripts/'+n in s for n in names) and '/bin/bash' not in s]
    assert not live
    atomic_json(ROOT / 'terminal_check.json', dict(utc=datetime.now(timezone.utc).isoformat(), live_workers=live,
        free_bytes=shutil.disk_usage(ROOT).free,
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()))
    hashes = dict(inputs)
    for folder in [ROOT, AREA, DIAG]:
        hashes.update({str(p): sha(p) for p in folder.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})
    for name in names + ['component_texture_owner.py', 'nearby_texture_regions.py', Path(__file__).name]:
        path = Path(__file__).resolve().with_name(name);hashes[str(path)] = sha(path)
    for pattern in ['dec5_component_coherent_forearm_*.log', 'dec5_visible_patch_*.log', 'dec5_forearm_seam_attribution*.log']:
        for path in Path('/mnt/data').glob(pattern):
            hashes[str(path)] = sha(path)
    atomic_json(ROOT / 'artifact_manifest.json', dict(hashes=hashes, full_video_approved=False, production_updated=False))
    print('Frozen', len(hashes), 'hashes; nine RGB controls rejected', flush=True)


def verify():
    hashes = read(ROOT / 'artifact_manifest.json')['hashes']
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Changed artifact: '+path)
    print('Rechecked', len(hashes), 'hashes', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--check', action='store_true');a = p.parse_args()
    if a.check:
        verify()
    elif (ROOT / 'artifact_manifest.json').exists():
        raise ValueError('Already frozen; use --check')
    else:
        freeze()
