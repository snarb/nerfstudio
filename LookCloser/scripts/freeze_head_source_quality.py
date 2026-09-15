"""Freeze matched source-quality evidence; no mesh or artifact-free approval."""
import argparse
from pathlib import Path
import numpy as np
from joint_temporal_texture import read, sha, atomic_json
from review_jaw_repair_transfer import verified_image
from render_smooth_temporal_mesh_video import verify_request
from study_view_consistent_head_texture import CASES, BASE

ROOT = Path('/mnt/data/dec5_head_source_quality_heldout')
MANIFEST = ROOT/'artifact_manifest.json'
STUDY = Path('/mnt/data/dec5_view_consistent_head_texture')
PIXEL = Path('/mnt/data/dec5_pixel_angular_head_texture')
UNWARP = Path('/mnt/data/dec5_unwarped_head_texture')
LAYERS = Path('/mnt/data/dec5_head_seam_layers')
VISIBILITY = Path('/mnt/data/dec5_native_self_visibility')


def freeze(check=False):
    if check:
        files = read(MANIFEST)['files']
        for p, digest in files.items():
            if sha(p) != digest:
                raise ValueError('Changed evidence: '+p)
        print('Verified', len(files), 'study hashes'); return
    if MANIFEST.exists():
        raise ValueError('Already frozen; use --check')
    pairs = []
    inspected = []
    for root in [STUDY/'incidence2', STUDY/'angular_only', PIXEL, UNWARP]:
        for frame, view in CASES:
            dest = root/frame/view
            pairs.append((dest, BASE/frame/view, frame))
            inspected.append(dest/'review/head.png')
    for mode in ['baseline', 'incidence2', 'angular_only', 'pixel_angular']:
        pairs.append((ROOT/mode, ROOT/'baseline', '001193'))
        if mode != 'baseline':
            inspected.append(ROOT/(mode+'_heldout.png'))
    pairs.append((UNWARP/'unwarped', ROOT/'incidence2', '001193'))
    inspected.append(UNWARP/'unwarped_heldout.png')
    external = {}
    for dest, baseline, frame in pairs:
        q = verify_request(dest)
        _, result = verified_image(dest, frame)
        _, old = verified_image(baseline, frame)
        for key in ['camera', 'mesh_sha256', 'source_cameras', 'fixed_exposure']:
            if result[key] != old[key]:
                raise ValueError('Unmatched source-quality control: '+key)
        np.testing.assert_array_equal(
            np.load(dest/'frames'/frame/'target_depth.npz')['depth'],
            np.load(baseline/'frames'/frame/'target_depth.npz')['depth'])
        for row in q['inventory']:
            for key in ['mesh', 'metadata']:
                if sha(row[key]) != row[key+'_sha256']:
                    raise ValueError('Changed geometry input')
                external[row[key]] = row[key+'_sha256']
        for name, digest in q['script_hashes'].items():
            external[str(Path(__file__).with_name(name))] = digest
    # Only maps actually viewed by the main agent; do not imply all panels read.
    for folder, names in {
        '001193_native_G004_C005_121037': ['clay', 'source_boundary', 'added_faces', 'geometry_support'],
        '001123_native_K004_C005_1210BC': ['source_id', 'clay', 'source_boundary', 'geometry_support'],
        '001193_moving': ['clay', 'source_boundary', 'added_faces', 'geometry_support'],
    }.items():
        inspected.extend(LAYERS/folder/(name+'.png') for name in names)
    # Separate sidecar preserves immutable metric/request hashes and explicitly
    # supersedes the provisional pending labels inside original diagnostics.
    atomic_json(ROOT/'visual_review.json', dict(
        status='reviewed_texture_gain_geometry_failures_remain',
        reviewer='main_agent_actual_image_inspection',
        inspected={str(p): sha(p) for p in inspected},
        supersedes_provisional_pending_labels=True,
        notes='Incidence2 improves held-out face fidelity; pure angular controls worsen LPIPS. Registration-off adds a small gain. Source seams on continuous old geometry are distinct from real crown/jaw holes. Full dynamic movie retains hand/lipstick/jaw failures.',
        artifact_free=False, geometry_repair_approved=False))
    gt = Path('/mnt/data/dec5_gradient_heldout_fidelity/evaluation/001193')
    for root in [ROOT, UNWARP]:
        metrics = read(root/'metrics.json')
        if metrics['full_frame_metrics'] or metrics['loss_reported']:
            raise ValueError('Wrong evaluation protocol')
        for row in metrics['rows']:
            if not all(np.isfinite(row[k]) for k in ['face_psnr', 'face_ssim', 'face_lpips']):
                raise ValueError('Nonfinite metric')
        for name, key in [('gt.png', 'gt_sha256'), ('roi.json', 'roi_sha256')]:
            if sha(gt/name) != metrics[key]:
                raise ValueError('Changed held-out evidence')
            external[str(gt/name)] = metrics[key]
    repo = Path(__file__).resolve().parents[1]
    tests = Path('/mnt/data/dec5_head_source_quality_tests.log')
    if '12 passed' not in tests.read_text():
        raise ValueError('Focused tests required')
    paths = [Path(__file__), tests, repo/'experiments/dec5_head_source_quality.md',
             repo/'tests/test_view_consistent_source_quality.py',
             repo/'tests/test_finalize_view_consistent_video.py']
    files = {str(p): sha(p) for root in [ROOT, STUDY, PIXEL, UNWARP, LAYERS, VISIBILITY]
             for p in root.rglob('*') if p.is_file() and p != MANIFEST and '__pycache__' not in p.parts}
    for p, digest in external.items():
        if sha(p) != digest:
            raise ValueError('Changed dependency: '+p)
    files.update(external); files.update({str(p): sha(p) for p in paths})
    atomic_json(MANIFEST, dict(files=files, matched_render_controls=len(pairs),
        status='texture_gain_not_mesh_repair', artifact_free=False,
        selected_heldout_lpips=0.0722508504986763,
        validation_scope='One held-out face benchmark used for selection, not 150-frame generalization',
        full_video='/mnt/data/dec5_incidence2_unwarped_dynamic_150'))
    print('Frozen', len(files), 'study hashes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    freeze(parser.parse_args().check)
