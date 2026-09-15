"""Freeze the protected-production three-time control, not a movie release."""
from pathlib import Path
import argparse
import math
import shutil
import subprocess
import time
from joint_temporal_texture import read, sha, atomic_json
from freeze_forearm_secondary_reference import verify

DATA = Path('/mnt/data')
HERE = Path(__file__).resolve().parent
ROOT = DATA / 'dec5_protected_constrained_forearm'
REVIEW = DATA / 'dec5_protected_forearm_transfer_review'
PILOT = DATA / 'dec5_protected_forearm_initial_review'
MANIFEST = ROOT / 'artifact_manifest.json'
FRAMES = ['001029', '001033', '001037']
VIEWS = ['moving', 'H004_A005_1210M6']


def run(check=False):
    if check:
        manifest = read(MANIFEST)
        verify(manifest['retained_hashes'])
        verify(manifest['external_hashes'])
        print('Verified', len(manifest['retained_hashes']), 'retained and',
              len(manifest['external_hashes']), 'external files', flush=True)
        return
    if MANIFEST.exists():
        raise ValueError('Already frozen; use --check')
    processes = subprocess.check_output(['ps', '-eo', 'pid,etime,args'], text=True)
    patterns = ['scripts/study_coherent_forearm_replacement.py --',
                'scripts/study_confidence_boundary_completion.py render',
                'scripts/audit_coherent_forearm_replacement.py --',
                'scripts/diagnose_constrained_surface_regression.py --']
    if any(p in line for line in processes.splitlines() for p in patterns):
        raise ValueError('Relevant workers still active')
    ancestor = DATA / 'dec5_constrained_forearm_surface_guard16/artifact_manifest.json'
    old = read(ancestor)
    verify(old['retained_hashes'])
    verify(old['external_hashes'])
    external = {**old['external_hashes'], str(ancestor): sha(ancestor)}
    snapshots = ROOT / 'config'
    snapshots.mkdir(exist_ok=True)
    reviews = {}
    for frame in FRAMES:
        folder = ROOT / frame
        request = read(folder / 'request.json')
        result = read(folder / 'geometry_result.json')
        audit = read(folder / 'independent_audit.json')
        loss = read(folder / 'production_loss_diagnosis.json')
        if request['guard_max_rounds'] != 16 or 'feasible_depth_constraints' not in request:
            raise ValueError('Mixed recipes')
        protection = request['protected_production']
        external[protection['mesh']] = protection['mesh_sha256']
        parent = DATA / 'dec5_phase30_early_texture_dynamic_150/request.json'
        external[str(parent)] = protection['parent_request_sha256']
        if result['request_sha256'] != sha(folder / 'request.json') or not result['observed_guard_passed'] or result['rounds'][-1]['removed_triangles']:
            raise ValueError('Unverified geometry')
        if audit['geometry_result_sha256'] != sha(folder / 'geometry_result.json') or not audit['protected_production_prefix_exact'] or not audit['feasible_depth_constraints_replayed'] or audit['checks'] != 124 or audit['qualified_veto_pixels']:
            raise ValueError('Incomplete independent audit')
        if loss['geometry_result_sha256'] != sha(folder / 'geometry_result.json') or loss['removed_production_triangles'] or loss['new_holes_over_previous_production_faces']:
            raise ValueError('Production regression')
        verify({str(folder / n): h for n, h in result['hashes'].items()})
        external.update(request['source_depth_sha256'])
        external.update(result['color_guard_provenance']['source_rgb_hashes'])
        for name, digest in request['scripts'].items():
            source = HERE / name
            if sha(source) != digest:
                raise ValueError('Producer changed: ' + name)
            target = snapshots / (digest + '_' + name)
            if target.exists() and sha(target) != digest:
                raise ValueError('Snapshot changed')
            if not target.exists():
                shutil.copyfile(source, target)
        for view in VIEWS:
            rendered = ROOT / 'rgb' / frame / view / 'guarded' / 'frames' / frame
            if (rendered / 'frame.png').is_symlink():
                raise ValueError('Candidate is a reused image')
            receipt = read(rendered / 'complete.json')
            verify({str(rendered / n): h for n, h in receipt['hashes'].items()})
        reviews[frame] = dict(
            scope='local forearm; not whole actor or video',
            status='pass' if frame == '001029' else 'fail',
            comparative_regression_gate='pass',
            notes={'001029': 'Good forearm retained; tiny marks reduced, mild color seam remains.',
                   '001033': 'Side hole reduced; wrist/hand breakup and side-skin remnants remain.',
                   '001037': 'Lower forearm fuller; conspicuous wrist hole, broken hand and seams remain.'}[frame],
            actually_inspected_panels={str(REVIEW / frame / (v + '_comparison.png')):
                                      sha(REVIEW / frame / (v + '_comparison.png')) for v in VIEWS})
    metrics = read(REVIEW / 'metrics.json')
    if metrics['frames'] != FRAMES or len(metrics['rows']) != 6 or metrics['scope'] != 'fixed manual train forearm skin; not heldout face or full-frame':
        raise ValueError('Unexpected metrics protocol')
    external.update(metrics['prediction_hashes'])
    for frame in FRAMES:
        rows = {r['variant']: r for r in metrics['rows'] if r['frame'] == frame}
        for name in ['psnr', 'ssim', 'lpips']:
            if not all(math.isfinite(r[name]) for r in rows.values()):
                raise ValueError('Nonfinite metric')
        if not (rows['protected']['psnr'] > rows['previous']['psnr'] and
                rows['protected']['ssim'] > rows['previous']['ssim'] and
                rows['protected']['lpips'] < rows['previous']['lpips'] and
                rows['protected']['skin_rgb_holes'] < rows['previous']['skin_rgb_holes'] and
                rows['protected']['skin_depth_holes'] < rows['previous']['skin_depth_holes']):
            raise ValueError('Claimed uniform improvement not reproduced')
    extent = read(REVIEW / 'rgb_change_extent.json')
    if len(extent['records']) != 6 or any(r['changed_rgb_pixels_top_1400_rows'] for r in extent['records']):
        raise ValueError('Head-region RGB changed')
    atomic_json(REVIEW / 'visual_review.json', dict(
        frames=reviews, status='local_regression_gate_passed_remaining_visual_failures',
        production_accepted=False, full_video_rerendered=False,
        head_region_rgb_unchanged=True, main_face_csv_changed=False))
    atomic_json(PILOT / 'visual_review.json', dict(
        frame='001029', scope='local forearm only', status='pass',
        allows='same-recipe three-time transfer, not production rollout',
        panels={str(PILOT / '001029' / (v + '_comparison.png')):
                sha(PILOT / '001029' / (v + '_comparison.png')) for v in VIEWS}))
    tests = DATA / 'dec5_protected_forearm_tests.log'
    if '24 passed' not in tests.read_text():
        raise ValueError('Focused tests incomplete')
    for name in ['audit_coherent_forearm_replacement.py', 'diagnose_constrained_surface_regression.py',
                 'review_confidence_boundary_completion.py', 'review_constrained_forearm_surface.py',
                 'study_confidence_boundary_completion.py', 'study_early_texture_prior.py',
                 'render_smooth_temporal_mesh_video.py', 'wide_dynamic_camera_flight.py', Path(__file__).name]:
        target = snapshots / name
        if target.exists() and sha(target) != sha(HERE / name):
            raise ValueError('Current snapshot changed')
        if not target.exists():
            shutil.copyfile(HERE / name, target)
    report = HERE.parent / 'experiments/dec5_protected_forearm_surface.md'
    external.update({str(p): sha(p) for p in DATA.glob('dec5_protected_forearm*.log') if 'freeze' not in p.name})
    external[str(report)] = sha(report)
    verify(external)
    atomic_json(ROOT / 'final_process_check.json', dict(
        unix_time=time.time(), workers=[], scope='final observed state',
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                     '--format=csv,noheader'], text=True).strip(),
        free_bytes=shutil.disk_usage(DATA).free))
    retained = {str(p): sha(p) for root in [ROOT, REVIEW, PILOT]
                for p in sorted(root.rglob('*')) if p.is_file()}
    atomic_json(MANIFEST, dict(
        status='completed_local_improvement_with_remaining_visual_failures', frames=FRAMES,
        local_forearm_visual_pass=1, local_forearm_visual_fail=2, tests_passed=24,
        independent_native_ray_checks=372, production_changed=False, full_video_rerendered=False,
        retained_hashes=retained, external_hashes=external))
    print('Frozen', len(retained), 'files; local improvement, movie unchanged', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    run(parser.parse_args().check)
