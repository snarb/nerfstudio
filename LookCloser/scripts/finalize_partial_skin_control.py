"""Freeze two negative controls, including direct native visual verdicts."""
from pathlib import Path
import argparse
import shutil
import subprocess
import time
import numpy as np
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json
from freeze_forearm_secondary_reference import verify

DATA = Path('/mnt/data')
HERE = Path(__file__).resolve().parent
ROOT = DATA / 'dec5_partial_skin_annotation_roundoff_control'
FAILED = DATA / 'dec5_partial_skin_annotation_control'
REVIEW = DATA / 'dec5_partial_skin_annotation_review'
PROBE = DATA / 'dec5_forearm_camera_phase_probe'
MANIFEST = ROOT / 'artifact_manifest.json'


def run(check=False):
    if check:
        m = read(MANIFEST)
        verify(m['retained_hashes']); verify(m['external_hashes'])
        print('Verified', len(m['retained_hashes']), 'retained and', len(m['external_hashes']), 'external files', flush=True)
        return
    if MANIFEST.exists():
        raise ValueError('Already frozen; use --check')
    processes = subprocess.check_output(['ps', '-eo', 'pid,etime,args'], text=True)
    patterns = ['scripts/study_coherent_forearm_replacement.py --', 'scripts/study_confidence_boundary_completion.py render',
                'scripts/audit_coherent_forearm_replacement.py --', 'scripts/diagnose_constrained_surface_regression.py --']
    if any(p in line for line in processes.splitlines() for p in patterns):
        raise ValueError('Relevant worker still active')
    ancestor = DATA / 'dec5_protected_constrained_forearm/artifact_manifest.json'
    parent = read(ancestor)
    verify(parent['retained_hashes']); verify(parent['external_hashes'])
    external = {**parent['external_hashes'], str(ancestor): sha(ancestor)}
    snapshots = ROOT / 'config'; snapshots.mkdir(exist_ok=True)
    for folder in [FAILED / '001037', ROOT / '001037']:
        request = read(folder / 'request.json')
        for name, digest in request['scripts'].items():
            source = next((p for p in [HERE / name, FAILED / 'config' / name] if p.exists() and sha(p) == digest), None)
            if source is None:
                raise ValueError('Missing exact producer ' + name)
            target = snapshots / (digest + '_' + name)
            if target.exists() and sha(target) != digest:
                raise ValueError('Changed snapshot')
            if not target.exists(): shutil.copyfile(source, target)
        external.update(request['source_depth_sha256'])
    folder = ROOT / '001037'
    result = read(folder / 'geometry_result.json'); audit = read(folder / 'independent_audit.json')
    if result['request_sha256'] != sha(folder / 'request.json') or not result['observed_guard_passed']:
        raise ValueError('Unverified final geometry')
    verify({str(folder / n): h for n, h in result['hashes'].items()})
    if audit['geometry_result_sha256'] != sha(folder / 'geometry_result.json') or not audit['protected_production_prefix_exact'] or audit['checks'] != 124 or audit['qualified_veto_pixels']:
        raise ValueError('Incomplete independent check')
    external.update(result['color_guard_provenance']['source_rgb_hashes'])
    metrics = read(REVIEW / 'metrics.json')
    if len(metrics['rows']) != 2 or metrics['scope'] != 'fixed manual train forearm skin; not heldout face or full-frame':
        raise ValueError('Unexpected metric scope')
    verify(metrics['prediction_hashes']); external.update(metrics['prediction_hashes'])
    extent = []
    for view in ['moving', 'H004_A005_1210M6']:
        paths = [DATA / 'dec5_protected_constrained_forearm' / 'rgb' / '001037' / view / 'guarded/frames/001037/frame.png',
                 ROOT / 'rgb' / '001037' / view / 'guarded/frames/001037/frame.png']
        a, b = [np.array(Image.open(p)) for p in paths]
        change = np.any(a != b, axis=2)
        extent.append(dict(view=view, changed_pixels=int(change.sum()), changed_top_1400=int(change[:1400].sum())))
        receipt = read(paths[1].parent / 'complete.json')
        verify({str(paths[1].parent / n): h for n, h in receipt['hashes'].items()})
    atomic_json(REVIEW / 'visual_review.json', dict(
        frame='001037', status='fail', production_accepted=False,
        notes='Lateral gap larger; wrist hole and broken hand remain. Rejected; no transfer.',
        actually_inspected_panels={str(REVIEW / '001037' / (v + '_comparison.png')):
                                  sha(REVIEW / '001037' / (v + '_comparison.png')) for v in ['moving', 'H004_A005_1210M6']},
        rgb_identity_diagnostic_not_quality_metric=extent))
    probe = read(PROBE / 'request.json')
    if sha(Path(probe['parent']) / 'request.json') != probe['parent_request_sha256']:
        raise ValueError('Changed camera parent')
    for record in read(PROBE / 'results.json')['records']:
        verify({record['path']: record['sha256']})
    seen = [PROBE / f / 'overview.png' for f in probe['frames']]
    seen += [PROBE / '001037' / ('offset_' + p + '_arm.png') for p in ['000', '120']]
    atomic_json(PROBE / 'visual_review.json', dict(
        status='rejected_as_complete_hand_workaround', full_rgb_video_started=False,
        notes='Damage persists in usable views; some phases clip the hand at the image border. Offscreen clipping is not repair.',
        actually_inspected_images={str(p): sha(p) for p in seen},
        no_new_quality_metrics=True, mesh_improvement_claimed=False))
    tests = DATA / 'dec5_partial_skin_roundoff_tests.log'
    if '24 passed' not in tests.read_text(): raise ValueError('Tests incomplete')
    for name in ['audit_coherent_forearm_replacement.py', 'diagnose_constrained_surface_regression.py',
                 'probe_forearm_camera_phase.py', 'study_confidence_boundary_completion.py',
                 'review_confidence_boundary_completion.py', Path(__file__).name]:
        target = snapshots / name
        if target.exists() and sha(target) != sha(HERE / name): raise ValueError('Changed snapshot')
        if not target.exists(): shutil.copyfile(HERE / name, target)
    external.update({str(p): sha(p) for p in DATA.glob('dec5_partial_skin*.log') if 'freeze' not in p.name})
    report = HERE.parent / 'experiments/dec5_forearm_workaround_limits.md'
    external[str(report)] = sha(report)
    external[str(DATA / 'dec5_forearm_camera_phase_probe.log')] = sha(DATA / 'dec5_forearm_camera_phase_probe.log')
    verify(external)
    atomic_json(ROOT / 'final_process_check.json', dict(
        unix_time=time.time(), workers=[], free_bytes=shutil.disk_usage(DATA).free,
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], text=True).strip()))
    retained = {str(p): sha(p) for root in [ROOT, FAILED, REVIEW, PROBE] for p in sorted(root.rglob('*')) if p.is_file()}
    atomic_json(MANIFEST, dict(status='completed_negative_controls', tests_passed=24,
        new_rgb_renders=2, clay_renders=20, independent_native_ray_checks=124,
        production_changed=False, full_video_changed=False, retained_hashes=retained, external_hashes=external))
    print('Frozen', len(retained), 'files; negative controls, movie unchanged', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--check', action='store_true')
    run(p.parse_args().check)
