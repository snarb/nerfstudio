"""Freeze reviewed, unpromoted admission controls; verify without rerendering."""
from pathlib import Path
import argparse
import shutil
import subprocess
import time
from joint_temporal_texture import read, sha, atomic_json

DATA = Path('/mnt/data')
HERE = Path(__file__).resolve().parent
MANIFEST = DATA / 'dec5_forearm_admission_order_manifest.json'
ROOTS = [DATA / ('dec5_forearm_' + suffix) for suffix in [
    'admission_plane_bounded', 'admission_quadric_bounded',
    'plane_observed_ring', 'quadric_observed_ring',
    'admission_order_transfer_review', 'admission_order_review',
    'plane_ring_review', 'quadric_ring_review']]
OLD = [DATA / ('dec5_forearm_' + suffix) for suffix in [
    'annotation_domain_control_manifest.json', 'rgb_guard_artifact_manifest.json',
    'depth_observability_manifest.json']]


def verify_files(hashes):
    for filename, expected in hashes.items():
        if sha(filename) != expected:
            raise ValueError('Changed retained file: ' + filename)


def freeze():
    if MANIFEST.exists():
        raise ValueError('Already frozen; use --check')
    previous = {}
    for path in OLD:
        data = read(path)
        hashes = data.get('retained_hashes', data.get('hashes'))
        if not isinstance(hashes, dict):
            raise ValueError('Unrecognized old manifest: ' + str(path))
        verify_files(hashes)
        previous[str(path)] = dict(sha256=sha(path), verified_files=len(hashes))
    movie = DATA / 'dec5_phase30_early_texture_dynamic_150'
    published = {str(movie/'video.mp4'): 'ce1ff5f7fd612a46e39b6ae89d10aee7534f895ea7e7887d722dc10903bccd40',
                 str(movie/'frames.zip'): '5e2d078a2e249c90a9a4b3d4112a7652bda533d45422c91832c7355480c2f7b6'}
    verify_files(published)
    snapshots = DATA/'dec5_forearm_admission_order_config'
    snapshots.mkdir(exist_ok=True)
    producers = [HERE/'study_forearm_production_delta.py',
        DATA/'dec5_forearm_admission_plane/config/study_forearm_production_delta.py']
    provenance = []
    for root in ROOTS[:4]:
        for request_path in sorted(root.glob('??????/request.json')):
            request = read(request_path)
            producer = next((p for p in producers if p.is_file() and sha(p)==request['script_sha256']), None)
            if producer is None:
                raise ValueError('Missing exact producer source: '+str(request_path))
            sources = [producer, HERE/'ordered_forearm_admission.py', HERE/'confidence_boundary_completion.py',
                       HERE/'curve_forearm_delta.py']
            if sha(sources[1]) != request['admission_order']['helper_sha256']:
                raise ValueError('Changed admission source')
            if sha(sources[2]) != request['admission_order']['grid_helper_sha256']:
                raise ValueError('Changed grid source')
            if sha(sources[3]) != request['curvature_helper_sha256']:
                raise ValueError('Changed curvature source')
            if 'observed_ring_feather' in request:
                sources.append(HERE/'observed_ring_curvature.py')
                if sha(sources[-1]) != request['observed_ring_feather']['helper_sha256']:
                    raise ValueError('Changed ring source')
            for source in sources:
                target = snapshots/(sha(source)+'_'+source.name)
                if target.exists() and sha(target) != sha(source):
                    raise ValueError('Changed existing source snapshot')
                if not target.exists():
                    shutil.copyfile(source, target)
            provenance.append(dict(request=str(request_path), sha256=sha(request_path),
                producer_snapshot=str(snapshots/(sha(producer)+'_'+producer.name))))
    for root in [ROOTS[4], ROOTS[6], ROOTS[7]]:
        review_path = root/'visual_review.json'
        review = read(review_path)
        panels = {}
        for frame, verdict in review['frames'].items():
            if verdict['status'] in ['pending', 'uncertain']:
                raise ValueError('Incomplete actual review')
            for view in ['moving', 'H004_A005_1210M6']:
                p = root/frame/(view+'_comparison.png')
                panels[str(p)] = sha(p)
        review['reviewed_panel_sha256'] = panels
        atomic_json(review_path, review)
    # Require fresh independent geometry receipts for every changed canary.
    for root in ROOTS[1:4]:
        for p in root.glob('??????/geometry_result.json'):
            audit = read(p.parent/'independent_admission_audit.json')
            if audit['geometry_result_sha256'] != sha(p) or not audit['curvature_and_transfer_replay_exact']:
                raise ValueError('Missing matching independent geometry replay')
            fresh = root/'fresh_audit'/(p.parent.name+'.json')
            if audit['fresh_ray_audit_sha256'] != sha(fresh):
                raise ValueError('Changed fresh audit')
            if read(fresh)['checks'] != 124 or read(fresh)['qualified_veto_pixels'] != 0:
                raise ValueError('Incomplete fresh ray checks')
    testlog = DATA/'dec5_forearm_admission_tests_full.log'
    if '38 passed' not in testlog.read_text():
        raise ValueError('Expected focused tests did not pass')
    report = HERE.parent/'experiments/dec5_forearm_admission_order.md'
    retained = {str(p): sha(p) for root in ROOTS+[snapshots] for p in sorted(root.rglob('*')) if p.is_file()}
    # Never hash this process's still-open stdout log into its own manifest.
    retained.update({str(p): sha(p) for p in DATA.glob('dec5_forearm_admission*.log') if 'freeze' not in p.name})
    retained.update({str(p): sha(p) for p in DATA.glob('dec5_forearm_*ring*.log')})
    retained[str(report)] = sha(report)
    process = subprocess.check_output(['ps', '-eo', 'pid,etime,args'], text=True)
    workers = [line for line in process.splitlines() if any(name in line for name in [
        'scripts/study_forearm_production_delta.py prepare',
        'scripts/study_confidence_boundary_completion.py render',
        'scripts/audit_contrastive_forearm_guard.py --'])]
    if workers:
        raise ValueError('Relevant workers still running')
    check = dict(unix_time=time.time(), relevant_workers=[],
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], text=True).strip(),
        disk_free_bytes=shutil.disk_usage(DATA).free)
    atomic_json(MANIFEST, dict(status='controls_completed_not_production_promoted',
        frames=['001029','001033','001037'], artifact_free=False,
        shape_first='partial_improvement_001037_negligible_elsewhere',
        observed_ring_feather='rejected_both_001037_controls', tests_passed=38,
        metric_scope='fixed manual train forearm skin, not heldout face or full-frame',
        main_face_campaign_csv_unchanged=True, previous_manifests=previous,
        unchanged_published_hashes=published, producer_provenance=provenance,
        final_supervision=check, retained_hashes=retained, script_sha256=sha(__file__)))
    print('Frozen', len(retained), 'files; not production accepted', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    if a.check:
        data = read(MANIFEST)
        verify_files(data['retained_hashes'])
        verify_files(data['unchanged_published_hashes'])
        print('Retained controls and published movie hashes pass')
    else:
        freeze()
