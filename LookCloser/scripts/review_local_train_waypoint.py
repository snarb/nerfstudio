"""Record the main agent's actually inspected seven-time camera canary."""
from joint_temporal_texture import read, sha, atomic_json
from study_local_train_waypoint import OUTPUT, FRAMES


def record():
    findings = {
        '000899': ('fail', 'The periodic return exposes a substantially larger black under-chin/neck cutout behind the hand. Crown fragments remain. Reject the whole path, not merely this frame.'),
        '000929': ('pass', 'Local jaw remains continuous at this sample. Thin hand/face boundary and rough crown remain; not artifact-free approval.'),
        '001149': ('pass', 'Transition retains the face and jaw; pre-existing crown-side gaps and brown hair rim remain. No local jaw regression seen.'),
        '001169': ('pass', 'Changed view is visibly distinct; jaw/neck remains connected in the inspected crop. Rough hair rim and thin source seams remain.'),
        '001189': ('pass', 'Sideward late waypoint neighborhood hides the exposed front jaw region. Tiny contour notches and rough hair remain.'),
        '001193': ('pass', 'The isolated under-jaw fleck is not visible at the exact G/C pose. A thin under-chin/neck boundary and false sharp nose edge remain.'),
        '001197': ('pass', 'Late nearby pose retains the local avoidance benefit; rough hair and a thin neck boundary remain.'),
    }
    paths = [OUTPUT / 'review' / (f + '_head.png') for f in FRAMES]
    paths += [OUTPUT / 'review' / (f + '_jaw.png') for f in ['000899', '001193']]
    payload = dict(request_sha256=sha(OUTPUT / 'request.json'),
        status='rejected_periodic_early_jaw_regression',
        per_frame_status_scope='local_jaw_regression_gate_only_not_artifact_free_frame_quality',
        records=[dict(frame=f, status=findings[f][0], notes=findings[f][1]) for f in FRAMES],
        inspected_images=[dict(path=str(p), sha256=sha(p)) for p in paths],
        uninspected_panels='Other saved jaw panels and all overview panels were not used as visual evidence.',
        full_video_approved=False, geometry_improvement=False,
        source_quality_initial_gate_erratum='Inherited request initial_gate concerns parent texture validation only, not acceptance of this changed camera path.',
        why_rejected='A periodic displacement near the last actor instant wraps into early, different actor poses; exact train alignment at one time is not temporal surface coverage.',
        next_step='Constrain any replacement path using early and late actor poses jointly; retain the elevated early envelope. Do not hide 000899 regression or call this a repaired mesh.',
        script_sha256=sha(__file__))
    atomic_json(OUTPUT / 'visual_review.json', payload)
    print(payload['status'], 'actually inspected', len(paths), 'panels', flush=True)


if __name__ == '__main__': record()
