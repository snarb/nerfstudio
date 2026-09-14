"""Bind explicit human/LLM sheet verdicts to checked, ordered render inputs."""
import argparse
from pathlib import Path

from joint_temporal_texture import read, sha, atomic_json


def finalize(output, notes_path):
    request = read(output / 'request.json')
    manifest = read(output / 'sheets.json')
    notes = read(notes_path)
    if manifest['request_sha256'] != sha(output / 'request.json'):
        raise ValueError('Changed review request')
    if sha(request['source_request']) != request['source_request_sha256']:
        raise ValueError('Changed source request')
    expected = [r['frame_id'] for r in request['inputs']]
    actual = [f for sheet in manifest['sheets'] for f in sheet['frame_ids']]
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError('Review inventory mismatch')
    if set(notes['sheets']) != {Path(s['path']).name for s in manifest['sheets']}:
        raise ValueError('Every sheet needs an explicit verdict')
    records = []
    for item in request['inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Changed source image')
    for sheet in manifest['sheets']:
        if sha(sheet['path']) != sheet['sha256']:
            raise ValueError('Changed reviewed sheet')
        verdict = notes['sheets'][Path(sheet['path']).name]
        if not verdict.strip():
            raise ValueError('Empty visual notes')
        for frame in sheet['frame_ids']:
            records.append(dict(frame_id=frame, sheet_sha256=sheet['sha256'],
                                visual_status='reviewed_with_residual_artifacts', notes=verdict))
    atomic_json(output / 'visual_review.json', dict(
        status='native_jaw_review_complete_not_artifact_free', reviewer='LLM direct image inspection',
        request_sha256=sha(output / 'request.json'), notes_sha256=sha(notes_path),
        script_sha256=sha(__file__), reviewed_frame_count=len(records), records=records,
        summary=notes['summary'], artifact_free=False, whole_actor_accepted=False,
        image_quality_metrics_computed=False))
    print(f'Verified {len(records)} reviewed times; no artifact-free claim', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notes', type=Path, required=True)
    args = parser.parse_args()
    finalize(args.output, args.notes)
