"""Publish the opt-in texture replay with explicit residual failures and hashes."""
from pathlib import Path
import argparse
import hashlib
import os
import shutil
import subprocess
import zipfile

from joint_temporal_texture import read, sha, atomic_json
from run_view_consistent_dynamic_video import OUTPUT
from finalize_wide_dynamic_flight import audit, record_review
from review_early_texture_video import review


def record(output):
    notes = read(output / 'review_notes.json')
    for group, (status, text) in notes['overview'].items():
        record_review(output, group, text, status)
    review(output, notes['jaw'])


def validate_reviews(output, ids):
    if len(ids) != 150 or len(set(ids)) != 150:
        raise ValueError('Exactly 150 distinct actor times required')
    jaw = read(output / 'jaw_review/visual_review.json')
    if not jaw['complete'] or [r['frame_id'] for r in jaw['records']] != ids:
        raise ValueError('All 150 native jaw crops must actually be reviewed')
    for row in jaw['records']:
        if row['artifact_free'] or not row['notes'] or sha(output/'frames'/row['frame_id']/'frame.png') != row['render_sha256']:
            raise ValueError('Invalid native review')
    for sheet in read(output/'jaw_review/sheets.json')['sheets']:
        if sha(sheet['path']) != sheet['sha256']:
            raise ValueError('Changed native review sheet')
    video = read(output/'video_manifest.json')
    if video['encoded_visual_status'] != 'reviewed_with_known_failures' or len(video['review_evidence']) != 15:
        raise ValueError('Actual decoded MP4 review required')
    for row in video['review_evidence']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Changed encoded evidence')
    for name, row in video['videos'].items():
        if sha(output/name) != row['sha256']:
            raise ValueError('Changed encoded video')
    return video


def publish(output):
    checked = audit(output, require_reviews=True)
    request = read(output/'request.json')
    ids = request['ordered_frame_ids']
    validate_reviews(output, ids)
    geometry = read(output/'head_geometry_audit.json')
    if geometry['frames'] != 150 or geometry['status'] != 'preservation_and_locality_pass' or geometry['request_sha256'] != sha(output/'request.json'):
        raise ValueError('Independent geometry preservation audit required')
    parent = read('/mnt/data/dec5_phase30_early_texture_dynamic_150/request.json')
    for row, old in zip(request['inventory'], parent['inventory'], strict=True):
        for key in ['frame_id', 'camera', 'mesh_sha256', 'metadata_sha256']:
            if row[key] != old[key]:
                raise ValueError('Texture-only replay changed camera/time/geometry')
    temporary = output/'frames.partial.zip'
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
        for frame in ids:
            archive.write(output/'frames'/frame/'frame.png', f'frames/{frame}.png')
    with zipfile.ZipFile(temporary) as archive:
        if archive.namelist() != [f'frames/{f}.png' for f in ids]:
            raise ValueError('Archive order/inventory mismatch')
        for frame in ids:
            if hashlib.sha256(archive.read(f'frames/{frame}.png')).hexdigest() != checked['render_hashes'][frame]:
                raise ValueError('Archive checksum mismatch')
    os.replace(temporary, output/'frames.zip')
    scripts = Path(__file__).resolve().parent
    snapshot = output/'script_snapshot'; snapshot.mkdir(exist_ok=True)
    for name, digest in request['script_hashes'].items():
        if Path(name).name != name or sha(scripts/name) != digest:
            raise ValueError('Changed implementation')
        shutil.copyfile(scripts/name, snapshot/name)
    for name in ['finalize_view_consistent_video.py', 'finalize_wide_dynamic_flight.py', 'review_early_texture_video.py', 'audit_head_completion_geometry.py']:
        shutil.copyfile(scripts/name, snapshot/name)
    shutil.copyfile(scripts.parent/'experiments/dec5_head_source_quality.md', output/'report.md')
    hashes = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*'))
              if p.is_file() and p.name not in {'publication.json', 'supervisor.lock', 'publication.log'}
              and '.partial.' not in p.name and '__pycache__' not in p.parts}
    atomic_json(output/'publication.json', dict(
        status='texture_improvement_with_known_geometry_failures', artifact_free=False,
        source_time_count=150, camera_and_actor_dynamic=True, geometry_changed=False,
        hash_count=len(hashes), hashes=hashes,
        heldout_scope='One face benchmark used for model selection; not a 150-frame metric aggregate',
        repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    print(f'Published known-failure candidate: {output}', flush=True)


def check(output):
    publication = read(output/'publication.json')
    for name, digest in publication['hashes'].items():
        if sha(output/name) != digest:
            raise ValueError(f'Publication checksum mismatch: {name}')
    print(f"Verified {len(publication['hashes'])} retained checksums", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['record', 'publish', 'check'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    {'record': record, 'publish': publish, 'check': check}[args.action](args.output)
