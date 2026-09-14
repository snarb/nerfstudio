"""Audit bounded color proposals; an optional review binds human/LLM notes to hashes."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json
from bound_temporal_gradient_offset import bounded_rgb


def audit(output, notes=None):
    receipt = read(output/'result.json')
    source = Path(receipt['source_result']).parent
    for path, digest in [(source/'result.json', receipt['source_result_sha256']),
                         (source/'offset.npz', receipt['source_offset_sha256']),
                         (source/'baseline.png', receipt['baseline_sha256'])]:
        if sha(path) != digest:
            raise ValueError('Changed bounded proposal input')
    for name, digest in receipt['hashes'].items():
        if sha(output/name) != digest:
            raise ValueError('Changed bounded artifact')
    before = np.array(Image.open(source/'baseline.png'))
    after = np.array(Image.open(output/'corrected.png'))
    data = np.load(source/'offset.npz')
    support = (data['depth'] > 0) & (data['selection'] >= 0)
    expected = bounded_rgb(before, data['offset'].transpose(1,2,0), support,
                           receipt['maximum_display_channel_ratio'])
    if not np.array_equal(after, expected):
        raise ValueError('Bounded proposal cannot be reproduced')
    if not np.array_equal(before[~support], after[~support]):
        raise ValueError('Unsupported RGB changed')
    if ((before > 0) & (after == 0)).any():
        raise ValueError('An originally nonzero channel became black')
    result = dict(result_sha256=sha(output/'result.json'), corrected_sha256=sha(output/'corrected.png'),
                  exact_reproduction=True, unsupported_rgb_unchanged=True,
                  newly_zero_channels=0, audit_script_sha256=sha(__file__),
                  scope='numerical safety, not geometry or GT fidelity acceptance')
    atomic_json(output/'bounded_audit.json', result)
    if notes:
        verdict = read(notes)
        if verdict.get('actually_viewed_native') is not True or not verdict.get('notes'):
            raise ValueError('Review requires actual native inspection and observations')
        if verdict['visual_status'] not in {'pass','fail','uncertain'}:
            raise ValueError('Invalid review verdict')
        verdict.update(audit_sha256=sha(output/'bounded_audit.json'),
                       reviewed_images={str(output/n):sha(output/n) for n in ['corrected.png','comparison.png']},
                       notes_sha256=sha(notes))
        atomic_json(output/'visual_review.json', verdict)
    print(result, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--notes', type=Path)
    a=p.parse_args(); audit(a.output, a.notes)
