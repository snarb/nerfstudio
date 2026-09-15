"""Recompute saved depth-change counts and bind the actual eight-panel review."""
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json
from screen_wide_fringe_geometry import ROOT, VARIANTS


def audit():
    result = read(ROOT/'result.json')
    producer = Path(__file__).with_name('screen_wide_fringe_geometry.py')
    assert sha(producer) == result['script_sha256']
    assert result['cpu_only'] and not result['rgb_tested'] and not result['production_changed']
    for path, digest in result['inputs'].items(): assert sha(path) == digest
    assert {(r['frame'], r['variant']) for r in result['records']} == {
        (f, v) for f in ('001083', '001123') for v in VARIANTS}
    assert len(result['records']) == 8
    bindings = []
    for record in result['records']:
        out = ROOT/record['frame']/record['variant']
        assert read(out/'result.json') == record
        for name, digest in record['output_hashes'].items(): assert sha(out/name) == digest
        data = np.load(out/'depths.npz')
        assert set(data.files) == {'production', 'remove_only', 'replace'}
        for arm in data.files:
            d = data[arm]
            assert d.shape == (1920, 1080) and np.isfinite(d).all() and (d >= 0).all()
            assert Image.open(out/(arm+'.png')).size == (1080, 1920)
        old = data['production']; old_valid = old > 0
        for arm in ('remove_only', 'replace'):
            new = data[arm]; new_valid = new > 0
            counts = dict(lost=int(np.count_nonzero(old_valid & ~new_valid)),
                gained=int(np.count_nonzero(~old_valid & new_valid)),
                # Preserve the producer's float32 threshold operation exactly.
                # Subtracting first is not equivalent at one-ULP boundaries.
                deeper=int(np.count_nonzero(old_valid & new_valid & (new > old+.0001))),
                closer=int(np.count_nonzero(old_valid & new_valid & (new < old-.0001))))
            assert counts == record['changes'][arm]
        # Deleting triangles cannot produce a new or nearer surface.
        removed = data['remove_only']; valid = removed > 0
        assert not np.any(valid & ~old_valid)
        assert np.all(removed[valid] >= old[valid] - 1e-7)
        bindings.append(dict(frame=record['frame'], variant=record['variant'],
            panel=str(out/'head_comparison.png'), sha256=sha(out/'head_comparison.png')))
    atomic_json(ROOT/'audit.json', dict(status='integrity_and_removal_monotonicity_pass',
        records=8, depth_maps=24, result_sha256=sha(ROOT/'result.json'),
        script_sha256=sha(__file__), no_quality_score=True))
    atomic_json(ROOT/'visual_review.json', dict(status='partial_not_promoted', panels=bindings,
        reviewer='main LLM, all eight three-arm head panels actually viewed',
        findings=['Detached top fringe reduced, most clearly at 001123.',
            'Large head/cheek/neck shape remains visually consistent in these clay comparisons.',
            'Ragged hair silhouette and residual openings remain; this is not complete crown repair.',
            '001123 refined right arc retains a side hair/temple opening visible in both arms.',
            'Clay rendering cannot establish RGB seam, texture, held-out or temporal quality.'],
        production_promoted=False, rgb_tested=False))
    hashes = {str(p): sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'}
    hashes.update(result['inputs'])
    hashes[str(producer.resolve())] = sha(producer)
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    atomic_json(ROOT/'artifact_manifest.json', dict(hashes=hashes))
    for path, digest in read(ROOT/'artifact_manifest.json')['hashes'].items(): assert sha(path) == digest
    print('Rechecked', len(hashes), 'hashes; 24 finite maps, 8 monotone removal comparisons', flush=True)


if __name__ == '__main__': audit()
