"""Freeze the actually reviewed earlier-time transfer, not movie approval."""
import argparse
from pathlib import Path

from joint_temporal_texture import atomic_json, read, sha
from run_neighborhood_completion_transfer import ROOT, SOURCE
from review_jaw_repair_transfer import verified_image
from study_confidence_depth_prior import REGIONS

FRAMES = ('001083', '001123')
MANIFEST = ROOT / 'head_transfer_artifact_manifest.json'


def run(check=False):
    if check:
        files = read(MANIFEST)['files']
        for path, digest in files.items():
            if sha(path) != digest:
                raise ValueError('Changed artifact: ' + path)
        print('Verified', len(files), 'hashes')
        return
    if MANIFEST.exists():
        raise ValueError('Already frozen')
    external = set()
    for frame in FRAMES:
        root = ROOT / frame
        folder = root / 'interpolated' / frame
        audit = read(folder / 'audit.json')
        result = read(folder / 'result.json')
        assert audit['mesh_sha256'] == sha(folder / 'mesh.ply')
        assert audit['original_prefix_exact'] and audit['local_certificates_recomputed']
        assert len(audit['native_ray_checks']) == 124
        assert all(r['trusted_free_pixels'] == 0 for r in audit['native_ray_checks'])
        assert result['observed_guard_passed']
        for name, digest in result['hashes'].items():
            assert sha(folder / name) == digest
        review = read(root / 'head_review/result.json')
        assert review['frame'] == frame and not review['heldout_used']
        assert not review['full_frame_quality_metrics']
        assert review['script_sha256'] == sha(Path(__file__).with_name('study_head_neighborhood_transfer.py'))
        for record in review['records']:
            for path, digest in record['inputs'].items():
                assert sha(path) == digest
                render_root = Path(path).parents[2]
                verified_image(render_root, frame)
                request = read(render_root / 'request.json')
                assert request['same_footprint_for_both_geometry_variants']
                assert len(request['inventory']) == 1
                row = request['inventory'][0]
                assert row['frame_id'] == frame and sha(row['mesh']) == row['mesh_sha256']
                external.add(Path(row['mesh']))
                for name, digest in request['script_hashes'].items():
                    source = Path(__file__).with_name(name)
                    assert sha(source) == digest
                    external.add(source)
        panels = [root / 'head_review' / (name + '.png') for name in (
            'hair_native', 'face_skin_native', 'moving_head',
            'F004_E005_1210FP_head', REGIONS[frame]['camera'] + '_head')]
        notes = (
            'Crown breaks and detached fringe persist; moving and F/E RGB byte-identical. '
            'Region view has 42 fewer zero-depth hair-polygon pixels, not a resolved crown.'
            if frame == '001083' else
            'All three RGB pairs byte-identical. Broad crown opening and cheek/neck dark '
            'boundary remain. Zero interior-face misses do not certify the boundary.'
        )
        atomic_json(root / 'head_review/visual_review.json', dict(
            reviewer='main_agent', status='fail_broad_artifact_removal', notes=notes,
            inspected={str(p): sha(p) for p in panels},
            no_new_conspicuous_defect_observed=True, video_updated=False,
            coverage_counts_are_not_psnr_ssim_lpips=True))
        review['visual_status'] = 'reviewed_residual_artifacts'
        atomic_json(root / 'head_review/result.json', review)
        for name in ('request.json', 'result.json', 'mesh.ply', 'evidence.npz'):
            external.add(SOURCE / frame / name)
        for suffix in ('', '_rgb', '_audit'):
            external.add(Path('/mnt/data') / ('dec5_neighborhood_transfer_' + frame + suffix + '.log'))
        external.add(Path('/mnt/data') / ('dec5_neighborhood_head_' + frame + '_region.log'))
    repo = Path(__file__).parents[1]
    external.update([Path(__file__), Path(__file__).with_name('study_head_neighborhood_transfer.py'),
                     repo / 'experiments/dec5_head_neighborhood_transfer.md'])
    files = {str(p): sha(p) for frame in FRAMES for p in (ROOT / frame).rglob('*') if p.is_file()}
    files.update({str(p): sha(p) for p in external})
    atomic_json(MANIFEST, dict(frames=FRAMES, files=files,
        status='reviewed_negative_broad_transfer', video_updated=False,
        native_checks=248, focused_tests_passed=9, full_frame_quality_metrics=False))
    print('Frozen', len(files), 'hashes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    run(parser.parse_args().check)
