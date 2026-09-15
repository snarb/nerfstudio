"""Seal the two-frame diagnostic; replay admission, not a quality approval."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json, cameras
from apply_measured_head_mask_completion import ROOT, MASKS, BASE, SOURCE, FRAMES
from transfer_close_boundary_completion import MOVIE
from study_jaw_repair_transfer import mask_votes
from review_jaw_repair_transfer import verified_image

DIAG = Path('/mnt/data/dec5_crown_completion_rejection')


def run():
    workers = ['refine_measured_head_masks.py', 'apply_measured_head_mask_completion.py',
               'render_measured_head_mask_completion.py', 'audit_uniform_measured_head_masks.py',
               'diagnose_crown_completion_rejection.py', 'review_uniform_measured_head_masks.py']
    ps = subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines()
    live = [p for p in ps if any('python scripts/' + n in p for n in workers)
            and '/bin/bash' not in p]
    assert not live, live
    audit = read(MASKS / 'audit.json')
    assert len(audit['records']) == 2
    hashes = {}; records = []
    for frame in FRAMES:
        root = ROOT / frame
        req = read(root / 'controller_request.json')
        for p, h in req['scripts'].items():
            assert sha(p) == h, p
            hashes[p] = h
        assert sha(BASE / frame / 'result.json') == req['matched_raw_result_sha256']
        assert sha(MASKS / frame / 'request.json') == req['masks_request_sha256']
        assert sha(MASKS / frame / 'result.json') == req['masks_result_sha256']
        assert not req['texture_masks_changed'] and not req['heldout_used']
        for n in ['local_raw.ply', 'proposal_evidence.npz', 'poisson_raw.ply']:
            assert sha(root / n) == sha(BASE / frame / n)
        names = read(MASKS / frame / 'cameras.json')
        masks = np.load(MASKS / frame / 'masks.npz')['masks']
        rows, _, _ = cameras(frame)
        v = np.asarray(o3d.io.read_triangle_mesh(str(root / 'local_raw.ply')).vertices)
        proposals = np.load(root / 'proposal_evidence.npz')['proposals']
        saved = np.load(root / 'admission/samples.npz')
        support, outside = mask_votes(v, proposals, rows, masks, names)
        np.testing.assert_array_equal(support, saved['mask_support'])
        np.testing.assert_array_equal(outside, saved['mask_outside'])
        np.testing.assert_array_equal(np.flatnonzero((support >= 2) & (outside == 0)), saved['semantic_ids'])
        geometry = root / 'interpolated' / frame
        g = read(geometry / 'result.json'); ga = read(geometry / 'audit.json')
        assert sha(geometry / 'mesh.ply') == ga['mesh_sha256'] == g['hashes']['mesh.ply']
        assert len(ga['native_ray_checks']) == 124
        assert all(r['trusted_free_pixels'] == 0 for r in ga['native_ray_checks'])
        base = read(SOURCE / frame / 'request.json')
        original = o3d.io.read_triangle_mesh(base['source_mesh'])
        final = o3d.io.read_triangle_mesh(str(geometry / 'mesh.ply'))
        np.testing.assert_array_equal(np.asarray(final.vertices)[:len(original.vertices)], np.asarray(original.vertices))
        np.testing.assert_array_equal(np.asarray(final.triangles)[:len(original.triangles)], np.asarray(original.triangles))
        for view in ['moving', 'native_unmasked']:
            roots = [MOVIE if view == 'moving' else ROOT / 'rgb' / frame / view / 'baseline',
                     ROOT / 'rgb' / frame / view / 'completion']
            receipts = []
            for rgb in roots:
                im, r = verified_image(rgb, frame); receipts.append(r)
                assert im.shape == (1920, 1080, 3) and np.isfinite(im).all()
                if view == 'native_unmasked':
                    q = read(rgb / 'request.json')
                    assert q['native_target_mask_disabled'] and q['texture_source_masks_unchanged']
                hashes[str(rgb / 'request.json')] = sha(rgb / 'request.json')
                for n, h in read(rgb / 'frames' / frame / 'complete.json')['hashes'].items():
                    hashes[str(rgb / 'frames' / frame / n)] = h
            for k in ['camera', 'source_cameras', 'fixed_exposure']:
                assert receipts[0][k] == receipts[1][k]
            assert receipts[0]['mesh_sha256'] == base['source_mesh_sha256']
            assert receipts[1]['mesh_sha256'] == ga['mesh_sha256']
        records.append(dict(frame=frame, same_raw_proposal=True, semantic_votes_replayed=True,
                            original_prefix_exact=True, native_guard_checks=124, added_triangles=g['added']))
    inspected = [ROOT / 'review' / f / (view + '_crown.png')
                 for f in FRAMES for view in ['moving', 'native_unmasked']]
    inspected += [ROOT / 'review' / f / 'native_unmasked_jaw.png' for f in FRAMES]
    inspected += list(MASKS.glob('*/review/*_native.png'))
    inspected += [DIAG / (f + '_' + view + '.png') for f in FRAMES for view in ['moving', 'native_train']]
    assert len(inspected) == 18
    atomic_json(ROOT / 'visual_review.json', dict(
        utc=datetime.now(timezone.utc).isoformat(), inspected_images={str(p): sha(p) for p in inspected},
        status='not_promoted_insufficient_visible_repair',
        notes='Measured masks recover plausible curls but dilation includes mixed boundary/background pixels. '
              'Four crown comparisons and two native jaw comparisons inspected: conspicuous crown gaps remain; '
              'no meaningful visible repair. Native RGB differs from GT in pre-existing hair detail/appearance. '
              'Mask refinement is not ground-truth segmentation; no full-video quality approval.',
        production_updated=False, full_video_approved=False))
    atomic_json(ROOT / 'audit.json', dict(records=records, mask_queries_independently_replayed=True,
                study_workers_terminal=True, production_updated=False, quality_metrics_computed=False))
    for directory in [ROOT, MASKS, DIAG]:
        for p in directory.rglob('*'):
            if p.is_file() and p.name != 'artifact_manifest.json':
                hashes[str(p)] = sha(p)
    for n in [*workers, Path(__file__).name]:
        p = Path(__file__).resolve().with_name(n); hashes[str(p)] = sha(p)
    atomic_json(ROOT / 'artifact_manifest.json', dict(hashes=hashes, production_updated=False,
                visual_approval=False))
    print('Sealed', len(hashes), 'hashes;', records, flush=True)


def check():
    hashes = read(ROOT / 'artifact_manifest.json')['hashes']
    for p, h in hashes.items():
        assert sha(p) == h, p
    print('Rechecked', len(hashes), 'hashes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    check() if parser.parse_args().check else run()
