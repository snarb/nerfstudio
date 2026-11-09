#!/usr/bin/env python3
"""CPU-only R0 inference candidates: core UNION known cells, with optional ONE halo.

Does not instantiate a field or rewrite a checkpoint/EMA/worker mask. The exact
root-selected checkpoint step is mandatory; no automatic fallback to older data.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from build_marshawn_release_grid import witness_guard
from marshawn_release_common import authoritative_grid, point_cells
from probe_marshawn_occupancy import file_sha


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def derive(core, boxes, witnesses):
    """Keep native core, admit only witness cells; derive each halo from its base."""
    seed, minimal, protected_halo, ordinary_halo = witness_guard(core, boxes, witnesses)
    return dict(native=core.copy(), observed_voxel_seed=seed,
                minimal_guard=minimal, ordinary_halo1=ordinary_halo,
                witness_union_halo1=protected_halo)


def coverage(mask, boxes, xyz, subject_ids):
    _, inside, admitted = point_cells(xyz, mask, boxes)
    return dict(points=len(xyz), inside=int(inside.sum()),
                outside=int((~inside).sum()),
                all_point_misses=int((inside & ~admitted).sum()),
                all_point_miss_indices=np.flatnonzero(inside & ~admitted).tolist(),
                subject_points=len(subject_ids),
                subject_point_misses=int((inside[subject_ids] & ~admitted[subject_ids]).sum()))


def build(args):
    sources = {str(p.resolve()): file_sha(p) for p in
               [args.grid, args.grid_receipt, args.checkpoint, args.witnesses, args.subject]}
    core, active, _, boxes, policy = authoritative_grid(
        args.grid, args.grid_receipt, args.checkpoint)
    if policy['radius'] != 0 or not np.array_equal(active, core):
        raise ValueError('Requires actual stored R0 core; never re-dilate a stored R1 mask')
    if (int(policy['checkpoint_step']) != args.expected_step or
            args.checkpoint.name != 'step-%09d.ckpt' % args.expected_step):
        raise ValueError('Explicit selected checkpoint step mismatch; no fallback permitted')
    with np.load(args.grid, allow_pickle=False) as z:
        ema = z['ema'].copy()  # Preserve original bytes, dtype AND shape.
        resolution = z['resolution'].copy()
    points = np.load(args.witnesses, allow_pickle=False)['xyz']
    subject = np.load(args.subject, allow_pickle=False)['xyz']
    if points.shape != (9022, 3):
        raise ValueError('Exactly9022 original normalized-world train witnesses required')
    distance, subject_ids = cKDTree(points).query(subject)
    subject_ids = np.unique(subject_ids)
    if not np.isfinite(distance).all() or distance.max() > 1e-8 or len(subject_ids) != 495:
        raise ValueError('Exactly495 unique original subject witnesses required')
    masks = derive(core, boxes, points)
    profiles = ['native', 'minimal_guard', 'ordinary_halo1', 'witness_union_halo1']
    counts = {name: dict(active_cells=int(masks[name].sum()),
                        added_over_native=int((masks[name] & ~core).sum()),
                        **coverage(masks[name], boxes, points, subject_ids))
              for name in profiles}
    for name in ['minimal_guard', 'witness_union_halo1']:
        if counts[name]['all_point_misses'] or counts[name]['subject_point_misses']:
            raise ValueError('Protected candidate failed exact known-point coverage')
    mvs = None
    if args.mvs:
        sources[str(args.mvs.resolve())] = file_sha(args.mvs)
        xyz = np.load(args.mvs, allow_pickle=False)['xyz']
        mvs = {name: coverage(masks[name], boxes, xyz, np.empty(0, dtype=int))
               for name in profiles}
    args.output.mkdir(parents=True, exist_ok=False)
    policies = {
        'minimal_guard': 'actual GPU R0 core UNION9022 observed witness cells; NO halo',
        'witness_union_halo1': 'actual GPU R0 core UNION9022 observed witness cells, then exactly ONE halo',
    }
    records = {}
    for name, description in policies.items():
        folder = args.output/name
        folder.mkdir()
        archive = folder/'grid.npz'
        np.savez_compressed(archive, core=core, active=masks[name], ema=ema,
            observed_voxel_seed=masks['observed_voxel_seed'],
            protected_core=masks['minimal_guard'],
            ordinary_halo1=masks['ordinary_halo1'],
            witness_union_halo1=masks['witness_union_halo1'],
            world_aabb=boxes[0].reshape(2, 3), cascade_aabbs=boxes,
            resolution=resolution)
        with np.load(archive, allow_pickle=False) as result:
            if (result['ema'].shape != ema.shape or result['ema'].dtype != ema.dtype or
                    result['ema'].tobytes() != ema.tobytes() or
                    result['core'].tobytes() != core.tobytes()):
                raise ValueError('Output changed authoritative core or EMA')
        record = dict(profile=name, policy=description, inference_only=True,
            source_policy=policy, checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=sources[str(args.checkpoint.resolve())],
            source_grid_sha256=sources[str(args.grid.resolve())],
            source_receipt_sha256=sources[str(args.grid_receipt.resolve())],
            grid_archive_sha256=file_sha(archive), selected_step=args.expected_step,
            witnesses_sha256=sources[str(args.witnesses.resolve())],
            subject_sha256=sources[str(args.subject.resolve())],
            script_sha256=file_sha(__file__), core_cells=int(core.sum()),
            observed_seed_cells=int(masks['observed_voxel_seed'].sum()),
            added_seed_cells=int((masks['minimal_guard'] & ~core).sum()),
            candidate_active_cells=int(masks[name].sum()),
            ordinary_halo1_cells=int(masks['ordinary_halo1'].sum()),
            added_cells_over_ordinary_halo1=int((masks[name] & ~masks['ordinary_halo1']).sum()),
            candidate_all_point_misses=counts[name]['all_point_misses'],
            candidate_subject_point_misses=counts[name]['subject_point_misses'],
            ema_bytes_sha256=array_sha(ema), ema_shape=list(ema.shape),
            ema_dtype=str(ema.dtype), authoritative_native_core_unchanged=True,
            ema_unchanged=True, learned_field_unchanged=True,
            checkpoint_rewritten=False, field_instantiated=False,
            interpretation='Conservative traversal protection only; not physical occupied/free labels')
        (folder/'receipt.json').write_text(json.dumps(record, indent=2)+'\n')
        records[name] = dict(grid=str(archive.resolve()), receipt=str((folder/'receipt.json').resolve()),
                             grid_sha256=file_sha(archive))
    # Detect concurrent source overwrites; never emit final PASS for unstable input.
    if any(file_sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Source changed during CPU derivation')
    report = dict(passed=True, selected_step=args.expected_step,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_sha(args.checkpoint),
        source_files_sha256=sources, script_sha256=file_sha(__file__),
        counts=counts, MVS_candidate_coverage=mvs, MVS_candidates_used_as_seeds=False,
        outputs=records, native_core_bytes_sha256=array_sha(core),
        ema_bytes_sha256=array_sha(ema), source_files_unchanged=True,
        field_instantiated=False, optimizer_steps=0, worker_mask_changed=False,
        no_halo_minimal_added_cells=int((masks['minimal_guard'] & ~core).sum()),
        protected_halo_added_over_ordinary=int((masks['witness_union_halo1'] & ~masks['ordinary_halo1']).sum()),
        renderer_note='Use explicit temporary mask override with restore. Legacy halo-only candidate loader is valid only for witness_union_halo1, not minimal_guard.',
        limitations=['Witness-cell protection includes uncertain fitted features and silhouette junctions.',
                     'Zero known misses does not certify hidden geometry, full volume free space or clean RGB.',
                     'No HDR/field query, renderer or manual atlas inspection is performed by this CPU builder.',
                     'Selected checkpoint must be explicit; no substitution of2k for planned8k or terminal best.'])
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['grid', 'grid-receipt', 'checkpoint', 'witnesses', 'subject', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-step', type=int, required=True)
    p.add_argument('--mvs', type=Path)
    a = p.parse_args()
    if not 0 <= a.expected_step < 22000:
        p.error('Expected step must be explicit and below22000')
    report = build(a)
    print(json.dumps(dict(selected_step=report['selected_step'], counts=report['counts'])))


if __name__ == '__main__':
    main()
