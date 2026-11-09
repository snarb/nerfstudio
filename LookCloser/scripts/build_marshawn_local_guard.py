#!/usr/bin/env python3
"""CPU extension: expand ONLY newly missing witness cells, never the whole core."""
from pathlib import Path
import argparse,json
import numpy as np
from build_marshawn_minimal_guard import build,coverage
from marshawn_occupancy_audit import dilate_one
from probe_marshawn_occupancy import file_sha
from scipy.spatial import cKDTree


def local_guard(core, observed_voxel_seed):
    if (core.dtype != np.bool_ or observed_voxel_seed.dtype != np.bool_ or
            core.shape != observed_voxel_seed.shape):
        raise ValueError('Matching bool native core and witness-cell masks required')
    new_seed = observed_voxel_seed & ~core
    # Deliberately NOT dilate(core|seed), and NOT dilate(all observed seed cells).
    return core | dilate_one(new_seed), new_seed


def all_witness_guard(core, observed_voxel_seed):
    """Expand observed neighborhoods once, including witnesses already in core."""
    if (core.dtype != np.bool_ or observed_voxel_seed.dtype != np.bool_ or
            core.shape != observed_voxel_seed.shape):
        raise ValueError('Matching bool native core and witness-cell masks required')
    return core | dilate_one(observed_voxel_seed)


def build_three(args):
    report = build(args)  # Existing strict R0/step/source9022/495 validation.
    source = args.output/'minimal_guard/grid.npz'
    with np.load(source, allow_pickle=False) as z:
        data={key:z[key].copy() for key in z.files}
    core,seed=data['core'],data['observed_voxel_seed']
    active,new_seed=local_guard(core,seed)
    if ((data['active'] & ~active).any() or
            (active & ~data['witness_union_halo1']).any()):
        raise ValueError('Local mask must contain minimal union and fit inside global ONEhalo')
    points=np.load(args.witnesses,allow_pickle=False)['xyz']
    subject=np.load(args.subject,allow_pickle=False)['xyz']
    _,subject_ids=cKDTree(points).query(subject);subject_ids=np.unique(subject_ids)
    cov=coverage(active,data['cascade_aabbs'],points,subject_ids)
    if cov['all_point_misses'] or cov['subject_point_misses']:
        raise ValueError('Local mask missed original witnesses')
    folder=args.output/'local_new_seed_halo1';folder.mkdir()
    archive=folder/'grid.npz';data.update(active=active,new_seed_cells=new_seed,
        local_new_seed_halo1=active)
    np.savez_compressed(archive,**data)
    with np.load(archive,allow_pickle=False) as z:
        if (z['core'].tobytes()!=core.tobytes() or
                z['ema'].tobytes()!=data['ema'].tobytes() or
                z['ema'].shape!=data['ema'].shape or z['ema'].dtype!=data['ema'].dtype):
            raise ValueError('Local archive changed authoritative core or EMA')
    base=json.loads((args.output/'minimal_guard/receipt.json').read_text())
    record=dict(base,profile='local_new_seed_halo1',
        policy='actual GPU R0 core UNION exactlyONEhalo(only observed-witness cells absent from core)',
        grid_archive_sha256=file_sha(archive),candidate_active_cells=int(active.sum()),
        local_new_seed_cells=int(new_seed.sum()),
        local_added_over_native=int((active&~core).sum()),
        local_added_over_minimal=int((active&~data['protected_core']).sum()),
        candidate_all_point_misses=0,candidate_subject_point_misses=0,
        script_sha256=file_sha(__file__),
        dependency_script_sha256=file_sha(Path(__file__).with_name('build_marshawn_minimal_guard.py')),
        renderer_note='Explicit temporary mask override required; this is not a stored authoritative R1 grid.')
    (folder/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    report['counts']['local_new_seed_halo1']=dict(active_cells=int(active.sum()),
        added_over_native=int((active&~core).sum()),
        added_over_minimal=int((active&~data['protected_core']).sum()),**cov)
    report['outputs']['local_new_seed_halo1']=dict(grid=str(archive.resolve()),
        receipt=str((folder/'receipt.json').resolve()),grid_sha256=file_sha(archive))
    if args.mvs:
        xyz=np.load(args.mvs,allow_pickle=False)['xyz']
        report['MVS_candidate_coverage']['local_new_seed_halo1']=coverage(
            active,data['cascade_aabbs'],xyz,np.empty(0,dtype=int))
    if any(file_sha(p)!=h for p,h in report['source_files_sha256'].items()):
        raise ValueError('Source changed while preparing local policy')
    report.update(script_sha256=file_sha(__file__),
        base_comparison_sha256=file_sha(args.output/'comparison.json'),
        local_definition='core | dilate_one(observed_seed & ~core)',
        local_new_seed_cells=int(new_seed.sum()),
        local_expansion_centers_are_new_seed_only=True,
        maximum_local_addition_bound=27*int(new_seed.sum()),
        profile_order=['native','minimal_guard','local_new_seed_halo1','witness_union_halo1'],
        geometry_criteria='Point coverage only. Neighbor-ray surface-band repair, RGB/opacity, BODY/free-prefix and manual atlas remain to be measured on this selected field; no8k result is transferred.',
        full_HD_rendered=False,field_instantiated=False)
    (args.output/'three_policy_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def build_four(args):
    report = build_three(args)
    with np.load(args.output/'minimal_guard/grid.npz', allow_pickle=False) as z:
        data = {key:z[key].copy() for key in z.files}
    core, seed = data['core'], data['observed_voxel_seed']
    active = all_witness_guard(core, seed)
    local_new, _ = local_guard(core, seed)
    if ((local_new & ~active).any() or
            (active & ~data['witness_union_halo1']).any()):
        raise ValueError('Local witness mask must contain local-new and fit inside global ONEhalo')
    points=np.load(args.witnesses,allow_pickle=False)['xyz']
    subject=np.load(args.subject,allow_pickle=False)['xyz']
    _,subject_ids=cKDTree(points).query(subject);subject_ids=np.unique(subject_ids)
    cov=coverage(active,data['cascade_aabbs'],points,subject_ids)
    if cov['all_point_misses'] or cov['subject_point_misses']:
        raise ValueError('Local all-witness mask missed original witnesses')
    folder=args.output/'local_all_witness_halo1';folder.mkdir()
    archive=folder/'grid.npz'
    original_ema=data['ema'].copy()
    data.update(active=active,local_all_witness_halo1=active)
    np.savez_compressed(archive,**data)
    with np.load(archive,allow_pickle=False) as z:
        if (z['core'].tobytes()!=core.tobytes() or
                z['ema'].tobytes()!=original_ema.tobytes() or
                z['ema'].shape!=original_ema.shape or z['ema'].dtype!=original_ema.dtype):
            raise ValueError('Local archive changed authoritative core or EMA')
    base=json.loads((args.output/'minimal_guard/receipt.json').read_text())
    record=dict(base,profile='local_all_witness_halo1',
        policy='actual GPU R0 core UNION exactlyONEhalo(ALL observed-witness cells)',
        grid_archive_sha256=file_sha(archive),candidate_active_cells=int(active.sum()),
        local_expansion_centers=int(seed.sum()),
        local_added_over_native=int((active&~core).sum()),
        candidate_all_point_misses=0,candidate_subject_point_misses=0,
        script_sha256=file_sha(__file__),
        renderer_note='Temporary inference mask only. Core and EMA unchanged; guard is not physical-occupancy truth.')
    (folder/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    report['counts']['local_all_witness_halo1']=dict(active_cells=int(active.sum()),
        added_over_native=int((active&~core).sum()),
        added_over_minimal=int((active&~data['protected_core']).sum()),**cov)
    report['outputs']['local_all_witness_halo1']=dict(grid=str(archive.resolve()),
        receipt=str((folder/'receipt.json').resolve()),grid_sha256=file_sha(archive))
    if args.mvs:
        xyz=np.load(args.mvs,allow_pickle=False)['xyz']
        report['MVS_candidate_coverage']['local_all_witness_halo1']=coverage(
            active,data['cascade_aabbs'],xyz,np.empty(0,dtype=int))
    if any(file_sha(p)!=h for p,h in report['source_files_sha256'].items()):
        raise ValueError('Source changed while preparing all-witness policy')
    report.update(script_sha256=file_sha(__file__),
        local_all_definition='core | dilate_one(observed_seed)',
        local_all_expansion_centers=int(seed.sum()),
        three_policy_comparison_sha256=file_sha(args.output/'three_policy_comparison.json'),
        profile_order=['native','minimal_guard','local_new_seed_halo1',
            'local_all_witness_halo1','witness_union_halo1'])
    (args.output/'four_policy_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['grid','grid-receipt','checkpoint','witnesses','subject','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-step',type=int,required=True)
    p.add_argument('--mvs',type=Path)
    a=p.parse_args()
    if not 0<=a.expected_step<22000:p.error('Explicit selected step below22000 required')
    r=build_four(a);print(json.dumps(dict(selected_step=r['selected_step'],counts=r['counts'])))
