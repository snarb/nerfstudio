#!/usr/bin/env python3
"""Derived inference admission: actual GPU core UNION train witnesses, ONE halo.

Keeps native core and EMA unchanged; no checkpoint or learned field is rewritten.
The resulting cells are conservative traversal admission, not geometry labels.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.spatial import cKDTree
from marshawn_release_common import authoritative_grid,point_cells
from marshawn_occupancy_audit import validate_grid,dilate_one
from probe_marshawn_occupancy import file_sha


def witness_guard(core,boxes,points):
    core,boxes=validate_grid(core,boxes)
    if len(core)!=1:raise ValueError('Explicit single base cascade required')
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Finite world XYZ points required')
    indices,inside,_=point_cells(points,core,boxes)
    if not inside.all():raise ValueError('Witnesses outside selected AABB')
    seed=np.zeros_like(core);seed[(0,*indices.T)]=True
    protected=core|seed
    return seed,protected,dilate_one(protected),dilate_one(core)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['grid','grid-receipt','checkpoint','witnesses','subject','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();core,active,ema,boxes,source=authoritative_grid(a.grid,a.grid_receipt,a.checkpoint)
    if source['radius']!=0:raise ValueError('This candidate starts from stored radius0 core, never stored halo1')
    points=np.load(a.witnesses)['xyz'];subject=np.load(a.subject)['xyz'];distance,subject_ids=cKDTree(points).query(subject)
    if len(points)!=9022 or distance.max()>1e-8:raise ValueError('Exact9022 world witnesses/subject match required')
    subject_ids=np.unique(subject_ids)
    if len(subject_ids)!=495:raise ValueError('Expected495 unique subject witnesses')
    seed,protected,candidate,ordinary=witness_guard(core,boxes,points)
    _,inside,ordinary_occupied=point_cells(points,ordinary,boxes)
    _,_,candidate_occupied=point_cells(points,candidate,boxes)
    assert candidate_occupied.all()
    a.output.mkdir(parents=True,exist_ok=False);archive=a.output/'grid.npz'
    np.savez_compressed(archive,core=core,observed_voxel_seed=seed,protected_core=protected,
        active=candidate,ordinary_halo1=ordinary,ema=ema,
        world_aabb=boxes[0].reshape(2,3),cascade_aabbs=boxes,resolution=np.array(core.shape[1:]))
    record=dict(policy='actual GPU native core UNION9022 observed train witness voxels, then exactly ONE cube-radius1 halo',
        inference_only=True,authoritative_native_core_unchanged=True,ema_unchanged=True,
        learned_field_unchanged=True,checkpoint_rewritten=False,checkpoint=str(a.checkpoint),
        checkpoint_sha256=file_sha(a.checkpoint),source_grid_sha256=file_sha(a.grid),
        source_receipt_sha256=file_sha(a.grid_receipt),source_policy=source,
        witnesses_sha256=file_sha(a.witnesses),subject_sha256=file_sha(a.subject),script_sha256=file_sha(__file__),
        grid_archive_sha256=file_sha(archive),world_aabb=boxes[0].reshape(2,3).tolist(),
        core_cells=int(core.sum()),observed_seed_cells=int(seed.sum()),added_seed_cells=int((protected&~core).sum()),
        ordinary_halo1_cells=int(ordinary.sum()),candidate_active_cells=int(candidate.sum()),
        added_cells_over_ordinary_halo1=int((candidate&~ordinary).sum()),
        ordinary_halo1_point_misses=np.flatnonzero(~ordinary_occupied).tolist(),
        candidate_all_point_misses=int((~candidate_occupied).sum()),
        candidate_subject_point_misses=int((~candidate_occupied[subject_ids]).sum()),
        ema_bytes_sha256=hashlib.sha256(ema.tobytes()).hexdigest(),
        interpretation='Admission safety around observed train features. Guard/unknown cells are not claimed physically occupied.',
        limitations=['Fitted SfM features do not certify complete geometry or hidden volume.',
            'This admission correction cannot remove erroneous field density or camera/hardware collisions.'])
    (a.output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['core_cells','added_seed_cells','ordinary_halo1_cells','candidate_active_cells','added_cells_over_ordinary_halo1','ordinary_halo1_point_misses','candidate_all_point_misses','candidate_subject_point_misses']}))


if __name__=='__main__':main()
