#!/usr/bin/env python3
"""All XYZ slices for the declared observed-witness inference guard."""
from pathlib import Path
import argparse,json
import numpy as np
from marshawn_occupancy_audit import validate_grid,dilate_one,write_atlas
from probe_marshawn_occupancy import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['candidate','receipt','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();receipt=json.loads(a.receipt.read_text())
    if file_sha(a.candidate)!=receipt['grid_archive_sha256']:raise ValueError('Candidate archive hash mismatch')
    with np.load(a.candidate,allow_pickle=False) as data:
        active,boxes=validate_grid(data['active'].copy(),data['cascade_aabbs'].copy())
        core,protected,seed=[data[key].copy() for key in ['core','protected_core','observed_voxel_seed']]
    if not np.array_equal(protected,core|seed) or not np.array_equal(active,dilate_one(protected)):
        raise ValueError('Candidate is not observed union followed by exactly one halo')
    labels=np.zeros(active.shape,np.uint8);labels[active&core]=1
    labels[active&~protected]=2;labels[active&protected&~core]=3
    a.output.mkdir(parents=True,exist_ok=False);pages=write_atlas(active,boxes,a.output,2,labels)
    path=a.output/'index.html';text=path.read_text().replace('With EMA labels:','With authoritative source labels:')
    text=text.replace('blue admitted threshold core','blue actual unchanged GPU native core').replace('amber declared radius1 guard','amber one-cell margin after observed union').replace('gray other admission','gray observed train witness protection seed')
    path.write_text(text)
    for page in pages:page['sha256']=file_sha(a.output/page['path'])
    report=dict(candidate_sha256=file_sha(a.candidate),candidate_receipt_sha256=file_sha(a.receipt),
        candidate_policy=receipt,script_sha256=file_sha(__file__),shape_lxyz=list(active.shape),aabbs=boxes.tolist(),
        core_cells=int(core.sum()),observed_added_seed_cells=int((protected&~core).sum()),
        guard_cells=int((active&~protected).sum()),active_cells=int(active.sum()),
        skipped_unverified=int((~active).sum()),pages=pages,
        palette=dict(blue='Actual GPU native core unchanged',gray='Added observed train-witness protection seed',
            amber='ONE cube halo after native-core/observed-seed union',black='Skipped, geometry unverified'),
        warning='No label proves complete physical geometry. No CPU EMA core reconstruction, carving or hidden-volume claim.')
    (a.output/'atlas.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(pages=len(pages),active_cells=int(active.sum()))))


if __name__=='__main__':main()
