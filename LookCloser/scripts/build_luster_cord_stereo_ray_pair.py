#!/usr/bin/env python3
"""Build matched train-only lower-cord ray manifests with/without stereo depth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build(audit_path: Path, fit_path: Path, checkpoint: Path, output_prefix: Path,
          fractions: tuple[int,...]) -> None:
    audit=json.loads(audit_path.read_text())
    fit=json.loads(fit_path.read_text())
    if (audit.get('train_only') is not True or audit.get('masks_used') is not False
            or audit['fit_sha256']!=sha(fit_path)
            or audit['transforms_sha256']!=fit['transforms_sha256']
            or audit['heldout_camera_not_used']!=fit['validation_camera_id']):
        raise ValueError('Stereo audit is not pinned to the train-only cord fit')
    train=set.intersection(*(set(fit['fit_camera_ids'][key])
                             for key in ('junction','first_bead')))
    if len(set(fractions))!=len(fractions) or not fractions:
        raise ValueError('Need distinct selected lower-cord fractions')
    pairs=[]
    points={}
    evidence={}
    for index in fractions:
        if not 0<=index<len(audit['rows']):
            raise ValueError('Invalid stereo fraction index')
        row=audit['rows'][index]
        if not row['provisional_high_confidence'] or row['candidate_offset_normalized']>=0:
            raise ValueError('Selected stereo point is not a supported front-depth candidate')
        offset=np.asarray(audit['normalized_depth_offsets'])
        take=int(np.argmin(abs(offset-row['candidate_offset_normalized'])))
        main=row['patch_scores']['9']['cameras']
        supported=[int(camera_id) for camera_id in audit['target_cameras']
                   if (main[str(camera_id)]['curve'][take]>.5
                       and main[str(camera_id)]['best_ncc']-main[str(camera_id)]['curve'][take]<.15)]
        if (len(supported)<3 or audit['reference_camera'] not in train
                or not set(supported).issubset(train)
                or audit['reference_camera'] in supported):
            raise ValueError('Stereo point lacks three independent training witnesses')
        cameras=sorted({audit['reference_camera'],*supported})
        pairs.extend([camera_id,index] for camera_id in cameras)
        xyz=np.asarray(row['candidate_normalized_xyz'],dtype=float)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all():
            raise ValueError('Stereo point has invalid 3D coordinates')
        points[str(index)]=xyz.tolist()
        evidence[str(index)]=dict(fraction=row['fraction'],
                                  offset_normalized=row['candidate_offset_normalized'],
                                  reference_camera=audit['reference_camera'],
                                  supporting_cameras=cameras,
                                  median_ncc=row['patch_scores']['9']['median']['best_ncc'],
                                  leave_one_camera_out=row['leave_one_camera_out'])
    pair_set={tuple(pair) for pair in pairs}
    if len(pair_set)!=len(pairs):
        raise ValueError('Duplicate selected stereo ray')
    common=dict(frame='000470',train_only=True,masks_used=False,
                scope='Matched sparse train-only lower-cord stereo ray gate, not an image mask',
                source_audit=str(audit_path),source_audit_sha256=sha(audit_path),
                source_checkpoint=str(checkpoint),source_checkpoint_sha256=sha(checkpoint),
                fit_sha256=sha(fit_path),transforms_sha256=fit['transforms_sha256'],
                selected_camera_fraction_indices=sorted(pairs),evidence=evidence)
    for name,overrides in [('line',{}),('stereo',points)]:
        path=output_prefix.with_name(output_prefix.name+f'_{name}.json')
        if path.exists():
            raise FileExistsError(path)
        payload=dict(common,stereo_points=overrides)
        path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(selected_rays=len(pairs),fractions=list(fractions),
                          supporting_cameras={key:value['supporting_cameras']
                                              for key,value in evidence.items()})))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True)
    parser.add_argument('--fraction-indices',type=int,nargs='+',default=[8,10])
    args=parser.parse_args()
    build(args.audit,args.fit,args.checkpoint,args.output_prefix,
          tuple(args.fraction_indices))
