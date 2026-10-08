#!/usr/bin/env python3
"""Pin sparse train-only cord prior rays whose current depth-band RGB is credible."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def select(audit_path: Path, output: Path, max_rgb_error: float, min_band_mass: float,
           min_views_per_fraction: int) -> None:
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(max_rgb_error) or not 0 < max_rgb_error < 1:
        raise ValueError('Invalid RGB error threshold')
    if not np.isfinite(min_band_mass) or not 0 < min_band_mass < 1:
        raise ValueError('Invalid band mass threshold')
    if min_views_per_fraction < 2:
        raise ValueError('Need multiview support for each selected fraction')
    audit=json.loads(audit_path.read_text())
    if (audit.get('masks_used') is not False
            or audit.get('scope')!='Read-only train-camera line colour diagnostic, not a mask'):
        raise ValueError('Expected the unmasked train-only ray-colour audit')
    request=json.loads(Path(audit['source_request']).read_text())
    if (request['resume']!=audit['checkpoint']
            or sha(Path(audit['checkpoint']))!=audit['checkpoint_sha256']
            or sha(Path(request['model']['cord_sampling_fit_path']))!=audit['fit_sha256']
            or sha(Path(request['data'])/'transforms.json')!=audit['transforms_sha256']):
        raise ValueError('Audit source checkpoint, geometry or calibration changed')
    transforms=json.loads((Path(request['data'])/'transforms.json').read_text())
    train={int(Path(name).stem.split('_')[1]) for name in transforms['train_filenames']}
    heldout={int(Path(name).stem.split('_')[1]) for name in transforms['val_filenames']}
    fit=json.loads(Path(request['model']['cord_sampling_fit_path']).read_text())
    allowed=set.intersection(*(set(fit['fit_camera_ids'][key])
                               for key in ('junction','first_bead')))
    if fit['train_only'] is not True or not allowed.issubset(train) or allowed & heldout:
        raise ValueError('Cord geometry must use training cameras only')
    groups={}
    reviewed=[]
    for row in audit['rows']:
        camera_id=int(row['camera_id'])
        if camera_id not in allowed:
            raise ValueError('Audit includes a camera without train-only annotations')
        fraction_index=round((float(row['fraction'])-.08)/(.84/11))
        if not 0<=fraction_index<12 or abs(float(row['fraction'])-(.08+fraction_index*.84/11))>1e-5:
            raise ValueError('Audit fraction does not match the model rays')
        error=float(np.linalg.norm(np.asarray(row['band_rgb'])-np.asarray(row['source_gt'])))
        mass=float(row['band_mass'])
        accepted=bool(np.isfinite(error) and np.isfinite(mass)
                      and error<=max_rgb_error and mass>=min_band_mass)
        reviewed.append(dict(camera_id=camera_id,fraction_index=fraction_index,
                             pixel=[int(row['x']),int(row['y'])],
                             band_rgb_l2_error=error,band_mass=mass,
                             individually_qualified=accepted))
        if accepted:
            groups.setdefault(fraction_index,set()).add(camera_id)
    pairs={(row['camera_id'],row['fraction_index']) for row in reviewed}
    if len(reviewed)!=len(allowed)*12 or len(pairs)!=len(reviewed):
        raise ValueError('Expected exactly twelve unique rays per annotated camera')
    selected=sorted([camera_id,fraction_index] for camera_id,fraction_index in pairs
                    if camera_id in groups.get(fraction_index,set())
                    and len(groups[fraction_index])>=min_views_per_fraction)
    if len(selected)<min_views_per_fraction:
        raise ValueError('No sufficiently supported train-only rays')
    manifest=dict(frame='000470',train_only=True,masks_used=False,
                  scope='Sparse train-only RGB/depth-band confidence filter, not an image mask',
                  source_audit=str(audit_path),source_audit_sha256=sha(audit_path),
                  source_checkpoint=audit['checkpoint'],
                  source_checkpoint_sha256=audit['checkpoint_sha256'],
                  fit_sha256=audit['fit_sha256'],transforms_sha256=audit['transforms_sha256'],
                  thresholds=dict(max_rgb_l2_error=max_rgb_error,min_band_mass=min_band_mass,
                                  min_views_per_fraction=min_views_per_fraction),
                  selected_camera_fraction_indices=selected,reviewed=reviewed)
    output.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(selected_rays=len(selected),selected_fractions=len(groups),
                          source_checkpoint=audit['checkpoint'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-rgb-error',type=float,default=.12)
    parser.add_argument('--min-band-mass',type=float,default=.3)
    parser.add_argument('--min-views-per-fraction',type=int,default=2)
    args=parser.parse_args()
    select(args.audit,args.output,args.max_rgb_error,args.min_band_mass,
           args.min_views_per_fraction)
