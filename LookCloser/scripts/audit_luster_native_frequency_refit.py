#!/usr/bin/env python3
"""Compare transferred and genuinely fitted 4K maps along a train-fitted cord."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: np.ndarray, metadata: dict) -> dict:
    ratio = (float(metadata['max_res'])/float(metadata['min_res']))**(1/(int(metadata['n_levels'])-1))
    levels = np.clip(np.rint(np.log(values/metadata['min_res'])/np.log(ratio)),
                     0,metadata['n_levels']-1)
    return dict(median_scalar_resolution=float(np.median(values)),
                mean_scalar_resolution=float(np.mean(values)),
                p10_scalar_resolution=float(np.quantile(values,.1)),
                p90_scalar_resolution=float(np.quantile(values,.9)),
                mean_frequency_level=float(np.mean(levels)),
                fraction_level_12_or_higher=float(np.mean(levels>=12)))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-data',type=Path,required=True)
    parser.add_argument('--refit-data',type=Path,required=True)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--camera-ids',type=int,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    transforms_path=args.baseline_data/'transforms.json'
    if transforms_path.read_bytes()!=(args.refit_data/'transforms.json').read_bytes():
        raise ValueError('Image calibration or train/eval split changed')
    transforms=json.loads(transforms_path.read_text())
    fit=json.loads(args.fit.read_text())
    if fit['transforms_sha256']!=sha(transforms_path) or fit.get('train_only') is not True:
        raise ValueError('Cord fit is not pinned to this train-only calibration')
    frames={int(frame['camera_id']):frame for frame in transforms['frames']}
    train={Path(name).stem for name in transforms['train_filenames']}
    refit={f'cam_{camera_id:03d}_000470' for camera_id in args.camera_ids}
    if len(refit)!=len(args.camera_ids) or not refit.issubset(train):
        raise ValueError('Refit camera IDs must be unique training images')
    baseline_maps=args.baseline_data/'lookcloser_frequencies'
    refit_maps=args.refit_data/'lookcloser_frequencies'
    for directory in (baseline_maps,refit_maps):
        if {path.stem for path in directory.glob('*.pt')}!=train:
            raise ValueError(f'Frequency-map inventory incomplete: {directory}')
    unchanged=0
    for stem in train-refit:
        for suffix in ('.pt','.json'):
            if sha(baseline_maps/f'{stem}{suffix}')!=sha(refit_maps/f'{stem}{suffix}'):
                raise ValueError(f'Non-refitted camera map changed: {stem}')
        unchanged+=1
    start=np.array(fit['fits']['junction']['normalized_world_xyz'])
    end=np.array(fit['fits']['first_bead']['normalized_world_xyz'])
    rows=[]
    for camera_id in args.camera_ids:
        frame=frames[camera_id]
        stem=Path(frame['file_path']).stem
        if stem not in train:
            raise ValueError(f'Camera {camera_id} is not a training image')
        maps=[]
        for data in (args.baseline_data,args.refit_data):
            path=data/'lookcloser_frequencies'/f'{stem}.pt'
            metadata=json.loads(path.with_suffix('.json').read_text())
            values=torch.load(path,map_location='cpu',weights_only=True).float().numpy()
            if (metadata['image_shape'] != [frame['h'],frame['w']]
                    or metadata['source_sha256']!=sha(data/frame['file_path'])
                    or values.shape != ((frame['h']-metadata['patch_size'])//metadata['stride']+1,
                                        (frame['w']-metadata['patch_size'])//metadata['stride']+1)
                    or not np.isfinite(values).all() or values.min()<=0):
                raise ValueError(f'Frequency map geometry/source mismatch for {camera_id}')
            maps.append((path,metadata,values))
        if 'transfer' not in maps[0][1].get('derivation','') or 'derivation' in maps[1][1]:
            raise ValueError('Expected transferred baseline and native fitted candidate')
        if maps[1][1].get('fit_steps_per_level')!=1000:
            raise ValueError('Native frequency fit did not follow the pinned 1000-step recipe')
        if maps[0][2].shape!=maps[1][2].shape:
            raise ValueError('Map dimensions changed')
        pose=np.array(frame['transform_matrix'])
        points=start[None]+np.linspace(.08,.92,100)[:,None]*(end-start)[None]
        local=(points-pose[:3,3])@pose[:3,:3]
        forward=-local[:,2]
        x=frame['fl_x']*local[:,0]/forward+frame['cx']
        y=-frame['fl_y']*local[:,1]/forward+frame['cy']
        if not np.all((forward>0)&(x>=0)&(x<frame['w'])&(y>=0)&(y<frame['h'])):
            raise ValueError(f'Fitted cord line leaves camera {camera_id}')
        result=dict(camera_id=camera_id,projected_cord_line_samples=100)
        for label,(path,metadata,values) in zip(('transferred','native_refit'),maps):
            stride=metadata['stride']
            patch_center=(metadata['patch_size']-1)/2
            px=np.clip(np.rint((x-patch_center)/stride).astype(int),0,values.shape[1]-1)
            py=np.clip(np.rint((y-patch_center)/stride).astype(int),0,values.shape[0]-1)
            result[label]=dict(map_sha256=sha(path),metadata_sha256=sha(path.with_suffix('.json')),
                               full_map=summarize(values,metadata),
                               projected_cord=summarize(values[py,px],metadata))
        rows.append(result)
    receipt=dict(scope='Train-only projected-line frequency comparison, not a cord mask',
                 masks_used=False,fit_sha256=sha(args.fit),
                 transforms_sha256=sha(transforms_path),
                 baseline_data=str(args.baseline_data),refit_data=str(args.refit_data),
                 unchanged_transferred_maps=unchanged,rows=rows)
    args.output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:row[k] for k in ('camera_id','projected_cord_line_samples')}
                      for row in rows]))


if __name__=='__main__':
    main()
