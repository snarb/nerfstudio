#!/usr/bin/env python3
"""Verify calibrated unmasked RGB, train-only maps, and mixed-resolution coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(data):
    from nerfstudio.data.dataparsers.luster_opacity_dataparser import LusterOpacityDataParserConfig
    meta=json.loads((data/'transforms.json').read_text())
    receipt=json.loads((data/'audit.json').read_text())
    if receipt['transforms_sha256']!=sha(data/'transforms.json'):
        raise ValueError('Prepared calibration/split changed')
    if any('mask_path' in r or 'depth_file_path' in r for r in meta['frames']):
        raise ValueError('Masks or implicit depth loader detected')
    rows={Path(r['file_path']).name:r for r in meta['frames']}
    for row in receipt['receipts']:
        name=f"cam_{row['camera_id']:03d}_000470.png"
        if sha(data/'images'/name)!=row['hd_sha256']:
            raise ValueError(f'RGB changed: {name}')
    expected={Path(n).stem for n in meta['train_filenames']}
    maps={p.stem:p for p in (data/'lookcloser_frequencies').glob('*.pt')}
    if set(maps)!=expected:
        raise ValueError('Missing maps or eval-map leakage')
    shapes={}
    for stem,path in maps.items():
        frequency=torch.load(path,map_location='cpu',weights_only=True)
        sidecar=json.loads(path.with_suffix('.json').read_text())
        row=rows[stem+'.png']
        if sidecar['image_shape']!=[row['h'],row['w']]:
            raise ValueError(f'Per-camera map shape metadata mismatch: {stem}')
        if sidecar['source_sha256']!=sha(data/row['file_path']):
            raise ValueError(f'Frequency fit used different RGB: {stem}')
        if sidecar['fit_steps_per_level']!=1000 or sidecar['max_res']!=8192:
            raise ValueError('Frequency fit recipe mismatch')
        if frequency.ndim!=2 or not torch.isfinite(frequency).all() or not (frequency>0).all():
            raise ValueError(f'Invalid map: {stem}')
        shapes[stem]=list(frequency.shape)
    parser=LusterOpacityDataParserConfig(data=data).setup()
    train=parser.get_dataparser_outputs('train');test=parser.get_dataparser_outputs('val')
    if (len(train.image_filenames),len(test.image_filenames))!=(162,3):
        raise ValueError('Parsed split differs from 162/3')
    if {int(p.stem.split('_')[1]) for p in test.image_filenames}!={12,95,150}:
        raise ValueError('Wrong held-out cameras')
    if not np.allclose(train.scene_box.aabb.numpy(),meta['opacity_aabb'],atol=1e-7):
        raise ValueError('Parser did not preserve explicit bounds')
    result=dict(complete=True,masks_used=False,train=162,eval=3,frequency_maps=162,
                transforms_sha256=sha(data/'transforms.json'),map_shapes=shapes)
    (data/'ready_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='map_shapes'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    audit(args.data)
