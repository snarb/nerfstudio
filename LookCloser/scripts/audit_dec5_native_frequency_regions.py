#!/usr/bin/env python3
"""CPU-only inspection of native frequency levels by train semantic region."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    data=a.frame/'data';meta=json.loads((data/'transforms.json').read_text())
    train=set(meta['train_filenames']);rows=[]
    totals={k:np.zeros(16,np.int64) for k in ['hair','face','trusted_background']}
    for f in meta['frames']:
        if f['file_path'] not in train:continue
        stem=Path(f['file_path']).stem;path=data/'lookcloser_frequencies'/(stem+'.pt')
        info=json.loads(path.with_suffix('.json').read_text())
        values=torch.load(path,map_location='cpu',weights_only=True).numpy().squeeze()
        if values.ndim!=2 or not np.isfinite(values).all() or np.any(values<=0):
            raise ValueError('Invalid frequency map: '+str(path))
        if info['image_shape']!=[3072,6144] or info['crop_coords_xywh'] is not None:
            raise ValueError('Expected uncropped native6K frequency fit')
        levels=np.clip(np.rint(np.log(values/info['min_res'])/np.log(info['per_level_scale'])),0,15).astype(np.uint8)
        prob=np.load(a.frame/'semantics'/(f['physical_camera']+'.npz'))['probability'].astype(np.float32)/65535
        prob=cv2.resize(prob,(levels.shape[1],levels.shape[0]),interpolation=cv2.INTER_AREA)
        masks=dict(hair=prob[...,0]>.9,face=prob[...,1]>.9,trusted_background=prob[...,2]>.98)
        hists={name:np.bincount(levels[mask],minlength=16) for name,mask in masks.items()}
        for name,hist in hists.items():totals[name]+=hist
        rows.append(dict(camera=f['physical_camera'],shape=list(values.shape),
                         histograms={k:v.tolist() for k,v in hists.items()}))
        if f['physical_camera']=='E004_A005_1210QB':
            rgb=cv2.imread(str(data/f['file_path']));rgb=cv2.resize(rgb,(768,384),interpolation=cv2.INTER_AREA)
            heat=cv2.applyColorMap(cv2.resize(levels*17,(768,384),interpolation=cv2.INTER_NEAREST),cv2.COLORMAP_TURBO)
            # Inputs are sideways; rotate both identically for portrait inspection.
            cv2.imwrite(str(a.output/'native_rgb_frequency.jpg'),np.concatenate([np.rot90(rgb),np.rot90(heat)],axis=1))
    if len(rows)!=64:raise ValueError('Expected exactly64 train frequency maps')
    summary={}
    for name,hist in totals.items():
        n=int(hist.sum());summary[name]=dict(cells=n,histogram=hist.tolist(),
            mean_level=float(np.dot(hist,np.arange(16))/n) if n else None,
            fraction_top4=float(hist[12:].sum()/n) if n else None)
    (a.output/'metrics.json').write_text(json.dumps(dict(train_maps=len(rows),
        native_resolution=[6144,3072],regions=summary,rows=rows,
        semantics_used_for_diagnostic_only=True,heldout_rgb_loaded=False),indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
