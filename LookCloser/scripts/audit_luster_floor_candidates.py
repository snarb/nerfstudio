#!/usr/bin/env python3
"""Inspect qualified train-depth support for a floor plane before using a prior."""
import argparse
import json
from pathlib import Path

import numpy as np
from run_luster_opacity import sha,write


def audit(data,manifest_path,output):
    if output.exists():raise FileExistsError('Preserve previous geometry audits')
    meta=json.loads((data/'transforms.json').read_text());manifest=json.loads(manifest_path.read_text())
    if not manifest['train_only'] or manifest['transforms_sha256']!=sha(data/'transforms.json'):
        raise ValueError('Require the same calibrated train-only depth')
    frames={Path(f['file_path']).name:f for f in meta['frames'] if f['file_path'] in meta['train_filenames']}
    rng=np.random.default_rng(470);positions=[];cameras=[];pixels=[]
    for name,receipt in manifest['maps'].items():
        f=frames[name];file=manifest_path.parent/receipt['path']
        if sha(file)!=receipt['sha256']:raise ValueError('Depth hash mismatch')
        with np.load(file) as a:
            z=a['depth'];valid=a['valid']&np.isfinite(z)&(z>0)
        y,x=np.nonzero(valid)
        take=rng.choice(len(x),min(5000,len(x)),replace=False)
        y,x=y[take],x[take];depth=z[y,x]*manifest['normalization_scale']
        local=np.stack([(x+.5-f['cx'])/f['fl_x']*depth,-(y+.5-f['cy'])/f['fl_y']*depth,-depth],-1)
        pose=np.asarray(f['transform_matrix']);world=local@pose[:3,:3].T+pose[:3,3]
        positions.append(world);cameras.extend([f['camera_id']]*len(world));pixels.extend(np.stack([x,y],-1))
    points=np.concatenate(positions);cameras=np.asarray(cameras);pixels=np.asarray(pixels)
    # Candidate search only. No assumption that a low horizontal mode is floor.
    lower=points[:,2]<np.quantile(points[:,2],.1)
    bins=np.arange(float(points[:,2].min())-.001,float(np.quantile(points[:,2],.1))+.002,.002)
    histogram,edges=np.histogram(points[lower,2],bins=bins)
    modes=[]
    for i in np.argsort(histogram)[::-1][:10]:
        center=(edges[i]+edges[i+1])*.5
        select=abs(points[:,2]-center)<.003
        p=points[select];cam=cameras[select]
        modes.append(dict(z=float(center),points=len(p),cameras=len(np.unique(cam)),
                          xy_p10_p90=np.quantile(p[:,:2],[.1,.9],axis=0).tolist(),
                          radius_p10_p90=np.quantile(np.linalg.norm(p[:,:2],axis=-1),[.1,.9]).tolist()))
    output.mkdir(parents=True)
    np.savez_compressed(output/'samples.npz',points=points,cameras=cameras,pixels=pixels)
    result=dict(train_only=True,transforms_sha256=manifest['transforms_sha256'],source_sha256=sha(Path(__file__)),
        manifest=str(manifest_path),sampled_points=len(points),z_quantiles=np.quantile(points[:,2],[0,.001,.01,.1,.5,.9,1]).tolist(),
        lower_horizontal_modes=modes,status='Candidates only; image review and independent plane support required')
    write(output/'manifest.json',result);print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.data,args.manifest,args.output)
