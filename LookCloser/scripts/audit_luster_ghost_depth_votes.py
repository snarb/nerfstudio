#!/usr/bin/env python3
"""Test ghost-layer points against independent qualified training depth rays."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from export_luster_opacity import load
from run_luster_opacity import sha, write


def audit(selection,path_file,layers,manifest_path,output):
    if output.exists():raise FileExistsError('Preserve prior audits')
    selected=json.loads(selection.read_text());selected=selected.get('selected',selected)
    trainer,request=load(selected);dataset=trainer.pipeline.datamanager.train_dataset
    manifest=json.loads(manifest_path.read_text())
    if not manifest['train_only'] or manifest['transforms_sha256']!=sha(Path(request['data'])/'transforms.json'):
        raise ValueError('Require matching train-only reference')
    path=json.loads(path_file.read_text());intr=path['intrinsics']
    camera=Cameras(camera_to_worlds=torch.tensor(path['poses'][23][:3],dtype=torch.float32)[None],
        fx=intr['fx'],fy=intr['fy'],cx=intr['cx'],cy=intr['cy'],width=intr['w'],height=intr['h'],camera_type=CameraType.PERSPECTIVE)
    source=json.loads(layers.read_text())['rays']
    xy=np.array([x['xy'] for x in source])
    rays=camera.generate_rays(0,coords=torch.tensor(xy[:,::-1].copy())+.5)
    records=[]
    for i,x in enumerate(source):
        significant=[c for c in x['clusters'] if c['mass']>.1]
        first=significant[0]
        samples=[s for s in x['samples'] if first['t_min']<=s['t']<=first['t_max']]
        center=sum(s['weight']*s['t'] for s in samples)/sum(s['weight'] for s in samples)
        point=(rays.origins[i]+rays.directions[i]*center).numpy()
        records.append(dict(xy=x['xy'],point=point.tolist(),first_layer_rgb=first['conditional_rgb'],votes=[]))
    points=np.array([x['point'] for x in records])
    for i,image in enumerate(dataset.image_filenames):
        receipt=manifest['maps'][image.name];file=manifest_path.parent/receipt['path']
        if sha(file)!=receipt['sha256']:raise ValueError('Depth map receipt mismatch')
        with np.load(file) as z:
            depth,unc,valid=[z[k] for k in ['depth','uncertainty','valid']]
        cam=dataset.cameras[i:i+1].to('cpu');pose=cam.camera_to_worlds[0].numpy()
        local=(points-pose[:3,3])@pose[:3,:3]
        predicted=-local[:,2]
        u=float(cam.fx)*local[:,0]/predicted+float(cam.cx)-.5
        v=-float(cam.fy)*local[:,1]/predicted+float(cam.cy)-.5
        h,w=depth.shape
        for j,(xx,yy,zz) in enumerate(zip(u,v,predicted)):
            if not np.isfinite([xx,yy,zz]).all() or zz<=0:continue
            px,py=int(round(xx)),int(round(yy))
            if not(1<=px<w-1 and 1<=py<h-1):continue
            region=np.s_[py-1:py+2,px-1:px+2]
            if not valid[region].all():continue
            reference=depth[region]*manifest['normalization_scale']
            tolerance=np.maximum(2*unc[region]*manifest['normalization_scale'],.001*reference)
            # Extra1cm normalized margin makes empty-space votes conservative.
            kind=('empty' if zz<float((reference-tolerance).min())-.01 else
                  'surface' if abs(zz-float(reference[1,1]))<=float(tolerance[1,1])+.01 else 'behind')
            records[j]['votes'].append(dict(camera=image.name,xy=[px,py],point_camera_z=float(zz),
                                          measured_z=float(reference[1,1]),kind=kind))
    for x in records:
        x['counts']={kind:sum(v['kind']==kind for v in x['votes']) for kind in ['empty','surface','behind']}
    write(output,dict(scope='Qualified3x3 train-depth patches,1cm safety margin; empty-space audit not an opacity mask',
        manifest=str(manifest_path),source_sha256=sha(Path(__file__)),points=records))
    print(json.dumps([dict(xy=x['xy'],point=x['point'],counts=x['counts']) for x in records]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--path',type=Path,required=True)
    parser.add_argument('--layers',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.selection,args.path,args.layers,args.manifest,args.output)
