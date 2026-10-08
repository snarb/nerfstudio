#!/usr/bin/env python3
"""Build auxiliary rays on visually reviewed floor patches from a measured plane."""
import argparse
import json
from pathlib import Path

import numpy as np
from run_luster_opacity import sha,write


def build(data,plane_path,output):
    if output.exists():raise FileExistsError('Preserve existing floor ray priors')
    meta=json.loads((data/'transforms.json').read_text());plane=json.loads(plane_path.read_text())
    if not plane.get('train_only'):raise ValueError('Floor estimate must use training cameras only')
    coefficients=np.array(plane['plane_z_ax_by_c']);normal=np.array([-coefficients[0],-coefficients[1],1.])
    boxes={16:[920,1620,1100,1740],33:[950,1710,1140,1840],38:[1000,1650,1200,1850]}
    origins=[];directions=[];depths=[];tolerances=[];pixels=[];cameras=[]
    for cid,box in boxes.items():
        f=next(f for f in meta['frames'] if f['camera_id']==cid)
        if f['file_path'] not in meta['train_filenames']:raise ValueError('Only train floor patches permitted')
        x0,y0,x1,y1=box
        if not(0<=x0<x1<=f['w'] and 0<=y0<y1<=f['h']):raise ValueError('Floor patch outside image')
        x,y=np.meshgrid(np.linspace(x0,x1-1,16),np.linspace(y0,y1-1,16))
        xy=np.stack([x.ravel(),y.ravel()],-1)
        local=np.stack([(xy[:,0]+.5-f['cx'])/f['fl_x'],-(xy[:,1]+.5-f['cy'])/f['fl_y'],-np.ones(len(xy))],-1)
        pose=np.asarray(f['transform_matrix']);direction=local@pose[:3,:3].T;direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
        origin=np.repeat(pose[None,:3,3],len(xy),axis=0)
        incidence=direction@normal
        depth=(coefficients[2]-origin@normal)/incidence
        tolerance=.01/np.abs(incidence)  # Conservative1cm normalized height uncertainty.
        target=origin+direction*depth[:,None]
        if not np.isfinite(depth).all() or (depth-tolerance<=.05).any() or (abs(target)>=1.5).any():
            raise ValueError('Unrenderable floor band')
        origins.append(origin);directions.append(direction);depths.append(depth);tolerances.append(tolerance)
        pixels.append(xy);cameras.extend([cid]*len(xy))
    output.mkdir(parents=True);archive=output/'rays.npz'
    np.savez_compressed(archive,origins=np.concatenate(origins).astype('float32'),
        directions=np.concatenate(directions).astype('float32'),depth=np.concatenate(depths).astype('float32'),
        tolerance=np.concatenate(tolerances).astype('float32'),pixels=np.concatenate(pixels),cameras=np.asarray(cameras))
    result=dict(train_only=True,transforms_sha256=sha(data/'transforms.json'),plane_path=str(plane_path),plane_sha256=sha(plane_path),
        plane_z_ax_by_c=coefficients.tolist(),floor_patch_boxes=boxes,rays=len(cameras),height_tolerance=.01,
        archive=archive.name,archive_sha256=sha(archive),builder_sha256=sha(Path(__file__)),
        scope='Auxiliary geometry rays from three reviewed visible-floor patches; all original RGB rays retained')
    write(output/'manifest.json',result);print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--plane',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.data,args.plane,args.output)
