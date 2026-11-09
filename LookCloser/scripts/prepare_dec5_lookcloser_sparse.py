#!/usr/bin/env python3
"""Export reliable train-only fixed-pose SfM observations for LookCloser."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from nerfstudio.data.utils.colmap_parsing_utils import read_model,qvec2rotmat
from dec5_hair import cv_pose,project,VALIDATION,TEST
from run_dec5_hair_campaign import atomic_json,sha


def prepare(data,model,output):
    payload=json.loads((data/'transforms.json').read_text())
    frames=[f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
    if len(frames)!=59 or any(f['physical_camera'] in VALIDATION+TEST for f in frames):raise ValueError('Requires train59')
    by_name={f['file_path']:i for i,f in enumerate(frames)}
    cameras,images,points=read_model(model,ext='.bin')
    if {im.name for im in images.values()}!=set(by_name):raise ValueError('SfM split mismatch')
    image_indices={};max_pose_error=0.
    for iid,image in images.items():
        index=by_name[image.name];f=frames[index];pose=cv_pose(f);rotation=qvec2rotmat(image.qvec)
        error=max(np.abs(rotation-pose[:3,:3].T).max(),np.abs(image.tvec+rotation@pose[:3,3]).max())
        max_pose_error=max(max_pose_error,float(error))
        np.testing.assert_allclose(cameras[image.camera_id].params,[f[k] for k in ['fl_x','fl_y','cx','cy']],atol=1e-8,rtol=0)
        image_indices[iid]=index
    if max_pose_error>1e-7:raise ValueError('Sparse initializer was not triangulated under the original fixed poses')
    xyz=[];op=[];oc=[];uv=[];errors=[]
    for point in points.values():
        if len(point.image_ids)<3:continue
        observations=[];residual=[]
        for iid,oid in zip(point.image_ids,point.point2D_idxs):
            image=images[iid];index=image_indices[iid];measured=image.xys[oid]
            predicted,z=project(point.xyz[None],frames[index])
            if z[0]<=0:continue
            residual.append(float(np.linalg.norm(predicted[0]-measured)))
            observations.append((index,measured))
        if len(observations)<3 or np.median(residual)>.75 or max(residual)>2:continue
        pidx=len(xyz);xyz.append(point.xyz);errors.extend(residual)
        for index,measured in observations:op.append(pidx);oc.append(index);uv.append(measured)
    if len(xyz)<100:raise ValueError('Too few reliable training points')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,xyz=np.asarray(xyz,np.float32),observation_point=np.asarray(op,np.int32),
        observation_camera=np.asarray(oc,np.int32),uv=np.asarray(uv,np.float32),
        camera_names=np.asarray([f['physical_camera'] for f in frames]))
    atomic_json(output.with_suffix('.json'),dict(points=len(xyz),observations=len(op),
        max_fixed_pose_difference=max_pose_error,original_reprojection_median_px=float(np.median(errors)),
        original_reprojection_p95_px=float(np.quantile(errors,.95)),
        gates=dict(minimum_views=3,maximum_median_reprojection_px=.75,maximum_reprojection_px=2),
        dataset_sha256=sha(data/'transforms.json'),npz_sha256=sha(output),uses_validation_rgb=False,uses_test_rgb=False,
        inherited_sparse_points=False,source_model=str(model),implementation_sha256=sha(__file__)))
    print('LookCloser sparse initializer points=%d observations=%d'%(len(xyz),len(op)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','model','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.data,a.model,a.output)


if __name__=='__main__':main()
