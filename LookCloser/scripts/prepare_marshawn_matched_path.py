#!/usr/bin/env python3
"""Same24 prepared-world poses, actual train-anchor fx/fy/cx/cy interpolation.

The older common-wide-focal path remains an extrapolation stress test. Its
vertical-FOV serialization was mathematically consistent; this is a validation
scope correction, not a horizontal/vertical FOV conversion repair.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from marshawn_release_common import validate_review_path,interpolate_intrinsics,load_camera_bundle
from audit_marshawn_body_rays import geometry_hash
from probe_marshawn_occupancy import file_sha


def prepare(data,pose_path,pose_receipt,output):
    meta=json.loads((data/'transforms.json').read_text());train=set(meta['train_filenames'])
    frames=[f for f in meta['frames'] if f['file_path'] in train];by={f['physical_camera']:f for f in frames}
    contract=validate_review_path(pose_path,pose_receipt,frames)
    path=json.loads(pose_path.read_text());old=json.loads(pose_receipt.read_text());rows=[]
    for index,record in enumerate(old['records']):
        values=interpolate_intrinsics(*[by[name] for name in record['anchors']],float(record['fraction']))
        rows.append(dict(index=index,anchors=record['anchors'],fraction=record['fraction'],**values,
            horizontal_fov_degrees=float(np.degrees(2*np.arctan(values['w']/(2*values['fl_x'])))),
            vertical_fov_degrees=float(np.degrees(2*np.arctan(values['h']/(2*values['fl_y']))))))
    poses=np.array([entry['camera_to_world'] for entry in path['camera_path']],np.float64).reshape(24,4,4)[:,:3]
    arrays=dict(camera_to_worlds=poses,fx=np.array([r['fl_x'] for r in rows]),fy=np.array([r['fl_y'] for r in rows]),
        cx=np.array([r['cx'] for r in rows]),cy=np.array([r['cy'] for r in rows]),
        width=np.array([r['w'] for r in rows]),height=np.array([r['h'] for r in rows]))
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'cameras.npz',**arrays)
    receipt=dict(views=24,bundle_sha256=file_sha(output/'cameras.npz'),script_sha256=file_sha(__file__),
        source_transforms_sha256=file_sha(data/'transforms.json'),train_geometry_sha256=geometry_hash(frames),
        source_pose_path_sha256=file_sha(pose_path),source_pose_receipt_sha256=file_sha(pose_receipt),
        poses_identical_to_legacy_path=True,records=rows,heldout_rgb_used=False,train_rgb_read=False,
        interpolation='Each actual pinhole fx/fy/cx/cy independently linear between declared train anchors',
        fov_encoding_used=False,normalization='Already prepared world; no additional pose normalization',
        history='Legacy path used a mathematically correct vertical FOV, deliberately widened to fit loose-box corners. Preserve as extrapolation stress; this bundle restores matched capture intrinsics.')
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (output/'cameras.json').write_text(json.dumps(dict(camera_type='perspective',views=24,
        encoding='Direct fx/fy/cx/cy, not Nerfstudio FOV path format',
        frames=[dict(camera_to_world=poses[i].tolist(),**row) for i,row in enumerate(rows)]),indent=2)+'\n')
    load_camera_bundle(output/'cameras.npz',output/'receipt.json',frames)
    print(json.dumps(dict(output=str(output),views=24,poses_unchanged=True,
        fx_range=[float(arrays['fx'].min()),float(arrays['fx'].max())],fy_range=[float(arrays['fy'].min()),float(arrays['fy'].max())])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','pose-path','pose-receipt','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.data,a.pose_path,a.pose_receipt,a.output)
