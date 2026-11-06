"""Validate RGB resolution against unchanged native-world geometry provenance."""
import hashlib,json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_native_resolution_binding(rgb_transforms,geometry_transforms,geometry_sha256,train_scale):
    if train_scale not in (1,2):raise ValueError('Only HD identity or2xTRAIN resolution is qualified')
    rgb_path,geometry_path=Path(rgb_transforms),Path(geometry_transforms)
    if sha(geometry_path)!=geometry_sha256:raise ValueError('Original HD geometry transforms changed')
    rgb=json.loads(rgb_path.read_text());geometry=json.loads(geometry_path.read_text())
    def camera_rows(value):
        rows={row['physical_camera']:row for row in value['frames']}
        if len(rows)!=66 or len(rows)!=len(value['frames']):raise ValueError('Exactly66 unique physical cameras required')
        return rows
    current,original=camera_rows(rgb),camera_rows(geometry)
    if set(current)!=set(original):raise ValueError('Physical camera membership changed')
    def split(value,key):
        paths=set(value[key]);rows=[row['physical_camera'] for row in value['frames'] if row['file_path'] in paths]
        if len(paths)!=len(value[key]) or len(rows)!=len(paths):raise ValueError('Duplicate or missing split filenames')
        return rows
    train=split(geometry,'train_filenames');evaluation=split(geometry,'val_filenames')
    if len(train)!=63 or set(evaluation)!={'DC','HC','KA'} or set(train)&set(evaluation):raise ValueError('Original split is not63TRAIN+3EVAL')
    for key in ['train_filenames','val_filenames','test_filenames']:
        if split(rgb,key)!=split(geometry,key):raise ValueError('Physical camera split/order changed: '+key)
    for camera,row in current.items():
        prior=original[camera];scale=train_scale if camera in train else 1
        pose=np.asarray(row['transform_matrix']);oldpose=np.asarray(prior['transform_matrix'])
        if pose.shape!=(4,4) or not np.array_equal(pose,oldpose):raise ValueError('Original native-world pose changed: '+camera)
        for key in ['w','h','fl_x','fl_y','cx','cy']:
            if not np.isfinite(row[key]) or row[key]!=prior[key]*scale:raise ValueError('Resolution/intrinsics mismatch: '+camera+' '+key)
        for key in ['k1','k2','k3','k4','p1','p2']:
            if row.get(key,0)!=0 or prior.get(key,0)!=0:raise ValueError('Expected already-undistorted source: '+camera)
    return dict(rgb_transforms=str(rgb_path),rgb_transforms_sha256=sha(rgb_path),geometry_transforms=str(geometry_path),geometry_transforms_sha256=geometry_sha256,train_scale=train_scale,eval_scale=1,physical_camera_order=train,all_native_poses_exact=True,geometry_rays_not_regenerated=True)
