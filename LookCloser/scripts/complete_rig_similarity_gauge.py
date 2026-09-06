"""Propagate a recorded train-rig similarity gauge to untouched query cameras.

The transform is taken from the train-only BA receipt, never estimated from query
RGB. This restores a common coordinate convention, not a calibrated query pose
correction or evidence that the refined rig is physically accurate.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np


def transform_camera_similarity(frame, scale, rotation, translation):
    rotation=np.asarray(rotation,dtype=float);translation=np.asarray(translation,dtype=float)
    pose=np.asarray(frame['transform_matrix'],dtype=float)
    if (not np.isfinite(scale) or scale<=0 or rotation.shape!=(3,3) or translation.shape!=(3,)
            or pose.shape!=(4,4) or not np.isfinite(pose).all()
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-8)
            or not np.isclose(np.linalg.det(rotation),1.,atol=1e-8)):
        raise ValueError('Invalid positive similarity or camera pose')
    if not np.allclose(pose[3],[0,0,0,1]):
        raise ValueError('Camera must be homogeneous c2w')
    result=deepcopy(frame);transformed=pose.copy()
    transformed[:3,:3]=rotation@pose[:3,:3]
    transformed[:3,3]=scale*(rotation@pose[:3,3])+translation
    result['transform_matrix']=transformed.tolist()
    return result


def complete_query_gauge(original, refined, manifest):
    train={row['physical_camera'] for row in manifest['camera_changes']}
    before={f['physical_camera']:f for f in original['frames']}
    after={f['physical_camera']:f for f in refined['frames']}
    if (len(before)!=len(original['frames']) or len(after)!=len(refined['frames'])
            or set(before)!=set(after) or not train<=set(before)):
        raise ValueError('Camera identities must match uniquely')
    query=set(before)-train
    if any(before[name]!=after[name] for name in query):
        raise ValueError('Only previously untouched query cameras may receive this gauge')
    if not query or not train or manifest['uses_eval_rgb'] or manifest['per_time_camera_optimization']:
        raise ValueError('Need a shared train-only fit with untouched query cameras')
    gauge=manifest['gauge_alignment'];result=deepcopy(refined)
    for index,frame in enumerate(result['frames']):
        if frame['physical_camera'] in query:
            result['frames'][index]=transform_camera_similarity(frame,**gauge)
    return result
