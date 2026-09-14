from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import probe_temporal_camera_phase as module
from render_patchmatch_camera_path import normalize_frame


def test_phase_wrap_transfers_calibration_pose_not_normalized_translation(monkeypatch):
    metas={}
    inventory=[]
    raw=[]
    for i in range(4):
        transform=np.eye(4);transform[:3,3]=[.1*i,-.2*i,.3*i]
        metadata=dict(dataparser_scale=.2+.1*i,dataparser_transform=transform[:3].tolist())
        metas[str(i)]=metadata
        pose=np.eye(4);pose[:3,3]=[i+1,2*i+1,3.]
        camera=dict(transform_matrix=pose.tolist(),physical_camera=str(i),fl_x=1000.,w=1920,h=1080)
        raw.append(camera)
        inventory.append(dict(metadata=str(i),camera=normalize_frame(camera,{},metadata),frame_id=str(i)))
    monkeypatch.setattr(module,'read',lambda p:metas[str(p)])
    original=deepcopy(inventory)
    for index in range(4):
        for phase in [-3,0,3]:
            result=module.shifted_camera(inventory,index,phase,{})
            expected=normalize_frame(raw[(index+phase)%4],{},metas[str(index)])
            np.testing.assert_allclose(result['transform_matrix'],expected['transform_matrix'],atol=1e-12)
            assert result['physical_camera']==str((index+phase)%4)
    assert inventory==original
