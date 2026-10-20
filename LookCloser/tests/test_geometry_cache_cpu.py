import sys
import types
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import guarded_transfer_inference as guard
import luster_hybrid_common as common


def fixture_cache(root,monkeypatch):
    (root/'logs').mkdir()
    checkpoint=root/'frozen.ckpt';checkpoint.write_bytes(b'frozen checkpoint')
    config=root/'config.yml';config.write_text('fixed config')
    digest=common.sha(checkpoint)
    common.write(root/'context.json',dict(checkpoint=str(checkpoint),
        checkpoint_sha256=digest,config=str(config),config_sha256=common.sha(config)))
    row=dict(w=3,h=2,camera_id=7,fl_x=10)
    monkeypatch.setattr(common,'native_rows',lambda:{7:row})
    directory=root/'geometry/cam007';directory.mkdir(parents=True)
    keys=['rgb','alpha','ray_distance_mean','depth_z_q10','depth_z_q50','depth_z_q90']
    for key in keys:
        np.save(directory/f'{key}.npy',np.zeros((2,3,3) if key=='rgb' else (2,3),np.float32))
    receipt=dict(state='complete',camera=7,checkpoint_sha256=digest,row=row,
                 state_hash='a'*64,arrays=keys,
                 hashes={key:common.sha(directory/f'{key}.npy') for key in keys})
    common.write(directory/'receipt.json',receipt)
    return directory,receipt


def test_complete_cache_never_requests_gpu(tmp_path,monkeypatch):
    fixture_cache(tmp_path,monkeypatch)
    monkeypatch.setenv('LUSTER_HYBRID_FRAME_MANIFEST','temporary test binding')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE','1')
    def forbidden(*args):raise AssertionError('GPU admission requested for a complete cache')
    monkeypatch.setattr(guard,'telemetry',forbidden)
    monkeypatch.setattr(guard.fcntl,'flock',forbidden)
    guard.run(types.SimpleNamespace(root=tmp_path,cameras=[7],chunk=1))


def test_corrupted_or_foreign_cache_is_rejected(tmp_path,monkeypatch):
    directory,receipt=fixture_cache(tmp_path,monkeypatch)
    assert guard.cached_geometry_complete(tmp_path,[7])
    common.write(directory/'receipt.json',{**receipt,'checkpoint_sha256':'b'*64})
    with pytest.raises(AssertionError):guard.cached_geometry_complete(tmp_path,[7])
    common.write(directory/'receipt.json',receipt)
    with (directory/'alpha.npy').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(AssertionError):guard.cached_geometry_complete(tmp_path,[7])


def test_changed_calibration_or_config_is_rejected(tmp_path,monkeypatch):
    fixture_cache(tmp_path,monkeypatch)
    monkeypatch.setattr(common,'native_rows',lambda:{7:dict(w=3,h=2,camera_id=7,fl_x=11)})
    with pytest.raises(AssertionError):guard.cached_geometry_complete(tmp_path,[7])
    (tmp_path/'config.yml').write_text('modified')
    with pytest.raises(AssertionError):guard.cached_geometry_complete(tmp_path,[7])


def test_partial_cache_requires_inference(tmp_path):
    assert not guard.cached_geometry_complete(tmp_path,[7])
