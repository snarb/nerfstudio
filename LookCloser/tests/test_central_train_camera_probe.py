from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_central_train_camera_probe import validate_camera


def fixture():
    native=dict(physical_camera='H004_C005_example',transform_matrix=np.eye(4).tolist(),
                fl_x=1000,fl_y=1000,cx=960,cy=540,w=1920,h=1080)
    moving=deepcopy(native);moving.update(fl_x=700,fl_y=700)
    target=deepcopy(native);target.update(physical_camera='virtual_probe',train_pose_physical_camera=native['physical_camera'])
    return native,moving,target


def test_native_and_movie_intrinsics_share_exact_pose():
    native,moving,target=fixture();validate_camera(target,native,moving,'native')
    target.update(fl_x=700,fl_y=700);validate_camera(target,native,moving,'flight_intrinsics')


def test_reject_native_mask_id_in_foreign_target():
    native,moving,target=fixture();target['physical_camera']=native['physical_camera']
    with pytest.raises(ValueError,match='mask'):validate_camera(target,native,moving,'native')


def test_reject_wrong_pose_intrinsics_and_rgb_path():
    native,moving,target=fixture()
    with pytest.raises(ValueError,match='intrinsic'):validate_camera(target,native,moving,'flight_intrinsics')
    wrong=deepcopy(target);wrong['transform_matrix'][0][3]=.01
    with pytest.raises(AssertionError):validate_camera(wrong,native,moving,'native')
    target['file_path']='train.exr'
    with pytest.raises(ValueError,match='RGB'):validate_camera(target,native,moving,'native')


def test_transfer_uses_actual_actor_time_and_disjoint_root(monkeypatch):
    import probe_central_train_pose_transfer as transfer
    monkeypatch.setattr(transfer.probe,'ROOT',Path('/unused'))
    monkeypatch.setattr(transfer.probe,'FRAME','001037')
    for frame in transfer.FRAMES:
        transfer.configure(frame)
        assert transfer.probe.FRAME==frame and transfer.probe.ROOT==transfer.ROOT/frame
    with pytest.raises(ValueError,match='Unplanned'):transfer.configure('001037')


def test_renderer_installation_is_isolated_per_actor_time():
    from run_central_train_probe_isolated import child_commands
    commands=child_commands(1,3)
    assert len(commands)==2
    assert [c[c.index('--frame')+1] for c in commands]==['001123','001193']
    assert all(c[-4:]==['--worker','1','--workers','3'] for c in commands)
    with pytest.raises(ValueError):child_commands(3,3)
