from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compose_cinematic_train_ending import dissolve_alpha,blend_display,intrinsics_sample
import compose_cinematic_train_ending as compositor
from copy import deepcopy


def test_eight_dissolve_frames_and_exact_last_second():
    alpha=np.array([dissolve_alpha(i) for i in range(150)])
    assert np.count_nonzero(alpha==0)==118
    assert np.count_nonzero((alpha>0)&(alpha<1))==8
    assert np.count_nonzero(alpha==1)==24
    assert (np.diff(alpha)>=0).all()
    a=np.zeros((2,3,3),np.uint8);b=np.full_like(a,211)
    np.testing.assert_array_equal(blend_display(a,b,0),a)
    np.testing.assert_array_equal(blend_display(a,b,1),b)
    np.testing.assert_array_equal(blend_display(a,b,.5),np.full_like(a,106))


def test_same_intrinsics_is_exact_identity_including_borders():
    image=np.arange(4*6*3,dtype=np.float32).reshape(4,6,3)
    row=dict(w=6,h=4,fl_x=5.,fl_y=7.,cx=3.,cy=2.)
    np.testing.assert_array_equal(intrinsics_sample(image,row,row),image)


def test_optical_twofold_zoom_uses_half_pixel_centers_not_integer_rays():
    x=np.broadcast_to(np.arange(6,dtype=np.float32)[None,:,None],(4,6,3)).copy()
    row=dict(w=6,h=4,fl_x=5.,fl_y=7.,cx=3.,cy=2.)
    target=dict(row,fl_x=10.,fl_y=14.)
    result=intrinsics_sample(x,row,target)
    expected=np.broadcast_to(np.array([1.25,1.75,2.25,2.75,3.25,3.75])[None,:,None],(4,6,3))
    np.testing.assert_allclose(result,expected,atol=1e-7)
    with pytest.raises(ValueError,match='outside'):
        intrinsics_sample(x,row,dict(target,cx=100.))


def test_configuration_rejects_noncoincident_pose_or_changing_lens(monkeypatch):
    camera=dict(physical_camera='H004_C005_1210SZ',transform_matrix=np.eye(4).tolist(),
        w=1920,h=1080,fl_x=500.,fl_y=500.,cx=960.,cy=540.)
    meta=dict(dataparser_scale=1.,dataparser_transform=np.eye(4)[:3].tolist())
    frames=[f'{899+2*i:06d}' for i in range(150)]
    q=dict(ordered_frame_ids=frames,profiles_sha256='digest',exposure_sha256='digest',
        calibration_sha256='digest',camera_path_report=dict(endpoint_train_camera=camera['physical_camera']),
        inventory=[dict(index=i,frame_id=f,metadata='meta',metadata_sha256='digest',camera=deepcopy(camera))
                   for i,f in enumerate(frames)])
    monkeypatch.setattr(compositor,'sha',lambda _: 'digest')
    def read(path):
        if str(path)=='meta':return meta
        if path==compositor.CALIBRATION:return dict(frames=[camera])
        return q
    monkeypatch.setattr(compositor,'read',read)
    compositor.configuration(Path('virtual'))
    q['inventory'][118]['camera']['transform_matrix'][0][3]=.01
    with pytest.raises(AssertionError):compositor.configuration(Path('virtual'))
    q['inventory'][118]['camera']=deepcopy(camera)
    q['inventory'][128]['camera']['fl_x']=501.
    with pytest.raises(AssertionError):compositor.configuration(Path('virtual'))
