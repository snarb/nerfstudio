from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_patchmatch_camera_path import calibration_path,normalize_frame


def camera(name,x):
    pose=np.eye(4);pose[0,3]=x
    return {'physical_camera':name,'transform_matrix':pose.tolist(),
            'fl_x':100.,'fl_y':110.,'cx':40.,'cy':30.,'w':80,'h':60,
            'file_path':f'{name}.png'}


def test_mesh_receipt_undoes_preapplied_coordinate_transform_once():
    applied=np.eye(4);applied[:3,:3]=[[0,1,0],[0,0,1],[1,0,0]];applied[:3,3]=[1,2,3]
    current=np.eye(4);current[:3,3]=[4,5,6]
    receipt=current@applied
    f=camera('train',7);source=deepcopy(f)
    actual=normalize_frame(f,{'applied_transform':applied[:3].tolist()},
                           {'dataparser_transform':receipt[:3].tolist(),'dataparser_scale':.1})
    expected=current@np.asarray(f['transform_matrix']);expected[:3,3]*=.1
    np.testing.assert_allclose(actual['transform_matrix'],expected)
    assert f==source


def test_path_anchors_and_intrinsics_are_calibration_only():
    a,b,c=camera('a',0),camera('b',2),camera('c',4)
    b['fl_x']=120
    path=calibration_path({'frames':[a,b,c]},['a','b','c'],4)
    assert len(path)==9
    assert [path[k]['physical_camera'] for k in [0,4,8]]==['a','b','c']
    assert path[2]['transform_matrix'][0][3]==1
    assert path[2]['fl_x']==110
    np.testing.assert_allclose(np.asarray(path[2]['transform_matrix'])[:3,:3],np.eye(3))


def test_distorted_source_is_rejected():
    f=camera('a',0);f['k1']=.1
    with pytest.raises(ValueError,match='undistorted'):
        normalize_frame(f,{}, {'dataparser_transform':np.eye(4)[:3].tolist(),'dataparser_scale':1})


def test_primary_bandwidth_flag_is_rejected_before_reading_data_without_prior(tmp_path,monkeypatch):
    from render_patchmatch_camera_path import main
    args=['path','--seam-cut-bandwidth-allow-primary']
    for option in ['data','mesh','mesh-metadata','calibration','output']:args+=['--'+option,str(tmp_path/option)]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(SystemExit):main()


@pytest.mark.parametrize('extra',[
    ['--seam-cut-local-bandwidth-penalty','nan'],
    ['--seam-cut-local-bandwidth-penalty','-.1'],
    ['--seam-cut-local-bandwidth-penalty','.01'],
    ['--seam-cut-local-bandwidth-penalty','.01','--aggregation-mode','seam-cut','--seam-cut-bandwidth-penalty','.01'],
])
def test_local_bandwidth_path_invalid_configuration_fails_early(tmp_path,monkeypatch,extra):
    from render_patchmatch_camera_path import main
    args=['path',*extra]
    for option in ['data','mesh','mesh-metadata','calibration','output']:args+=['--'+option,str(tmp_path/option)]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(SystemExit):main()


def test_mesh_camera_color_requires_exact_native_calibration_before_data_read(tmp_path,monkeypatch):
    from render_patchmatch_camera_path import main
    args=['path','--mesh-camera-color',str(tmp_path/'field.json')]
    for option in ['data','mesh','mesh-metadata','calibration','output']:args+=['--'+option,str(tmp_path/option)]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(SystemExit):main()


@pytest.mark.parametrize('option',['--surface-texture-registration','--source-rgb-depth-aware-sampling'])
def test_mesh_camera_path_rejects_rgb_sampling_overrides(tmp_path,monkeypatch,option):
    from render_patchmatch_camera_path import main
    args=['path','--mesh-camera-color',str(tmp_path/'field.json'),
          '--camera-color-calibration',str(tmp_path/'color.json'),'--camera-color-model','spatial',
          '--exact-mesh-visibility','--pixel-center-offset','.5',option]
    for key in ['data','mesh','mesh-metadata','calibration','output']:args+=['--'+key,str(tmp_path/key)]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(SystemExit):main()
