from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from refine_patchmatch_mesh_openmvs import compare_calibration,camera_inventory,read_openmvs_no_points_images


def test_camera_gate_rejects_pose_or_inventory_changes():
    before={'train.jpg':np.array([1920,1080,10000,10001,960,540,0.])}
    assert compare_calibration(before,{k:v.copy() for k,v in before.items()})['poses_optimized'] is False
    changed={'train.jpg':before['train.jpg'].copy()};changed['train.jpg'][-1]=.0001
    with pytest.raises(ValueError,match='fixed calibration'):compare_calibration(before,changed)
    with pytest.raises(ValueError,match='inventory'):compare_calibration(before,{'other.jpg':before['train.jpg']})


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_camera_gate_rejects_nonfinite(bad):
    with pytest.raises(ValueError,match='nonfinite'):
        compare_calibration({'train':np.array([1.])},{'train':np.array([bad])})


def test_binary_camera_inventory_preserves_precision(tmp_path):
    import struct
    params=[10190.051343591653,10020.4500052329,960.,540.]
    (tmp_path/'cameras.bin').write_bytes(struct.pack('<QiiQQ4d',1,1,1,1920,1080,*params))
    (tmp_path/'images.bin').write_bytes(struct.pack('<Qi7di',1,1,1.,0.,0.,0.,.001,.002,.003,1)
                                       +b'images/train.jpg\0'+struct.pack('<Q',0))
    row=camera_inventory(tmp_path)['train.jpg']
    np.testing.assert_array_equal(row[:6],[1920,1080,*params])
    np.testing.assert_array_equal(row[-3:],[.001,.002,.003])
    path=tmp_path/'images.bin';path.write_bytes(path.read_bytes()[8:])
    np.testing.assert_array_equal(camera_inventory(tmp_path,openmvs_no_points=True)['train.jpg'],row)


@pytest.mark.parametrize('data',[b'x',b'x'*64,b'x'*64+b'\0'+b'\0'*7])
def test_headerless_reader_rejects_truncation(tmp_path,data):
    path=tmp_path/'images.bin';path.write_bytes(data)
    with pytest.raises(ValueError):read_openmvs_no_points_images(path)
