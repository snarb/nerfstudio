from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from carve_patchmatch_mesh_free_space import free_space_evidence,train_frames


def test_far_surface_vetoes_foreground_but_occluder_does_not():
    depth=np.ones((9,9),np.float32);u=v=np.array([4.,4.,4.]);z=np.array([.9,1.,1.1])
    free,near=free_space_evidence(depth,u,v,z)
    np.testing.assert_array_equal(free,[True,False,False]);np.testing.assert_array_equal(near,[False,True,False])


def test_far_outliers_and_missing_or_mixed_depth_are_not_free_space():
    depth=np.zeros((9,9),np.float32);depth[4,4]=1
    u=v=np.array([4.]);z=np.array([.9])
    assert not free_space_evidence(depth,u,v,z)[0].any()
    depth[:]=1;depth[2:5,2:7]=1.2
    assert not free_space_evidence(depth,u,v,z)[0].any()
    depth[:]=1;depth[4,4]=.9
    assert free_space_evidence(depth,u,v,z)[0].item()


def test_train_inventory_forbids_heldout_even_if_mislabelled():
    frames=[{'file_path':f'frame_train_{n:05d}.jpg','depth_file_path':f'{n}.npy.gz','physical_camera':f'cam{n}'} for n in range(62)]
    payload={'frames':frames,'train_filenames':[f['file_path'] for f in frames]}
    assert len(train_frames(payload))==62
    frames[0]['physical_camera']='F004_B005_1210O9'
    with pytest.raises(ValueError,match='held-out'):train_frames(payload)
