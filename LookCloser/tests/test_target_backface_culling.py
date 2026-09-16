import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_target_backface_culling import front_faces,remap_ids


def test_oriented_surface_is_front_only_on_one_side():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    t=np.array([[0,1,2],[0,2,1]])
    np.testing.assert_array_equal(front_faces(v,t,[0,0,1]),[True,False])
    np.testing.assert_array_equal(front_faces(v,t,[0,0,-1]),[False,True])
    assert not front_faces(v,t,[0,0,0]).any()


def test_ray_ids_keep_original_face_identity_and_miss_sentinel():
    miss=np.iinfo(np.uint32).max
    np.testing.assert_array_equal(remap_ids(np.array([[0,miss],[1,2]],np.uint32),[7,11,19]),
                                  [[7,miss],[11,19]])
