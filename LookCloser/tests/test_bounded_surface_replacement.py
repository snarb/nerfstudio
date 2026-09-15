import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bounded_surface_replacement import removable_faces


def test_only_fully_in_domain_nearby_faces_removed():
    uv=np.array([[1,1],[2,1],[1,2],[3,3]],float);faces=np.array([[0,1,2],[1,2,3]])
    domain=np.zeros((4,4),bool);domain[:3,:3]=True;depth=np.ones((4,4))
    np.testing.assert_array_equal(removable_faces(uv,np.ones(4),faces,domain,depth),[True,False])
    z=np.ones(4);z[0]=1.013
    assert not removable_faces(uv,z,faces,domain,depth).any()


def test_nonfinite_out_of_frame_and_missing_model_preserved():
    faces=np.array([[0,1,2]]);domain=np.ones((4,4),bool);depth=np.ones((4,4))
    for uv in [np.array([[np.nan,1],[2,1],[1,2]]),np.array([[-1,1],[2,1],[1,2]])]:
        assert not removable_faces(uv,np.ones(3),faces,domain,depth).any()
    uv=np.array([[1,1],[2,1],[1,2]]);depth[1,1]=0
    assert not removable_faces(uv,np.ones(3),faces,domain,depth).any()
