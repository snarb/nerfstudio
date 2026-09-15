import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from triangulate_hand_landmarks import triangulate
from temporal_rigid_patch import project_native


def cameras():
    result=[]
    for x,y in [(-.2,0),(0,.1),(.2,0),(0,-.1)]:
        pose=np.diag([1.,-1.,-1.,1.]);pose[:3,3]=[x,y,0]
        result.append(dict(transform_matrix=pose,fl_x=1500.,fl_y=1500.,cx=960.,cy=540.))
    return result


def test_recovers_point_and_unused_view():
    rows=cameras(); point=np.array([[.03,.02,2.]])
    pixels=np.array([project_native(point,c)[0][0] for c in rows])
    estimated,errors=triangulate(rows[:3],pixels[:3])
    np.testing.assert_allclose(estimated,point[0],atol=1e-10)
    np.testing.assert_allclose(project_native(estimated[None],rows[3])[0][0],pixels[3],atol=1e-8)
    assert errors.max()<1e-8


def test_rejects_invalid_or_degenerate_input():
    rows=cameras()
    with pytest.raises(ValueError):triangulate(rows[:2],np.zeros((2,2)))
    with pytest.raises(ValueError):triangulate(rows[:3],np.full((3,2),np.nan))
    with pytest.raises(ValueError):triangulate([rows[0]]*3,np.array([[960.,540.]]*3))
