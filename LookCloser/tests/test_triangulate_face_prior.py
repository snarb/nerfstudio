import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from triangulate_face_prior import triangulate,projection_matrices,project
from study_multiview_face_prior import portrait_to_native

def cameras():
    rows=[]
    for x,y in [(-.3,0),(-.2,.1),(-.1,-.1),(0,.1),(.1,0),(.2,-.1),(.3,.1),(.4,0)]:
        pose=np.eye(4);pose[:3,3]=[x,y,0]
        rows.append(dict(transform_matrix=pose,fl_x=1000,fl_y=950,cx=960,cy=540,w=1920,h=1080))
    return rows

@pytest.mark.parametrize('arm',['all_fit_robust','train_consensus'])
def test_calibrated_recovery_and_independent_view(arm):
    rows=cameras();truth=np.array([.03,.02,-2.]);uv,z=project(truth,projection_matrices(rows))
    point,error,selected,angle=triangulate(rows[:6],uv[0,:6],arm)
    np.testing.assert_allclose(point,truth,atol=1e-8)
    np.testing.assert_allclose(project(point,projection_matrices(rows[6:]))[0][0],uv[0,6:],atol=1e-7)
    assert selected.all() and error.max()<1e-7 and angle>1 and (z>0).all()

def test_consensus_rejects_wrong_correspondence_without_using_validation():
    rows=cameras();truth=np.array([.03,.02,-2.]);xy=project(truth,projection_matrices(rows))[0][0]
    xy[0]+=[80,-30]
    point,error,selected,_=triangulate(rows,xy,'train_consensus')
    np.testing.assert_allclose(point,truth,atol=1e-8)
    assert not selected[0] and selected.sum()==7 and error[0]>70

def test_degenerate_and_missing_observations_fail_closed():
    rows=cameras()
    with pytest.raises(ValueError):triangulate(rows[:2],np.zeros((2,2)),'all_fit_robust')
    with pytest.raises(ValueError):triangulate(rows[:3],np.full((3,2),np.nan),'train_consensus')
    with pytest.raises(ValueError):triangulate([rows[0]]*3,np.array([[500,500]]*3),'all_fit_robust')

def test_portrait_native_inverse_uses_pixel_index_not_image_width():
    xy=np.array([[0,0],[1079,1919],[300.25,700.75]])
    native=portrait_to_native(xy)
    np.testing.assert_allclose(native,[[1919,0],[0,1079],[1218.25,300.25]])
