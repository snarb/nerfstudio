from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_source_epipolar_residuals import fundamental,project,signed_epipolar_distance,peak_offset


def test_exact_correspondence_and_depth_motion_remain_epipolar():
    pose=np.eye(4)
    a={'transform_matrix':pose.tolist(),'fl_x':1000,'fl_y':1000,'cx':500,'cy':400}
    pose[0,3]=.1;b={**a,'transform_matrix':pose.tolist()};f=fundamental(a,b)
    pa=project(a,np.array([.02,.03,-2.]))
    for depth in (1.,2.,3.):
        pb=project(b,np.array([.01,.015,-1.])*depth)
        assert abs(signed_epipolar_distance(f,pa,pb))<1e-8
        shifted=pb+np.array([0,2,0])
        assert abs(abs(signed_epipolar_distance(f,pa,shifted))-2)<1e-8


def test_subpixel_quadratic_peak():
    y,x=np.indices((7,7));scores=-((x-3.2)**2+2*(y-2.7)**2)
    np.testing.assert_allclose(peak_offset(scores,3,3),[.2,-.3],atol=1e-8)
