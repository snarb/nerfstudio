from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_patchmatch_camera_colors import project_points


def test_calibration_projection_matches_native_rgb_pixel_centers():
    frame={'transform_matrix':np.eye(4).tolist(),'fl_x':120.,'fl_y':100.,'cx':10.,'cy':8.}
    xy=torch.tensor([[2.,3.],[12.,5.],[5.,9.]])
    depth=torch.tensor([1.,2.,3.])
    points=torch.stack(((xy[:,0]+.5-10)/120*depth,-(xy[:,1]+.5-8)/100*depth,-depth),dim=1)
    u,v,z=project_points(points,frame,.5)
    torch.testing.assert_close(torch.stack((u,v),1),xy)
    torch.testing.assert_close(z,depth)
    old_u,old_v,_=project_points(points,frame)
    torch.testing.assert_close(torch.stack((old_u,old_v),1),xy+.5)


def test_projection_is_camera_transform_invariant_and_rejects_unknown_offset():
    pose=torch.tensor([[0.,-1.,0.,2.],[1.,0.,0.,3.],[0.,0.,1.,4.],[0.,0.,0.,1.]])
    local=torch.tensor([[.1,.2,-2.],[-.1,.3,-1.]])
    frame={'transform_matrix':pose.tolist(),'fl_x':120.,'fl_y':100.,'cx':10.,'cy':8.}
    world=local@pose[:3,:3].T+pose[:3,3]
    actual=project_points(world,frame,.5)
    frame['transform_matrix']=np.eye(4).tolist()
    expected=project_points(local,frame,.5)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b)
    with pytest.raises(ValueError,match='offset'):project_points(world,frame,1.)
