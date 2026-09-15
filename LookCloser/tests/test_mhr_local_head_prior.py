import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_mhr_local_head_prior import torch_rotation
from study_mhr_local_head_prior import MODEL_POINTS,LANDMARKS
from study_multiview_face_prior import portrait_to_native,CROP

def test_differentiable_rotation_matches_calibrated_rodrigues():
    w=torch.tensor([.2,-.1,.3],dtype=torch.float64,requires_grad=True);r=torch_rotation(w)
    np.testing.assert_allclose(r.detach().numpy(),Rotation.from_rotvec(w.detach().numpy()).as_matrix(),atol=1e-12)
    r[0,1].backward();assert torch.isfinite(w.grad).all() and w.grad.abs().sum()>0

def test_zero_rotation_has_finite_nonzero_gradient():
    w=torch.zeros(3,dtype=torch.float64,requires_grad=True);r=torch_rotation(w);r[0,1].backward()
    np.testing.assert_array_equal(r.detach().numpy(),np.eye(3));assert torch.isfinite(w.grad).all() and w.grad[2]==-1

def test_approximate_chin_annotation_is_initialization_only():
    assert 152 in MODEL_POINTS and 152 not in LANDMARKS
    assert LANDMARKS==[1,33,133,362,263,61,291]

def test_skin_crop_coordinates_map_to_integer_native_rays():
    xy=np.array([[0.,0.],[100,500]])+np.array(CROP[:2]);native=portrait_to_native(xy)
    np.testing.assert_array_equal(native,[[1519,0],[1019,100]])
