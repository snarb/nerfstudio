from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from nerfstudio.utils.lookcloser_projection import normalized_frequency_resolution


def test_normalized_jacobian_matches_finite_difference_for_off_axis_anisotropic_box():
    dtype=torch.float64;box=torch.tensor([[-1.,-2.,-3.],[2.,1.,1.]],dtype=dtype)
    position=torch.tensor([[.7,-.3,-2.]],dtype=dtype);pose=torch.eye(4,dtype=dtype)[None,:3]
    focal=torch.tensor([[800.,900.]],dtype=dtype);size=torch.tensor([[1600.,900.]],dtype=dtype)
    result,valid=normalized_frequency_resolution(position,pose,focal,size,box,torch.ones(1,dtype=dtype))
    def project(p):return torch.stack([.5*p[0]/-p[2],-p[1]/-p[2]])
    epsilon=1e-6;columns=[]
    for axis in range(3):
        delta=torch.zeros(3,dtype=dtype);delta[axis]=(box[1]-box[0])[axis]*epsilon
        columns.append((project(position[0]+delta)-project(position[0]-delta))/(2*epsilon))
    expected=torch.linalg.svdvals(torch.stack(columns,-1))[0]
    assert valid.item();torch.testing.assert_close(result[0],expected,rtol=1e-9,atol=1e-10)


def test_normalized_jacobian_is_invariant_to_uniform_scene_and_image_scaling():
    p=torch.tensor([[.2,.3,-2.],[.3,-.1,-3.]])
    pose=torch.eye(4)[None,:3].repeat(2,1,1);box=torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]])
    focal=torch.tensor([[1000.,1000.]]).repeat(2,1);size=torch.tensor([[2000.,1000.]]).repeat(2,1);freq=torch.tensor([100.,500.])
    a,_=normalized_frequency_resolution(p,pose,focal,size,box,freq)
    b,_=normalized_frequency_resolution(p*7,pose,focal*2,size*2,box*7,freq)
    torch.testing.assert_close(a,b)


def test_behind_camera_and_nonfinite_rays_cannot_update_frequency_grid():
    p=torch.tensor([[0.,0.,1.],[float('nan'),0.,-1.]])
    pose=torch.eye(4)[None,:3].repeat(2,1,1);box=torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]])
    result,valid=normalized_frequency_resolution(p,pose,torch.ones(2,2),torch.ones(2,2),box,torch.ones(2))
    assert not valid.any();assert torch.equal(result,torch.zeros(2))
