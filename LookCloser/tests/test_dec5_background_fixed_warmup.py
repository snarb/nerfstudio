from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_background_separation import integrate_fixed_thickness
from dec5_foreground_training import integrate_packed_thickness


def test_fixed_and_packed_optical_thickness_agree_and_keep_opaque_ray_gradients():
    density=torch.tensor([[.2,.4,.6],[100.,200.,300.]],requires_grad=True)
    length=torch.tensor([3.,6.])
    fixed=integrate_fixed_thickness(density,length,2)
    packed=integrate_packed_thickness(density,length[:,None].expand(2,3)/3,torch.tensor([0,0,0,1,1,1]),2)
    torch.testing.assert_close(fixed,packed)
    fixed.sum().backward()
    torch.testing.assert_close(density.grad,torch.tensor([[1.,1.,1.],[2.,2.,2.]]))
    assert float(1-torch.exp(-fixed[1]))==1.  # Gradient survives opaque-alpha saturation.
