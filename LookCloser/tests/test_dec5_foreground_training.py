from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_foreground_training import crop_indices,opacity_target_thickness


def test_foreground_lookup_matches_actual_rotated_head_crop():
    image=np.arange(35).reshape(5,7);crop=[1,2,5,6]
    portrait=np.rot90(image)[crop[1]:crop[3],crop[0]:crop[2]]
    y,x=np.indices(image.shape)
    indices=torch.from_numpy(np.stack((np.zeros_like(y),y,x),-1).reshape(-1,3))
    row,col=crop_indices(indices,7,crop)
    valid=(row>=0)&(row<portrait.shape[0])&(col>=0)&(col<portrait.shape[1])
    np.testing.assert_array_equal(portrait[row[valid],col[valid]],image.ravel()[valid])


def test_optical_thickness_retains_empty_space_gradient_at_saturated_opacity():
    density=torch.tensor([100.],requires_grad=True)
    opacity=-torch.expm1(-density)
    assert opacity.item()==1.
    opacity.sum().backward();assert density.grad.item()<1e-30
    density.grad.zero_()
    objective=torch.nn.functional.smooth_l1_loss(density,opacity_target_thickness(torch.zeros(1)))
    objective.backward();assert density.grad.item()==1.
    alpha=torch.tensor([0.,.1,.5,.9,.999])
    torch.testing.assert_close(-torch.expm1(-opacity_target_thickness(alpha)),alpha)


def test_adaptive_optical_thickness_is_invariant_to_interval_partition_and_differentiable():
    from dec5_foreground_training import integrate_packed_thickness
    density=torch.tensor([2.,2.,3.],requires_grad=True)
    tau=integrate_packed_thickness(density,torch.tensor([.1,.4,.2]),torch.tensor([0,0,1]),3)
    # Uniform density 2 over length .5 gives opacity 1-exp(-1), regardless
    # of adaptive subdivision; the third ray has no occupied intervals.
    assert torch.allclose(1-torch.exp(-tau),torch.tensor([1-np.exp(-1),1-np.exp(-.6),0.],dtype=torch.float32))
    tau.sum().backward()
    assert torch.allclose(density.grad,torch.tensor([.1,.4,.2]))


def test_opacity_likelihood_has_soft_alpha_optimum_and_nonvanishing_empty_ray_gradient():
    from dec5_foreground_training import opacity_bernoulli_nll
    alpha=torch.tensor([0.,.3,1.])
    tau=torch.tensor([.001,-np.log(.7),5.],dtype=torch.float32,requires_grad=True)
    objective=opacity_bernoulli_nll(tau,alpha)
    objective.sum().backward()
    assert abs(float(tau.grad[0])-1)<1e-6
    assert abs(float(tau.grad[1]))<1e-6
    assert -0.01<float(tau.grad[2])<0
    assert bool(torch.isfinite(objective).all())
