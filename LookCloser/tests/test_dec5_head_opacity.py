from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_head_opacity_training import head_ray_intervals,head_opacity_nll


def test_oriented_box_intervals_keep_original_ray_distance_and_parallel_misses():
    transform=torch.tensor([[.5,0,0,-1.],[0,.25,0,.5],[0,0,1.,0.]])
    origin=torch.tensor([[0.,0.,.5],[0.,8.,.5],[3.,0.,.5]])
    direction=torch.tensor([[1.,0,0],[1.,0,0],[1.,0,0]])
    near,far,valid=head_ray_intervals(origin,direction,transform)
    assert valid.tolist()==[True,False,True]
    torch.testing.assert_close(near[[0,2]],torch.tensor([2.,0.]))
    torch.testing.assert_close(far[[0,2]],torch.tensor([4.,1.]))


def test_soft_opacity_likelihood_has_correct_physical_optimum_and_empty_gradient():
    alpha=torch.tensor([.25,.75],dtype=torch.float64)
    tau=(-torch.log1p(-alpha)).requires_grad_();head_opacity_nll(tau,alpha).sum().backward()
    torch.testing.assert_close(tau.grad,torch.zeros_like(tau),atol=1e-10,rtol=0)
    tau=torch.tensor([0.,100.],requires_grad=True)
    value=head_opacity_nll(tau,torch.zeros(2));value.sum().backward()
    assert bool(torch.isfinite(value).all()) and bool((tau.grad==1).all())
