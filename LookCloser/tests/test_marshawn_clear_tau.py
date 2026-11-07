import importlib.util
from pathlib import Path
import pytest
import torch
p=Path(__file__).resolve().parents[1]/'scripts/probe_marshawn_clear_tau.py';s=importlib.util.spec_from_file_location('clear_tau_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_midpoints_integrate_constant_density_and_have_unsaturated_gradient():
 near=torch.tensor([.2,.9]);far=torch.tensor([1.3,1.4]);density=torch.tensor(8.,requires_grad=True)
 for n in(256,512,1024):
  t=m.midpoint_distances(near,far,n);assert bool((t>near[:,None]).all())and bool((t<far[:,None]).all());tau=(density*torch.ones_like(t)).mean(1)*(far-near);g=torch.autograd.grad(tau.mean(),density,retain_graph=True)[0];torch.testing.assert_close(g,(far-near).mean());assert float((1-torch.exp(-tau))[0])>.999

def test_quadrature_rejects_empty_or_unreviewed_intervals():
 with pytest.raises(ValueError):m.midpoint_distances(torch.tensor([1.]),torch.tensor([.9]),512)
 with pytest.raises(ValueError):m.midpoint_distances(torch.tensor([0.]),torch.tensor([1.]),7)
