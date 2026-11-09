"""Independent integral/finite-difference checks for the uninstalled LOS prototype."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_true_log_los_prototype import loss_from_log_sigma

def test_exact_constant_density_integral_and_finite_difference():
    x=torch.tensor([[-20.,-20.,-20.],[-40.,-40.,-40.]],dtype=torch.float64,requires_grad=True)
    length=torch.tensor([.5,2.],dtype=torch.float64);v,_=loss_from_log_sigma(x,length,.01,'exact_tau')
    expected=.01*(.5*torch.exp(x[0,0])+2*torch.exp(x[1,0]))/2
    torch.testing.assert_close(v,expected,rtol=1e-12,atol=0)
    g=torch.autograd.grad(v,x)[0];h=.001;a=x.detach().clone();a[1,1]+=h;b=x.detach().clone();b[1,1]-=h
    fd=(loss_from_log_sigma(a,length,.01,'exact_tau')[0]-loss_from_log_sigma(b,length,.01,'exact_tau')[0])/(2*h)
    torch.testing.assert_close(g[1,1],fd,rtol=1e-4,atol=0)

def test_sample_multiplicity_does_not_change_integral():
    for mode in['exact_tau','robust_log1p_tau']:
        values=[]
        for n in[1,64]:
            x=torch.full((2,n),-2.,dtype=torch.float64,requires_grad=True);v,_=loss_from_log_sigma(x,torch.tensor([.5,2.]),.01,mode);g=torch.autograd.grad(v,x)[0];values.append((v,g.sum(1)))
        torch.testing.assert_close(values[0][0],values[1][0],rtol=1e-12,atol=0);torch.testing.assert_close(values[0][1],values[1][1],rtol=1e-12,atol=0)

def test_robust_gradient_bounded_and_vanishes_in_clear_space():
    x=torch.tensor([[-60.],[0.],[1000.]],dtype=torch.float64,requires_grad=True);v,_=loss_from_log_sigma(x,torch.ones(3),.01,'robust_log1p_tau');g=torch.autograd.grad(v,x)[0]
    assert torch.isfinite(v)and torch.isfinite(g).all()and(g>=0).all()and(g<=.01/3+1e-15).all()
    assert 0<float(g[0])<1e-25
    torch.testing.assert_close(g[-1],torch.tensor([.01/3],dtype=torch.float64),rtol=1e-12,atol=0)
    with pytest.raises(FloatingPointError):loss_from_log_sigma(x,torch.ones(3),.01,'exact_tau')

def test_invalid_contract_rejected():
    with pytest.raises(ValueError):loss_from_log_sigma(torch.zeros(2,3),torch.ones(2),.01,'implicit')
    with pytest.raises(ValueError):loss_from_log_sigma(torch.zeros(2,3),torch.tensor([1.,0.]),.01,'exact_tau')
