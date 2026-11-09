import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_true_los import exact_forward_true_exp, TrueTauFreeSpaceObjective


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.exp()
    @staticmethod
    def backward(ctx, grad):
        return grad*ctx.saved_tensors[0].clamp(-15, 15).exp()


def test_exact_forward_bytes_and_finite_difference_gradient():
    raw=torch.tensor([-60.,-40.,-20.,-5.,0.,4.],dtype=torch.float64,requires_grad=True)
    native=2.3*STE.apply(raw)
    output=exact_forward_true_exp(native,raw,2.3)
    assert torch.equal(native,output)
    gradient=torch.autograd.grad(output.sum(),raw)[0]
    h=.0001
    finite_difference=2.3*((raw.detach()+h).exp()-(raw.detach()-h).exp())/(2*h)
    torch.testing.assert_close(gradient,finite_difference,rtol=1e-8,atol=0)
    assert float(gradient[0])<1e-25


def test_wrapped_geometry_reduction_rng_and_schema_unchanged():
    class Field:
        aabb=torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]])
        average_init_density=1.
        def density_fn(self,points):
            self._density_before_activation=points[:,:1]*0-20
            return STE.apply(self._density_before_activation)
    class Geometry:
        receipt={'bound':'original'}
        def sample_batch(self,step):return {'points':torch.zeros(6,3,requires_grad=True)}
        def __call__(self,field,step):
            x=field.density_fn(self.sample_batch(step)['points'])
            value=x.reshape(2,3).float().mean(1).mul(torch.tensor([1.,2.])).mean()*.01
            return value,{'mean':value.detach()}
    geometry=Geometry();wrapped=TrueTauFreeSpaceObjective.from_existing(geometry)
    before=torch.get_rng_state().clone()
    old=geometry(Field(),10);new=wrapped(Field(),10)
    assert torch.equal(old[0],new[0]) and torch.equal(old[1]['mean'],new[1]['mean'])
    assert wrapped.receipt is geometry.receipt and torch.equal(before,torch.get_rng_state())
    assert not isinstance(wrapped,torch.nn.Module)
    with pytest.raises(ValueError):TrueTauFreeSpaceObjective.from_existing(wrapped)


def test_nonfinite_is_not_hidden():
    with pytest.raises(FloatingPointError):
        exact_forward_true_exp(torch.tensor([float('inf')]),torch.tensor([100.]),1.)
