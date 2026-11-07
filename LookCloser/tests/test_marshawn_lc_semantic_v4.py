"""Mixed precision proof regression; native tensors and tolerances stay intact."""
from pathlib import Path
import sys

import pytest
import torch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parent)]
from test_marshawn_lc_semantic_capture import Model,Rays
from marshawn_lc_semantic_capture import capture_lc_native as prior_capture
from marshawn_lc_semantic_capture_v4 import capture_lc_native


class MixedField(torch.nn.Module):
    def __init__(self,dtype):
        super().__init__()
        self.register_buffer('color',torch.tensor([[.125,.25,.5],[.875,.75,.625],[.375,.5,.625]],dtype=dtype))

    def forward(self,samples):
        return {'density':torch.tensor([[2.],[3.],[4.]],dtype=torch.float32),'rgb':self.color}


class MixedModel(Model):
    def __init__(self,dtype,defect=None):
        super().__init__(defect)
        self.field=MixedField(dtype)
        self.eval().requires_grad_(False)

    def renderer_rgb(self,rgb,weights,ray_indices,num_rays):
        product=weights*rgb
        return product.new_zeros((num_rays,3)).index_add(0,ray_indices.long(),product)


def capture(model,method=capture_lc_native):
    return method(model,Rays(),density_key='density',rgb_key='rgb')


def serial_density_rgb(row,ray_count):
    transmittance=torch.ones(ray_count,dtype=torch.float64)
    color=torch.zeros(ray_count,3,dtype=torch.float64)
    alpha=torch.zeros(ray_count,dtype=torch.float64)
    for i,start,end,sigma,rgb in zip(row['ray_indices'],row['starts'],row['ends'],row['sigma'],row['rgb']):
        optical_depth=sigma.double()*(end.double()-start.double())
        a=-torch.expm1(-optical_depth)
        weight=transmittance[i]*a
        color[i]+=weight*rgb.double()
        alpha[i]+=weight
        transmittance[i]*=torch.exp(-optical_depth)
    return color,alpha


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_native_product_promotion_matches_independent_serial_multiray(dtype):
    model=MixedModel(dtype)
    row=capture(model)
    expected_rgb,expected_alpha=serial_density_rgb(row,3)
    assert row['rgb'].dtype==dtype
    assert row['weight'].dtype==torch.float32
    assert row['native_rgb'].dtype==torch.promote_types(dtype,torch.float32)
    assert torch.equal(row['rgb'],model.field.color)
    assert torch.allclose(row['native_rgb'].double(),expected_rgb,atol=1e-7,rtol=1e-6)
    assert torch.allclose(row['native_alpha'].double(),expected_alpha,atol=1e-7,rtol=1e-6)
    assert row['native_alpha'][1]==0 and row['native_rgb'][1].tolist()==[0.,0.,0.]
    assert max(row['proof'].values())<2e-5
    assert row['field_forward_calls']==1
    assert 'forward' not in model.field.__dict__


def test_actual_v3_half_float_accumulator_failure_is_reproduced():
    with pytest.raises(RuntimeError,match='same scalar type'):
        capture(MixedModel(torch.float16),method=prior_capture)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_empty_batch_with_mixed_precision_field_never_calls_field(dtype):
    row=capture(MixedModel(dtype,defect='empty'))
    assert row['field_forward_calls']==0
    assert row['xyz'].shape==(0,3)
    assert row['native_alpha'].tolist()==[0.,0.,0.]
    assert row['native_rgb'].tolist()==[[0.,0.,0.]]*3
