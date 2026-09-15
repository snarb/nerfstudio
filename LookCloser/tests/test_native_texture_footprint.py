import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_texture_footprint import snap_centers,relevant_tap,sample_native,original_sample


def test_zero_weight_neighbors_do_not_contribute_or_block():
    uv=snap_centers(torch.tensor([[[[4.0002,3.9998]]]]))
    assert relevant_tap(uv,0,0).item()
    assert not any(relevant_tap(uv,x,y).item() for x,y in [(1,0),(0,1),(1,1)])
    image=torch.full((1,3,8,8),1e6);image[:,:,4,4]=torch.tensor([.2,.3,.4])
    torch.testing.assert_close(sample_native(image,uv)[0,:,0,0],torch.tensor([.2,.3,.4]),rtol=0,atol=0)


def test_nonzero_fractional_taps_stay_required_and_sampling_unchanged():
    uv=torch.tensor([[[[4.25,3.75]]]])
    torch.testing.assert_close(snap_centers(uv),uv,rtol=0,atol=0)
    assert all(relevant_tap(uv,x,y).item() for x,y in [(0,0),(1,0),(0,1),(1,1)])
    image=torch.rand((1,3,8,8))
    torch.testing.assert_close(sample_native(image,uv),original_sample(image,uv),rtol=0,atol=0)


def test_snap_bound_and_per_axis_behavior():
    uv=torch.tensor([[[[500.0002,800.3],[500.01,800.0002]]]])
    result=snap_centers(uv)
    assert (result-uv).abs().max()<=.001
    assert result[0,0,0,0]==500 and result[0,0,0,1]==uv[0,0,0,1]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_mixed_batch_never_samples_bright_neighbor_at_integral_point():
    image=torch.full((2,3,8,8),1e6,device='cuda');image[:,:,4,4]=.25
    uv=snap_centers(torch.tensor([[[[4.0002,3.9998],[3.4,3.6]]]]*2,device='cuda'))
    actual=sample_native(image,uv)
    assert torch.all(actual[:,:,0,0]==.25)
    torch.testing.assert_close(actual[:,:,:,1:],original_sample(image,uv)[:,:,:,1:],rtol=0,atol=0)


def test_source_transform_is_guarded_and_does_not_edit_renderer():
    import inspect
    import render_smooth_temporal_mesh_video as renderer
    from study_native_texture_footprint import transform
    original=inspect.getsource(renderer.render_one);changed=transform(original)
    compile(changed,'test_footprint','exec')
    assert inspect.getsource(renderer.render_one)==original
    with pytest.raises(ValueError):transform('def render_one(): pass')
