from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_dec5_depth_interpolation import normalized_depth_sample
from study_dec5_hair_geometry import sample


def test_unknown_zero_does_not_create_a_fake_half_depth_surface():
    depth=torch.tensor([[[2.,0.],[2.,0.]]]);uv=torch.tensor([[1.,.5],[.5,.5],[1.5,.5]])
    naive=sample(depth,uv);result=normalized_depth_sample(depth,uv)
    torch.testing.assert_close(naive[:,0],torch.tensor([1.,2.,0.]))
    torch.testing.assert_close(result[:,0],torch.tensor([2.,2.,0.]))


def test_fully_valid_depths_keep_the_original_bilinear_result():
    depth=torch.tensor([[[1.,2.],[3.,4.]]]);uv=torch.tensor([[1.,1.],[.8,.7]])
    torch.testing.assert_close(normalized_depth_sample(depth,uv),sample(depth,uv))
