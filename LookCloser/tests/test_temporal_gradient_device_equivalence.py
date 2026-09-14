import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hard_source_gradient_leveling import level_source_gradients


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA equivalence requires a GPU')
def test_same_float64_problem_on_cpu_and_cuda():
    torch.set_num_threads(2)
    y,x=torch.meshgrid(torch.arange(36),torch.arange(40),indexing='ij')
    source=(.3+.05*(x%2)+.02*(y%3)).float()[None].repeat(3,1,1)
    alternate=source+.06
    labels=(x>19).long();depth=torch.ones_like(x,dtype=torch.float32)
    valid=torch.ones_like(labels,dtype=torch.bool)
    prediction=torch.where(labels[None]>0,alternate,source)
    results=[]
    for device in ['cpu','cuda']:
        output,offset,stats=level_source_gradients(prediction.to(device),labels.to(device),
            [source.to(device),alternate.to(device)],[valid.to(device),valid.to(device)],depth.to(device))
        assert stats['solver']['max_relative_residual']<5e-9
        assert stats['solver']['dtype']=='float64'
        results.append((output.cpu(),offset.cpu()))
    for i in [0,1]:
        torch.testing.assert_close(results[0][i],results[1][i],atol=1e-6,rtol=0)
