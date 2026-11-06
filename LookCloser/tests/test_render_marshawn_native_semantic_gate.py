"""Packed depth regression: removed foreground must not hide the isolated box."""
from pathlib import Path
import sys

import math
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from render_marshawn_native_semantic_gate import first_opaque_depth


def test_packed_current_density_depth_and_empty_ray():
    # Ray 0 crosses in its second segment; ray 1 stays transparent.
    starts = torch.tensor([1., 2., 4., 1.])
    ends = torch.tensor([2., 3., 5., 2.])
    sigma = torch.tensor([0., 1., 5., 0.])
    trans = torch.tensor([1., 1., torch.exp(torch.tensor(-1.)).item(), 1.])
    alpha = 1 - torch.exp(-sigma * (ends - starts))
    result = first_opaque_depth(starts, ends, sigma, trans, alpha,
                                torch.tensor([0, 0, 0, 1]), 3)
    assert math.isclose(result[0].item(), 2 + math.log(2), rel_tol=1e-6)
    assert torch.isinf(result[1:]).all()


def test_removed_foreground_depth_moves_to_retained_surface():
    starts = torch.tensor([1., 5.])
    ends = starts + 1
    ids = torch.tensor([0, 0])
    for sigma, expected in [(torch.tensor([2., 2.]), 1.),
                            (torch.tensor([0., 2.]), 5.)]:
        alpha = 1 - torch.exp(-sigma)
        trans = torch.cat([torch.ones(1), (1-alpha[:1])])
        result = first_opaque_depth(starts, ends, sigma, trans, alpha, ids, 1)
        assert math.isclose(result.item(), expected + math.log(2) / 2, rel_tol=1e-6)


if __name__ == "__main__":
    test_packed_current_density_depth_and_empty_ray()
    test_removed_foreground_depth_moves_to_retained_surface()
    print("2 packed depth regression checks passed")
