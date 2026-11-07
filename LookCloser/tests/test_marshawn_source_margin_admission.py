import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_source_margin_admission import GeometricSourcePointAdmission
from marshawn_source_point_admission import SourcePointAdmission
from audit_marshawn_source_margin_cpu import admitted


def test_actual_source_margin_wrapper_matches_independent_cpu_and_unknown():
    rng = np.random.default_rng(9)
    mask = rng.random((128, 128, 128)) > .98
    bounds = np.array([[-.84, -.21, -.07], [.15, .42, .63]])
    points = rng.uniform(bounds[0]-.01, bounds[1]+.01, (10000, 3))
    # Include numerical-contact boundaries and outside-domain UNKNOWN positions.
    points[:6] = [bounds[0]-1e-7, bounds[0]-1e-6, bounds[0]+1e-7,
                  bounds[1]+1e-7, bounds[1]+1e-6, bounds[1]-1e-7]
    last = None
    for margin in [0., .001, .002]:
        lookup = GeometricSourcePointAdmission(torch.from_numpy(mask), bounds, margin=margin, base_class=SourcePointAdmission)
        actual = lookup(torch.from_numpy(points)).numpy()
        np.testing.assert_array_equal(actual, admitted(points, mask, bounds, margin=margin))
        if last is not None:
            assert actual[last].all()
        last = actual


def test_zero_wrapper_exact_proven_selector_and_cap():
    mask = torch.zeros((128, 128, 128), dtype=torch.bool);mask[64,64,64] = True
    bounds = [[-.84, -.21, -.07], [.15, .42, .63]]
    points = torch.tensor([[0,0,0],[-1,0,0]], dtype=torch.float32)
    wrapper = GeometricSourcePointAdmission(mask, bounds, margin=0, base_class=SourcePointAdmission)
    assert torch.equal(wrapper(points), SourcePointAdmission(mask,bounds)(points))
    with pytest.raises(ValueError):
        GeometricSourcePointAdmission(mask,bounds,margin=.002461,base_class=SourcePointAdmission)
