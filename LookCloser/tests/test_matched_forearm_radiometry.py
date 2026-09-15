import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from diagnose_matched_forearm_radiometry import bilinear, paired_summary
from probe_forearm_radiometry_shift import centered_ncc


def test_bilinear_matches_torch_and_clamps_border():
    rng = np.random.default_rng(17)
    a = rng.random((5,7,3))
    uv = rng.uniform([-2,-2], [9,7], (80,2))
    grid = torch.tensor((uv+.5)*[2/7,2/5]-1)[None,None]
    expected = torch.nn.functional.grid_sample(torch.tensor(a).permute(2,0,1)[None], grid,
        align_corners=False, padding_mode='border')[0,:,0].T.numpy()
    np.testing.assert_allclose(bilinear(a,uv), expected, atol=1e-14)
    np.testing.assert_allclose(bilinear(a[...,0],uv), expected[:,0], atol=1e-14)


def test_summary_only_uses_shared_domain_and_abstains():
    a = np.full((10,10,3), .2);b = a+.1
    valid = np.ones((10,10),bool);valid[:2] = False;b[:2] = 99
    r = paired_summary(a,b,valid)
    assert r['count'] == 80
    np.testing.assert_allclose(r['median_rgb_delta'], [25.5]*3)
    assert paired_summary(a,b,np.zeros_like(valid))['status'] == 'insufficient_overlap'


def test_centered_ncc_is_not_a_brightness_agreement_score():
    a = np.arange(90, dtype=float).reshape(30,3)/100
    assert abs(centered_ncc(a,a+[.2,.1,.4])-1) < 1e-12
    assert centered_ncc(np.ones_like(a),a) is None
