from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_warped_source_registration import relative_blur_profile


@pytest.mark.parametrize('side',['primary','source'])
def test_relative_blur_finds_known_bandwidth_difference(side):
    sharp=np.random.default_rng(3).random((48,48)).astype(np.float32)
    blurred=cv2.GaussianBlur(sharp,(0,0),1.)
    a,b=(sharp,blurred) if side=='primary' else (blurred,sharp)
    result=relative_blur_profile(a,b*2+.1)
    assert result['blurred_side']==side and result['sigma_pixels']==1.
    assert result['ncc']>.99999 and result['ncc_gain']>.1


def test_equal_patch_does_not_claim_blur():
    a=np.random.default_rng(1).random((48,48)).astype(np.float32)
    assert relative_blur_profile(a,a)['blurred_side']=='none'


def test_nonfinite_patch_rejected():
    with pytest.raises(ValueError,match='Nonfinite'):
        relative_blur_profile(np.full((48,48),np.nan),np.zeros((48,48)))
