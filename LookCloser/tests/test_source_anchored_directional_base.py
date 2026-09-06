from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_source_anchored_directional_base import relative_base_offset


def test_source_camera_identity_is_exact_even_for_nonconstant_models():
    random=np.random.default_rng(22);coeff=random.normal(size=(20,9,3)).astype(np.float32)
    world=random.normal(size=(20,3)).astype(np.float32)
    delta=relative_base_offset(coeff,world,[3,2,1],[3,2,1],2)
    np.testing.assert_array_equal(delta,np.zeros((20,3),np.float32))


def test_transfer_is_antisymmetric_and_constant_base_cancels():
    random=np.random.default_rng(33);coeff=random.normal(size=(20,9,3)).astype(np.float32)
    points=random.normal(size=(20,3)).astype(np.float32)
    a=relative_base_offset(coeff,points,[3,2,1],[-3,-2,-1],2)
    b=relative_base_offset(coeff,points,[-3,-2,-1],[3,2,1],2)
    np.testing.assert_array_equal(a,-b)
    coeff[:,0]+=5
    np.testing.assert_allclose(relative_base_offset(coeff,points,[3,2,1],[-3,-2,-1],2),a,atol=1e-6,rtol=1e-6)
