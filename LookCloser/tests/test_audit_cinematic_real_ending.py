from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_cinematic_real_ending import separable,response


def test_separable_replay_preserves_analytic_affine_image():
    y,x=np.mgrid[:7,:9]
    image=np.stack([x,y,x+y],axis=-1).astype(np.float32)
    source=dict(w=9,h=7,cx=4.5,cy=3.5,fl_x=6.,fl_y=8.)
    target=dict(source,fl_x=12.,fl_y=16.)
    expected=np.stack([.5*x+2,.5*y+1.5,.5*(x+y)+3.5],axis=-1)
    np.testing.assert_allclose(separable(image,source,target),expected,atol=1e-12)


def test_response_clamps_negative_values_and_matches_reinhard_half():
    image=np.array([[[-1.,0.,1.]]])
    np.testing.assert_array_equal(response(image,np.ones(3),1.),np.array([[[0,0,188]]],np.uint8))
