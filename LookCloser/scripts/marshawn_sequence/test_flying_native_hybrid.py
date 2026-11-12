import unittest
import numpy as np
from flying_native_hybrid import native_png_sample

class NativeSamplingTests(unittest.TestCase):
    def test_native_pixel_centers_survive_geometry_downscale(self):
        raw=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
        # Geometry is 4x4; native (2,4) maps by boundary-coordinate scaling.
        uv=np.array([[(2+.5)/2-.5,(4+.5)/2-.5]])
        rgb,valid=native_png_sample(raw,uv,dict(w=4,h=4))
        self.assertTrue(valid[0])
        np.testing.assert_allclose(rgb[0],raw[4,2]/255,atol=2e-7)

    def test_invalid_taps_rejected_and_linear_interpolation(self):
        raw=np.zeros((3,3,3),np.uint8);raw[:,1:]=255
        uv=np.array([[.5,.5],[-1,0],[np.nan,0],[2,2]])
        rgb,valid=native_png_sample(raw,uv,dict(w=3,h=3))
        np.testing.assert_array_equal(valid,[True,False,False,False])
        np.testing.assert_allclose(rgb[0],1.055*.5**(1/2.4)-.055,atol=2e-7)

if __name__=='__main__':unittest.main()
