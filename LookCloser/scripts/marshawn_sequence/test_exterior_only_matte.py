import unittest
import numpy as np
from exterior_only_matte import exterior_gate

class ExteriorTests(unittest.TestCase):
 def test_internal_hole_is_never_a_boundary(self):
  a=np.zeros((120,120),float);a[20:100,20:100]=1;a[50:70,50:70]=0
  gate,_,_,_=exterior_gate(a)
  self.assertTrue(np.all(gate[40:80,40:80]==0))
 def test_thin_bat_is_protected(self):
  a=np.zeros((120,160),float);a[30:100,20:80]=1;a[50:55,70:145]=1
  gate,_,_,_=exterior_gate(a)
  self.assertTrue(np.all(gate[50:55,80:145]==0))
 def test_touching_people_have_no_internal_seam_gate(self):
  a=np.zeros((120,160),float);a[20:100,20:90]=1;a[40:110,70:140]=1
  gate,_,_,_=exterior_gate(a)
  self.assertTrue(np.all(gate[45:95,72:88]==0))
 def test_only_near_outer_contour_changes(self):
  a=np.zeros((120,120),float);a[20:100,20:100]=1
  gate,_,_,_=exterior_gate(a)
  self.assertEqual(gate[0,0],0);self.assertEqual(gate[60,60],0);self.assertGreater(gate[20,60],0)
if __name__=='__main__':unittest.main()
