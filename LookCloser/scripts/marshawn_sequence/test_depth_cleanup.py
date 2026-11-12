"""Verify the actual sample-density hook's calibrated projection and removal."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from nerfstudio.field_components.field_heads import FieldHeadNames
from render_depth_cleanup import install_depth_gate, depth_support


class Field(torch.nn.Module):
    def forward(self, samples):
        n = len(samples.frustums.get_positions())
        return {FieldHeadNames.DENSITY:torch.ones(n,1),FieldHeadNames.RGB:torch.full((n,3),.5)}


class DepthCleanupTests(unittest.TestCase):
    def test_projection_depth_gate_preserves_rgb_and_unhooks(self):
        field=Field();model=SimpleNamespace(device='cpu',field=field)
        camera=SimpleNamespace(camera_to_worlds=torch.eye(4)[:3][None],
            fx=torch.tensor(4.),fy=torch.tensor(4.),cx=torch.tensor(4.),cy=torch.tensor(4.))
        reference=np.ones((8,8),np.float32);distance=np.zeros((8,8),np.float32)
        position=torch.tensor([[0.,0.,-1.],[0.,0.,-2.],[0.,0.,1.],[20.,0.,-1.]])
        samples=SimpleNamespace(frustums=SimpleNamespace(get_positions=lambda:position))
        handle=install_depth_gate(model,camera,reference,distance,.15,.1,8,6)
        output=field(samples)
        torch.testing.assert_close(output[FieldHeadNames.DENSITY][:,0],torch.tensor([1.,0.,0.,0.]))
        torch.testing.assert_close(output[FieldHeadNames.RGB],torch.full((4,3),.5))
        handle.remove()
        torch.testing.assert_close(field(samples)[FieldHeadNames.DENSITY],torch.ones(4,1))

    def test_connected_support_rejects_isolated_opaque_speck(self):
        alpha=np.zeros((100,100,1),np.float32);alpha[10:70,10:70]=1;alpha[90:93,90:93]=1
        depth=np.full_like(alpha,2.);depth[10:70,10:70]=1
        core,distance,reference=depth_support(dict(accumulation=alpha,depth=depth))
        self.assertTrue(core[20,20]);self.assertFalse(core[91,91])
        self.assertEqual(reference[91,91],1.)
        self.assertGreater(distance[91,91],20)

    def test_major_support_keeps_partial_lower_actor_but_not_overhead_island(self):
        alpha=np.zeros((400,400,1),np.float32)
        alpha[30:200,30:170]=1
        alpha[5:25,220:380]=1
        alpha[320:380,380:400]=1
        alpha[30:80,390:400]=1
        core,_,_=depth_support(dict(accumulation=alpha,depth=np.ones_like(alpha)),10000,True)
        self.assertTrue(core[100,100]);self.assertTrue(core[350,395])
        self.assertFalse(core[10,300]);self.assertFalse(core[40,395])

    def test_large_flat_island_is_not_a_body_anchor(self):
        alpha=np.zeros((400,600,1),np.float32)
        alpha[30:300,30:200]=1
        alpha[5:70,250:500]=1
        core,_,_=depth_support(dict(accumulation=alpha,depth=np.ones_like(alpha)),10000,True,120)
        self.assertTrue(core[100,100]);self.assertFalse(core[30,300])


if __name__=='__main__':unittest.main()
