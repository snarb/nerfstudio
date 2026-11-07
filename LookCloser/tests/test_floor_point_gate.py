"""CPU checks for boundary policy and preservation of field coordinates."""
import unittest
import torch
from marshawn_floor_point_gate_worker import install_point_gate


class Field:
    def _normalize_positions(self,positions):
        shape=positions.shape[:-1]
        return ((positions+7)/14).reshape(-1,3),positions[...,0]>0,shape,positions.reshape(-1,3)


class FloorGateTests(unittest.TestCase):
    def test_boundary_and_existing_rejection(self):
        field=Field();xyz=torch.tensor([[1.,0.,-.02001],[1.,0.,-.02],[1.,0.,-.01999],[-1.,0.,.5]])
        original=field._normalize_positions(xyz);restore=install_point_gate(field,-.02);result=field._normalize_positions(xyz)
        self.assertEqual(result[1].tolist(),[False,True,True,False])
        torch.testing.assert_close(result[0],original[0],rtol=0,atol=0);torch.testing.assert_close(result[3],original[3],rtol=0,atol=0)
        self.assertEqual(result[2],original[2]);field._normalize_positions=restore
        torch.testing.assert_close(field._normalize_positions(xyz)[1],original[1])
    def test_multidimensional_and_empty(self):
        for xyz in [torch.tensor([[[1.,0.,-.03],[1.,0.,.02]]]),torch.empty((0,3))]:
            field=Field();original=field._normalize_positions(xyz);install_point_gate(field,-.02);result=field._normalize_positions(xyz)
            self.assertEqual(result[1].shape,original[1].shape);torch.testing.assert_close(result[0],original[0],rtol=0,atol=0)
            torch.testing.assert_close(result[3],original[3],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
