"""Spatial convention and nonmutating density gating, without GPU allocation."""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from dec5_empty_space_gate import EmptySpaceGate
from nerfstudio.field_components.field_heads import FieldHeadNames


def gate_fixture():
 gate=EmptySpaceGate.__new__(EmptySpaceGate);gate.bounds=torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]])
 gate.volume=torch.zeros(1,1,5,5,5);gate.volume[...,3:]=1
 return gate


def test_voxel_xyz_conventions_interpolation_and_unknown_exterior():
 gate=gate_fixture();p=torch.tensor([[-.8,0,0],[.8,0,0],[0,.8,.8],[.2,0,0],[2.,0,0]])
 torch.testing.assert_close(gate.gate(p),torch.tensor([1.,0.,1.,.5,1.]),rtol=0,atol=1e-6)
 assert gate.gate(torch.empty(0,3)).shape==(0,)


def test_density_hook_keeps_original_outputs_and_other_heads():
 gate=gate_fixture();p=torch.tensor([[-.8,0,0],[.8,0,0]])
 original={FieldHeadNames.DENSITY:torch.tensor([[2.],[3.]]),FieldHeadNames.RGB:torch.ones(2,3)}
 samples=SimpleNamespace(frustums=SimpleNamespace(get_positions=lambda:p));result=gate.hook(None,(samples,),original)
 torch.testing.assert_close(original[FieldHeadNames.DENSITY],torch.tensor([[2.],[3.]]))
 torch.testing.assert_close(result[FieldHeadNames.DENSITY],torch.tensor([[2.],[0.]]))
 assert result[FieldHeadNames.RGB] is original[FieldHeadNames.RGB]
