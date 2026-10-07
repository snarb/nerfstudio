"""Spatial permissions must not enable alpha on skin through voxel quantization."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_luster_semantic_permissions import voxel_skin_veto


def test_skin_inside_voxel_footprint_is_vetoed_even_when_center_is_unprotected():
    row=dict(transform_matrix=np.eye(4).tolist(),fl_x=100.,fl_y=100.,cx=50.5,cy=50.5)
    protected=np.zeros((100,100),dtype=bool)
    protected[49:52,53:56]=True
    # Center projects to(52,50), outside skin; the voxel extends onto skin.
    centers=np.array([[.04,0.,-2.],[-.3,0.,-2.]])
    assert voxel_skin_veto(centers,np.array([.08,.08,.08]),row,protected).tolist()==[True,False]


def test_camera_plane_crossing_is_denied_and_hidden_voxel_is_ignored():
    row=dict(transform_matrix=np.eye(4).tolist(),fl_x=100.,fl_y=100.,cx=50.5,cy=50.5)
    protected=np.zeros((100,100),dtype=bool)
    centers=np.array([[0.,0.,0.],[0.,0.,2.]])
    assert voxel_skin_veto(centers,np.ones(3)*.1,row,protected).tolist()==[True,False]
