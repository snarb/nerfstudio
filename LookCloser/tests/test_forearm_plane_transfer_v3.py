import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_forearm_plane_transfer_v2 import semantic_domain
from study_forearm_plane_transfer_v3 import footprint_domain


def test_v2_default_domain_is_unchanged():
    inside=np.array([True,False,True])
    assert semantic_domain(None,None,inside) is inside


def test_v3_uses_continuous_renderer_coordinates_and_strict_footprint_edges():
    row=dict(transform_matrix=np.eye(4),fl_x=1,fl_y=1,cx=0,cy=0,w=20,h=20)
    u=np.array([2,2.1,17,16.9,5,5,5]);v=np.array([5,5,5,5,2,17,5])
    points=np.column_stack((u+.5,-v-.5,-np.ones(len(u))))
    inside=np.array([True]*6+[False])
    assert footprint_domain(row,points,inside).tolist()==[False,True,False,True,False,False,False]
