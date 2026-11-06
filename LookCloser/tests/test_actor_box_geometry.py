import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from actor_box_geometry import fit_floor_frame,anchor_components
from mask_visual_hull_aabb import MaskCamera,classify_cells


def test_floor_is_horizontal_and_rigid():
    points=np.array([[0,0,.2],[1,0,.3],[0,1,.4],[1,1,.5]])
    T,residual=fit_floor_frame(points)
    np.testing.assert_allclose(residual,0,atol=1e-12)
    np.testing.assert_allclose((points@T[:3,:3].T+T[:3,3])[:,2],0,atol=1e-12)
    np.testing.assert_allclose(T[:3,:3]@T[:3,:3].T,np.eye(3),atol=1e-12)
    with pytest.raises(ValueError,match='collinear'):
        fit_floor_frame([[0,0,0],[1,0,0],[2,0,0]])


def test_both_actors_and_detached_nearby_hair_retained_remote_alias_rejected():
    lo=np.array([[0,0,0],[1,0,0],[3,0,0],[10,0,0],[11,0,0],[100,0,0]],float)*.01
    hi=lo+.01
    anchors=np.array([[.004,.005,.005],[.014,.005,.005],[.104,.005,.005],[.114,.005,.005]])
    keep,report=anchor_components(lo,hi,anchors,nearby_margin=.025)
    assert keep.tolist()==[True,True,True,True,True,False]
    assert report['anchors_within_two_leaf_diagonals']==4
    one,_=anchor_components(lo,hi,anchors[:1],nearby_margin=.025)
    assert one.tolist()==[True,True,True,False,False,False]


def test_union_of_people_creates_cross_identity_ghost():
    a=np.zeros((101,101),bool);a[48:53,43:48]=True
    b=np.zeros_like(a);b[48:53,53:58]=True
    valid=np.ones_like(a);empty=np.zeros_like(a)
    first=np.eye(4);first[:3,3]=[0,0,100]
    second=np.eye(4);second[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]];second[:3,3]=[100,0,0]
    def views(fg,unknown):
        return [MaskCamera(str(i),p,1000,1000,50.5,50.5,fg,unknown,valid).prepare(0)
                for i,p in enumerate([first,second])]
    # This cell is person B in view one but person A in view two.
    lo=np.array([[.4999,-.0001,.4999]]);hi=lo+.0002
    assert classify_cells(views(a|b,empty),lo,hi)[0][0]==2
    assert classify_cells(views(a,b),lo,hi)[0][0]==1
    assert classify_cells(views(b,a),lo,hi)[0][0]==1
