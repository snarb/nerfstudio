import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from review_foundation_hand_geometry import grid_triangles


def test_grid_omits_invalid_vertices_and_large_depth_jumps():
    y,x=np.indices((3,3));p=np.stack([x*.0001,y*.0001,np.ones_like(x)],-1)
    valid=np.ones((3,3),bool)
    v,t=grid_triangles(p,valid);assert len(t)==8 and len(v)==9
    valid[1,1]=False;v,t=grid_triangles(p,valid);assert len(v)==8 and len(t)==2
    valid[:]=True;p[1,1,2]=2
    v,t=grid_triangles(p,valid);assert len(t)==2
