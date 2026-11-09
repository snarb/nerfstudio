import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_marshawn_actor_geometry import slab
from prepare_marshawn_actor_semantic_absence import protected_cells

def test_slab_parallel_and_boundary_not_silently_free():
 box=np.array([[-1.,-1.,-1.],[1.,1.,1.]])
 O=np.array([[0,0,3],[2,0,3],[1,0,3],[0,0,0.]])
 D=np.tile([0.,0.,-1.],(4,1));lo,hi,bad=slab(O,D,box)
 assert np.allclose(lo[[0,2]],[2,2]) and np.allclose(hi[[0,2]],[4,4])
 assert bad.tolist()==[False,True,False,False]
 assert lo[3]==-1 and hi[3]==1

def test_projected_whole_cell_protects_offcentre_pixel():
 row=dict(h=100,w=100,fl_x=80,fl_y=80,cx=50,cy=50,transform_matrix=np.eye(4).tolist())
 xyz=np.array([[0,0,-1.]])
 box=np.array([[-16.,-16.,-16.],[16.,16.,16.]])
 mask=protected_cells(row,xyz,box)
 # Centre alone would not protect this pixel; actual cell +/-0.125 and
 # conservative raster margin must cover it.
 assert mask[50,60] and mask[50,50] and not mask[0,0]
