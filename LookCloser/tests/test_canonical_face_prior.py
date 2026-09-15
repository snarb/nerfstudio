import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_canonical_face_prior import similarity,deform,smooth_basis,CORE,JAW
from probe_canonical_face_locality import boundary_edges,edge_distance

def test_similarity_recovers_metric_pose():
    x=np.random.default_rng(42).normal(size=(50,3));r=Rotation.from_rotvec([.4,-.3,.2]).as_matrix();t=np.array([.1,3,-2]);y=.035*x@r.T+t
    scale,rotation,translation=similarity(x,y)
    np.testing.assert_allclose(scale,.035,atol=1e-12);np.testing.assert_allclose(rotation,r,atol=1e-12);np.testing.assert_allclose(translation,t,atol=1e-12)
    assert np.linalg.det(rotation)>0

def test_zero_shape_is_similarity_and_does_not_mutate_vertices():
    v=np.random.default_rng(3).normal(size=(30,3));original=v.copy();b=np.ones((30,8));p=np.r_[np.zeros(3),[1,2,3],np.log(2)]
    np.testing.assert_allclose(deform(p,v,b),2*v+[1,2,3]);np.testing.assert_allclose(deform(np.r_[p,np.zeros(24)],v,b),2*v+[1,2,3]);np.testing.assert_array_equal(v,original)

def test_basis_excludes_affine_shape_and_is_bounded():
    n=10;xx,yy=np.meshgrid(np.arange(n),np.arange(n));v=np.column_stack((xx.ravel(),yy.ravel(),np.sin(xx.ravel())*.1));t=[]
    for y in range(n-1):
        for x in range(n-1):
            i=y*n+x;t.extend([[i,i+1,i+n],[i+1,i+n+1,i+n]])
    b,e=smooth_basis(v,np.array(t));assert b.shape==(100,8);assert (np.diff(e)>=0).all()
    np.testing.assert_allclose(np.column_stack((np.ones(len(v)),v)).T@b,0,atol=1e-10);np.testing.assert_allclose(np.max(np.abs(b),axis=0),1)

def test_jaw_contour_not_fixed_correspondence():
    assert not (set(CORE)&set(JAW))

def test_only_open_topology_edges_are_boundaries():
    edges=boundary_edges(np.array([[0,1,2],[0,2,3]]))
    assert set(map(tuple,edges))=={(0,1),(1,2),(2,3),(0,3)}

def test_exact_edge_distance_includes_segment_interior():
    v=np.array([[0.,0,0],[2,0,0]])
    np.testing.assert_allclose(edge_distance(np.array([[1.,1,0],[3,0,0],[1,0,0]]),v,np.array([[0,1]])),[1,1,0])
