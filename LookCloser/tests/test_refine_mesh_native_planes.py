import numpy as np
import pytest
import torch
from audit_native_plane_displacement import fit_native_plane, normal_displacement, displacement_consensus
from refine_mesh_native_planes import native_plane_votes, consensus_votes, smooth_supported_displacements, damp_noninverting


def test_vectorized_native_plane_matches_scalar_reference():
    camera=dict(fl_x=900.,fl_y=950.,cx=25.,cy=25.,transform_matrix=np.eye(4).tolist())
    yy,xx=np.mgrid[:50,:50];normal_cv=np.array([.2,-.1,1.])
    rays=np.stack(((xx+.5-25)/900,(yy+.5-25)/950,np.ones_like(xx)),-1)
    depth=.7/(rays@normal_cv)
    normal=normal_cv*[1,-1,-1];normal/=np.linalg.norm(normal)
    pixels=[(20,23),(27,28),(32,17)]
    points=np.array([rays[y,x]*depth[y,x]*[1,-1,-1]+.001*normal for x,y in pixels])
    normals=np.tile(normal,(3,1))
    got=native_plane_votes(torch.tensor(depth),torch.tensor(points),torch.tensor(normals),camera).numpy()
    expected=[]
    for point in points:
        z=-point[2];x=int(np.rint(900*point[0]/z+25-.5));y=int(np.rint(-950*point[1]/z+25-.5))
        plane=fit_native_plane(depth[y-2:y+3,x-2:x+3],[x,y],camera)
        expected.append(normal_displacement(plane,point,normal)['displacement'])
    np.testing.assert_allclose(got,expected,atol=1e-11)


def test_invalid_depths_do_not_vote():
    camera=dict(fl_x=900.,fl_y=950.,cx=25.,cy=25.,transform_matrix=np.eye(4).tolist())
    points=torch.tensor([[0,0,-.7]],dtype=torch.float64);normal=torch.tensor([[0,0,-1.]],dtype=torch.float64)
    for value in [0.,float('nan'),-1.,float('inf')]:
        assert torch.isnan(native_plane_votes(torch.full((50,50),value),points,normal,camera)).all()


def test_vector_consensus_matches_scalar_missing_and_disjoint():
    votes=torch.tensor([[.002,.001,float('nan')],[.0021,.0011,float('nan')],
        [.0019,-.002,float('nan')],[-.001,-.0021,float('nan')]],dtype=torch.float64)
    target,eligible,count,total=consensus_votes(votes)
    for j in range(3):
        raw=votes[:,j].numpy();scalar=displacement_consensus(raw[np.isfinite(raw)])
        assert bool(eligible[j])==scalar['eligible']
        assert int(count[j])==scalar['agreeing_views']
        if scalar['eligible']:
            assert float(target[j])==pytest.approx(scalar['displacement'])
        else:
            assert float(target[j])==0


def test_smoothing_fixes_unsupported_vertices():
    v=np.array([[0,0,0],[.001,0,0],[0,.001,0],[.001,.001,0]])
    t=np.array([[0,1,2],[1,3,2]])
    delta,stats=smooth_supported_displacements(v,t,np.array([.001,.002,0,0]),np.array([True,True,False,False]),np.array([8,8,0,0]))
    assert stats['converged']
    np.testing.assert_array_equal(delta[2:],[0,0])
    assert 0<delta[0]<.002 and 0<delta[1]<.002


def test_topology_guard_damps_inversion_without_deletion():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    t=np.array([[0,1,2]]);offset=np.array([[0.,0,0],[-2,0,0],[0,0,0]])
    got,stats=damp_noninverting(v,t,offset)
    assert len(got)==3 and stats['damped_vertices']==3
    assert np.cross(got[1]-got[0],got[2]-got[0])[2]>0


def test_zero_step_and_rigid_translation_preserved():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);t=np.array([[0,1,2]])
    for offset in [np.zeros_like(v),np.tile([.2,.1,-.3],(3,1))]:
        got,stats=damp_noninverting(v,t,offset)
        np.testing.assert_array_equal(got,v+offset)
        assert stats['damped_vertices']==0
