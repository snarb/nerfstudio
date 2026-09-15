from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prune_multiview_background_head import triangle_samples,deletion_rule


def test_all_seven_samples_and_original_coordinates():
    v=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
    p=triangle_samples(v,np.array([[0,1,2]]))[0]
    np.testing.assert_array_equal(p[:3],v)
    np.testing.assert_array_equal(p[3:6],[[1,0,0],[1,1,0],[0,1,0]])
    np.testing.assert_allclose(p[6],[2/3,2/3,0])


def test_one_supported_sample_protects_entire_triangle():
    votes=np.zeros((4,7),int);votes[1,3]=2;votes[2,6]=1
    np.testing.assert_array_equal(deletion_rule([12,50,12,11],votes),[True,False,True,False])
    with pytest.raises(ValueError):deletion_rule([20],np.zeros((1,4)))


def test_nonfinite_evidence_and_configured_protection():
    with pytest.raises(ValueError):deletion_rule([20],np.full((1,7),np.nan))
    with pytest.raises(ValueError):deletion_rule([20],np.zeros((1,7)),minimum_background=0)
    assert not deletion_rule([20],np.ones((1,7)),minimum_preserving_votes=1)[0]


def test_clear_background_requires_all_samples_and_valid_frustum(monkeypatch):
    import prune_multiview_background_head as p
    rows=[dict(physical_camera='a'),dict(physical_camera='b')]
    masks=np.zeros((2,30,30),bool);masks[1,15,15]=True
    uv=np.full((1,14,2),15.);z=np.ones((1,14));uv[0,7]=np.nan
    monkeypatch.setattr(p,'project',lambda points,rows:(uv,z))
    votes=p.clear_background_votes(np.zeros((2,7,3)),rows,masks,['a','b'])
    np.testing.assert_array_equal(votes,[[True,False],[False,False]])
