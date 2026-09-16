import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from run_face_consensus_visibility import consensus_proposals


def test_old_semantic_veto_not_required_but_three_other_skin_witnesses_are():
    valid=np.array([[1],[1],[1],[1],[0]],bool)
    skin=np.array([[0],[1],[1],[1],[1]],bool)
    weights=np.array([[1],[1],[1],[1],[2]])
    proposed,votes,_=consensus_proposals(valid,skin,weights,np.array([0]))
    assert votes.tolist()==[3] and np.argwhere(proposed).tolist()==[[4,0]]
    skin[1]=False
    assert not consensus_proposals(valid,skin,weights,np.array([0]))[0].any()


def test_no_missing_source_or_invalid_old_visibility_or_nonbetter_candidate():
    valid=np.array([[1],[1],[1],[1],[0]],bool);skin=np.ones_like(valid)
    weights=np.array([[1],[1],[1],[1],[2]])
    for source in [-1,255,4]:
        assert not consensus_proposals(valid,skin,weights,np.array([source]))[0].any()
    weights[4]=1
    assert not consensus_proposals(valid,skin,weights,np.array([0]))[0].any()
