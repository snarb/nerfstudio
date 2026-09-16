import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from study_face_interior_visibility import proposals


def test_new_source_requires_consensus_and_better_quality():
    valid=np.array([[1,1,1],[1,1,0],[1,1,0],[0,0,0]],bool)
    skin=np.ones((4,3),bool);weights=np.array([[1,1,1],[1,1,1],[1,1,1],[2,.9,2]])
    candidates,votes,old=proposals(valid,skin,weights,np.zeros(3,np.uint8))
    assert np.argwhere(candidates).tolist()==[[3,0]]
    assert votes.tolist()==[3,3,1]
    skin[0,0]=False
    assert not proposals(valid,skin,weights,np.zeros(3,np.uint8))[0].any()


def test_missing_old_source_cannot_be_painted():
    valid=np.array([[1],[1],[1],[0]],bool);skin=np.ones_like(valid)
    weights=np.array([[1],[1],[1],[2]])
    assert not proposals(valid,skin,weights,np.array([255],np.uint8))[0].any()
