import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from guard_poisson_jaw_completion import anchored_admission


def test_inference_requires_every_anchor_and_preserves_vetoes():
    strict=np.zeros(5,bool);anchors=np.full((5,3),3);anchors[1,2]=2
    free=np.zeros((62,5,10),bool);free[7,2,3]=True
    support=np.full(5,62);support[3]=1;outside=np.zeros(5,int);outside[4]=1
    np.testing.assert_array_equal(anchored_admission(strict,anchors,free,support,outside),[True,False,False,False,False])


def test_existing_strict_admission_is_not_removed():
    result=anchored_admission(np.array([True]),np.zeros((1,3),int),np.zeros((62,1,10),bool),[62],[0])
    assert result[0]
