import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from guard_mhr_anatomical_correction import anatomical_domain,AllContactGuard
from fit_mhr_constrained_correction import ConstraintGuard


def test_coplanar_overlap_is_rejected():
    v=np.array([[0.,0,0],[1.,0,0],[0.,1,0],[2.,0,0],[3.,0,0],[2.,1,0]])*.0002
    t=np.array([[0,1,2],[3,4,5]])
    trial=v.copy();trial[3:,0]-=.0003
    assert ConstraintGuard(v,t,v).check(trial)[0]
    ok,record=AllContactGuard(v,t,v).check(trial)
    assert not ok and record['reason']=='new_contact_or_overlap'
    assert record['new_all_intersection_pairs']==1


def test_domain_excludes_shoulders_not_neck():
    n=np.array([[0,150,8],[-2,151,9],[20,144,3],[0,130,0],[0,160,0],[12,150,0]])
    np.testing.assert_array_equal(anatomical_domain(n),[True,True,False,False,False,False])


def test_invalid_domain_coordinates_rejected():
    with pytest.raises(ValueError):anatomical_domain(np.array([[np.nan,150,0]]))
