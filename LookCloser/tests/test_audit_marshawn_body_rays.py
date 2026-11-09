import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_marshawn_body_rays import interval_mass,select_balanced


def test_band_mass_matches_constant_medium_and_partial_bin():
    mass=interval_mass([0,1,2],[1,2,3],[2,2,2],.25,1.75)
    assert mass==pytest.approx(np.exp(-.5)-np.exp(-3.5))
    assert interval_mass([0,1],[1,2],[0,0],0,2)==0
    assert interval_mass([],[],[],0,1)==0


def test_band_excludes_density_before_and_after_surface():
    assert interval_mass([0,1,2],[1,2,3],[1000,1,1000],1,2)==0
    assert interval_mass([0,1,2],[1,2,3],[0,1,1000],1,2)==pytest.approx(1-np.exp(-1))


def test_selection_balanced_unique_body_only_and_private_rng():
    points=np.arange(80)%8;camera=np.repeat(np.arange(5),16);before=np.random.get_state();a=select_balanced(points,camera,[1,2,3],12,42);after=np.random.get_state();b=select_balanced(points,camera,[1,2,3],12,42)
    assert len(np.unique(a))==12 and np.isin(points[a],[1,2,3]).all() and np.array_equal(a,b)
    counts=np.unique(camera[a],return_counts=True)[1];assert counts.max()-counts.min()<=1
    assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    with pytest.raises(ValueError):select_balanced(points,camera,[1,2,3],4,42)
