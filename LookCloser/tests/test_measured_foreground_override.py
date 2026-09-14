import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_measured_foreground_override import add_certified_seeds


def test_no_observed_seeds_cannot_change_mask():
    m=np.zeros((40,40),bool);m[:,10:20]=True
    np.testing.assert_array_equal(add_certified_seeds(m,np.zeros_like(m)),m)


def test_additions_preserve_original_and_cannot_escape_band():
    m=np.zeros((60,60),bool);m[:,10:20]=True;s=np.zeros_like(m);s[30,42]=True
    r=add_certified_seeds(m,s)
    assert r[m].all() and r[30,42] and not r[:,44:].any()
    assert not r[:25,20:].any() and not r[35:,20:].any()


def test_unbounded_seed_is_rejected():
    m=np.zeros((60,60),bool);m[:,10:20]=True;s=np.zeros_like(m);s[30,50]=True
    with pytest.raises(ValueError,match='Seed outside'):add_certified_seeds(m,s)
