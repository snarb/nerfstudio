import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from transfer_original_surface_backoff import audit_splice


def arrays():
    base=np.full((2,3,3),80,np.uint8);raw=base.copy();raw[0,1]=0
    fixed=base.copy();b=np.zeros((2,3),np.uint8);r=b.copy();r[0,1]=255
    mask=np.zeros((2,3),bool);mask[0,1]=True
    return [base,raw,fixed,b,r,b.copy(),mask]


def test_exact_restore():
    assert audit_splice(*arrays())==dict(recovered=1,new_black_before=1,new_black_after=0)


@pytest.mark.parametrize('kind',['outside_rgb','inside_rgb','outside_source','inside_source','colored_raw','invalid_baseline'])
def test_reject_unjustified_changes(kind):
    a=arrays()
    if kind=='outside_rgb':a[2][1,1]=81
    if kind=='inside_rgb':a[2][0,1]=79
    if kind=='outside_source':a[5][1,1]=1
    if kind=='inside_source':a[5][0,1]=1
    if kind=='colored_raw':a[1][0,1]=1
    if kind=='invalid_baseline':a[3][0,1]=255
    with pytest.raises(AssertionError):audit_splice(*a)
