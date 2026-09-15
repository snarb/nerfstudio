import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recover_original_surface_texture import selection


def test_only_unchanged_original_missing_texture_is_recovered():
    count=7
    old=np.ones((count,3),np.uint8)*100; new=np.zeros_like(old)
    os=np.zeros(count,np.uint8); ns=np.full(count,255,np.uint8)
    od=np.ones(count); d=od.copy(); oi=np.arange(count); i=oi.copy()
    ob=np.zeros((count,2)); b=ob.copy()
    i[1]=99          # added or different triangle, even with equal depth
    d[2]+=.001      # same face but a different surface intersection
    b[3,0]=.01      # different barycentric location
    ns[4]=0         # source exists, so black may be real observed color
    old[5]=0        # no trusted previous color
    new[6]=20       # do not overwrite an existing prediction
    mask=selection(old,new,os,ns,od,d,oi,i,ob,b,20)
    np.testing.assert_array_equal(mask,[True,False,False,False,False,False,False])


def test_no_backoff_without_original_geometry():
    mask=selection(np.ones((1,3),np.uint8),np.zeros((1,3),np.uint8),np.array([0]),np.array([255]),
        np.array([np.inf]),np.array([1.]),np.array([999]),np.array([0]),np.zeros((1,2)),np.zeros((1,2)),10)
    assert not mask.any()
