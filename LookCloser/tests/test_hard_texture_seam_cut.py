from __future__ import annotations
import importlib.util
import itertools
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("maxflow")
spec=importlib.util.spec_from_file_location("seam_cut",Path(__file__).resolve().parents[1]/"scripts/hard_texture_seam_cut.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_binary_cut_matches_exhaustive_visible_minimum():
    rng=np.random.default_rng(12)
    rgb=rng.random((2,2,2,3)).astype(np.float32)
    valid=np.ones((2,2,2),bool)
    valid[0,0,0]=False
    valid[1,1,1]=False
    penalty=.003
    result,stats=module.optimize_source_labels(rgb,valid,rank_penalty=penalty)
    def energy(label):
        total=float((label*penalty).sum())
        for y,x,dy,dx in [(0,0,0,1),(1,0,0,1),(0,0,1,0),(0,1,1,0)]:
            a,b=label[y,x],label[y+dy,x+dx]
            total+=.5*(np.abs(rgb[a,y,x]-rgb[b,y,x]).mean()+np.abs(rgb[a,y+dy,x+dx]-rgb[b,y+dy,x+dx]).mean())+.01*(a!=b)
        return total
    possible=[]
    yy,xx=np.indices((2,2))
    for state in itertools.product(range(2),repeat=4):
        label=np.asarray(state).reshape(2,2)
        if valid[label,yy,xx].all():possible.append(energy(label))
    assert energy(result)==pytest.approx(min(possible),abs=1e-6)
    assert all(a>=b for a,b in zip(stats['energy'],stats['energy'][1:]))


def test_visibility_holes_and_hard_source_identity():
    rgb=np.zeros((2,4,7,3),np.float32)
    rgb[0]=[.2,.3,.4];rgb[1]=[.8,.5,.1]
    valid=np.zeros((2,4,7),bool)
    valid[0,:,1:3]=True;valid[1,:,4:6]=True
    result,_=module.optimize_source_labels(rgb,valid)
    assert (result[:,[0,3,6]]==-1).all()
    assert (result[:,1:3]==0).all()
    assert (result[:,4:6]==1).all()


def test_one_complete_source_replaces_artificial_seam():
    rgb=np.zeros((2,8,8,3),np.float32);rgb[1]=.1
    valid=np.ones((2,8,8),bool);valid[0,:,5:]=False
    result,_=module.optimize_source_labels(rgb,valid)
    assert (result==1).all()
