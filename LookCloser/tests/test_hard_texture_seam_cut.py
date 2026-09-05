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


@pytest.mark.parametrize('confidence', [np.ones((2,2),np.float32), np.array([[0,.1],[1,.5]],np.float32)])
@pytest.mark.parametrize('extra', [np.zeros((2,2,2),np.float32), np.array([[[.1,.3],[.2,.4]],[[.7,0],[.5,.2]]],np.float32)])
def test_binary_cut_matches_exhaustive_visible_minimum(confidence,extra):
    rng=np.random.default_rng(12)
    rgb=rng.random((2,2,2,3)).astype(np.float32)
    valid=np.ones((2,2,2),bool)
    valid[0,0,0]=False
    valid[1,1,1]=False
    penalty=.003
    result,stats=module.optimize_source_labels(rgb,valid,rank_penalty=penalty,rank_confidence=confidence,source_costs=extra)
    def energy(label):
        yy,xx=np.indices((2,2))
        total=float((label*penalty*confidence+extra[label,yy,xx]).sum())
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


def test_visibility_rank_confidence_preserves_other_depth_layers_and_defaults():
    valid=np.ones((2,60,60),bool)
    valid[0,20:30,20:30]=False
    depth=np.ones((60,60),np.float32)
    depth[20:30,30:35]=.8
    confidence=module.visibility_rank_confidence(valid,depth,10)
    assert confidence[25,25]==0
    assert confidence[25,19]==pytest.approx(.01)
    assert confidence[25,32]==1  # nearby foreground is not the disoccluded layer
    assert confidence[0,0]==1
    assert (module.visibility_rank_confidence(valid,depth,0)==1).all()
    assert (module.visibility_rank_confidence(np.ones_like(valid),depth,10)==1).all()


def test_rank_confidence_ones_preserves_legacy_labels():
    rng=np.random.default_rng(13)
    rgb=rng.random((3,4,5,3)).astype(np.float32)
    valid=rng.random((3,4,5))>.3
    legacy,_=module.optimize_source_labels(rgb,valid)
    explicit,_=module.optimize_source_labels(rgb,valid,rank_confidence=np.ones((4,5),np.float32))
    np.testing.assert_array_equal(legacy,explicit)


@pytest.mark.parametrize('confidence', [np.ones((3,3)), np.full((4,5),np.nan), -np.ones((4,5))])
def test_invalid_rank_confidence_rejected(confidence):
    with pytest.raises(ValueError,match='confidence'):
        module.optimize_source_labels(np.zeros((2,4,5,3)),np.ones((2,4,5),bool),rank_confidence=confidence)


def test_consensus_penalizes_outlier_without_changing_primary_or_rgb():
    rgb=np.full((4,8,8,3),.4,np.float32);rgb[1]=.8;rgb[3]=.41
    valid=np.ones((4,8,8),bool);original=rgb.copy()
    costs=module.consensus_source_costs(rgb,valid)
    assert (costs[0]==0).all() and costs[1].min()>.3 and costs[2].max()==0
    np.testing.assert_array_equal(rgb,original)
    valid[2:]=False
    assert (module.consensus_source_costs(rgb,valid)==0).all()
