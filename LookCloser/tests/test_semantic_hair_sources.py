import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_semantic_hair_sources import select_hair_sources


def inputs():
    q=torch.tensor([[10.,10.,10.,10.],[5.,5.,5.,5.],[3.,3.,3.,3.],[1.,1.,1.,1.]])
    visible=torch.ones((4,4),dtype=torch.bool);known=visible.clone()
    sem=torch.zeros((4,3,4));sem[:,:2]=1
    margin=torch.full((4,4),25.);margin[0]=3
    baseline=torch.zeros(4,dtype=torch.long)
    return q,visible,sem,known,margin,baseline


def test_consensus_switch_skin_veto_and_unknown_preservation():
    q,v,s,k,m,b=inputs();s[:3,2,1]=1;k[2:,2]=False;m[:,3]=3
    selected,gate,_,_=select_hair_sources(q,v,s,k,m,b)
    assert selected.tolist()==[1,0,0,0]
    assert gate.tolist()==[True,False,False,True]


def test_never_chooses_occluded_or_creates_new_coverage():
    q,v,s,k,m,b=inputs();v[1,0]=False;q[1,0]=0;b[1]=255;m[0,2]=30
    selected,_,_,_=select_hair_sources(q,v,s,k,m,b)
    assert selected.tolist()==[2,255,0,1]


def test_requires_both_models_and_finite_evidence():
    q,v,s,k,m,b=inputs();s[:,1,0]=0
    selected,gate,_,_=select_hair_sources(q,v,s,k,m,b)
    assert not gate[0] and selected[0]==0
    s[0,0,0]=float('nan')
    with pytest.raises(ValueError):select_hair_sources(q,v,s,k,m,b)
