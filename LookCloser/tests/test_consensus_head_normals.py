from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_consensus_head_normals import consensus


def test_oriented_consensus_not_single_flipped_facet():
    centers=np.zeros((32,3));centers[:,0]=np.linspace(-.001,.001,32)
    normals=np.tile([0.,0.,1.],(32,1));normals[15]=[0,0,-1]
    areas=np.ones(32);areas[15]=10000
    good,coherence,dot,agreement,count=consensus(np.zeros((1,3)),np.array([[0,0,1.]]),centers,normals,areas)
    assert good[0] and coherence[0]>.8 and agreement[0]>.9 and count[0]==32
    assert not consensus(np.zeros((1,3)),np.array([[0,0,-1.]]),centers,normals,areas)[0][0]


def test_ambiguous_or_distant_normals_fail_closed():
    centers=np.zeros((32,3));normals=np.tile([0.,0.,1.],(32,1));normals[:16]*=-1
    assert not consensus(np.zeros((1,3)),np.array([[0,0,1.]]),centers,normals,np.ones(32))[0][0]
    normals[:16]*=-1
    assert not consensus(np.array([[1.,0,0]]),np.array([[0,0,1.]]),centers,normals,np.ones(32))[0][0]
