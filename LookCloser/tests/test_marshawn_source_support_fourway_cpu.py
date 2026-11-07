import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from audit_marshawn_source_support_fourway_cpu import ray_union
from marshawn_source_margin_admission import GeometricSourcePointAdmission
from marshawn_source_point_admission import SourcePointAdmission

def test_parallel_face_ray_and_outside():
    bounds=np.array([[0.,0.,0.],[1.,1.,1.]])
    lo=np.array([[0.,.2,.2]]);hi=np.array([[.5,.4,.4]])
    assert np.allclose(ray_union([0.,0.,.3],[0.,1.,0.],lo,hi,bounds),[[.2,.4]])
    assert ray_union([-1.,0.,.3],[0.,1.,0.],lo,hi,bounds)==[]

def test_intervals_match_point_lookup_and_margin_is_monotone():
    bounds=np.array([[0.,0.,0.],[.08,.08,.08]])
    mask=np.zeros((8,8,8),bool);mask[2:5,3:5,2:6]=True
    ids=np.argwhere(mask);o=np.array([0.,.036,.037]);d=np.array([1.,0.,0.]);t=np.linspace(.00001,.07999,10001)
    previous=np.zeros(len(t),bool)
    for margin in [0.,.0005,.001,.002]:
        lookup=GeometricSourcePointAdmission(torch.tensor(mask),bounds,margin=margin,base_class=SourcePointAdmission)
        keep=lookup(torch.tensor(o+t[:,None]*d)).numpy()
        intervals=ray_union(o,d,ids*.01-margin-2e-7,(ids+1)*.01+margin+2e-7,bounds)
        analytic=np.zeros(len(t),bool)
        for a,b in intervals:analytic|=(t>=a)&(t<=b)
        assert np.array_equal(keep,analytic)
        assert not (previous&~keep).any()
        previous=keep

def test_outside_unknown_is_not_positive_ray_support():
    bounds=np.array([[0.,0.,0.],[.08,.08,.08]])
    mask=np.zeros((8,8,8),bool)
    lookup=GeometricSourcePointAdmission(torch.tensor(mask),bounds,margin=.002,base_class=SourcePointAdmission)
    assert lookup(torch.tensor([[-.003,.04,.04]])).item()
    assert ray_union([-.003,0.,.04],[0.,1.,0.],np.empty((0,3)),np.empty((0,3)),bounds)==[]
