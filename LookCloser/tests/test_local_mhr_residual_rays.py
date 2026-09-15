from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_local_mhr_residual_rays import landscape_pixel,first_absent_stage
from audit_local_mhr_residual_rays import intersections


def test_portrait_rotation_exact_indices():
    a=np.arange(1080*1920).reshape(1080,1920)
    for x,y in [(145,825),(176,1343),(1032,1428),(0,0),(1079,1919)]:
        u,v=landscape_pixel(x,y);assert np.rot90(a)[y,x]==a[v,u]


def test_triangle_intersection_two_sided_and_miss():
    v=np.array([[0,0,1],[1,0,1],[0,1,1]],float)
    for t in [np.array([[0,1,2]]),np.array([[2,1,0]])]:
        ids,d=intersections([.25,.25,0,0,0,2],v,t)
        assert ids.tolist()==[0];np.testing.assert_array_equal(d,[.5])
        assert len(intersections([2,2,0,0,0,1],v,t)[0])==0
        assert len(intersections([.25,.25,2,0,0,1],v,t)[0])==0


def test_bottleneck_does_not_blame_admission_for_no_prior():
    names=['prior_full','anatomical_band','safe_band','local_before_centroid','raw_candidates','semantic_candidates']
    assert first_absent_stage(dict.fromkeys(names,False))=='prior_full'
    hits=dict.fromkeys(names,True);hits['semantic_candidates']=False
    assert first_absent_stage(hits)=='semantic_candidates'
