import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_train_gap_subfaces import selected_parents
from subdivide_conflicted_surface import subdivide,verify_coverage


def test_selection_is_query_evidence_not_parent_face_average():
    neg=np.array([[1,0,1,0],[1,0,1,0],[1,0,0,0],[0,0,0,0]],bool)
    samples=np.array([[0,1,2,3],[2,1,3,3]])
    np.testing.assert_array_equal(selected_parents(neg,samples),[True,False])


def test_two_rounds_keep_surface_and_neighbour_conformity():
    v=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
    t=np.array([[0,1,2],[0,2,3]]);v0,t0=v.copy(),t.copy();parents=np.arange(2)
    for _ in range(2):v,t,parents=subdivide(v,t,parents==0,parents)
    assert verify_coverage(v0,t0,v,t,parents)['per_parent_area_preserved']
    assert len(t)>2
