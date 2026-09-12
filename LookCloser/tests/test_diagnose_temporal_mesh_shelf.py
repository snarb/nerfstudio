import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_temporal_mesh_shelf import select_contained_faces


def test_selection_excludes_background_sentinel():
    ids=np.array([[0,0,2**32-1],[1,1,2**32-1]],np.uint32)
    selected,_=select_contained_faces(ids,[(0,0),(2,0),(2,1),(0,1)],2)
    assert selected.tolist()==[0,1]


def test_selection_requires_face_containment_not_just_one_pixel():
    ids=np.array([[0,0,0,0],[1,1,2,2]],np.uint32)
    selected,mask=select_contained_faces(ids,[(0,0),(1,0),(1,1),(0,1)],3,min_fraction=.9)
    assert selected.tolist()==[1]
    assert mask.sum()==4


def test_selection_does_not_select_unseen_face_ids():
    ids=np.zeros((3,3),np.uint32)
    selected,_=select_contained_faces(ids,[(0,0),(2,0),(2,2),(0,2)],10)
    assert selected.tolist()==[0]


def test_selection_includes_boundary_fraction():
    ids=np.zeros((2,4),np.uint32)
    selected,_=select_contained_faces(ids,[(0,0),(1,0),(1,1),(0,1)],1,min_fraction=.5)
    assert selected.tolist()==[0]
