import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_jaw_repair_transfer import mask_votes
from test_study_jaw_depth_footprint import inputs


def test_mask_gate_preserves_available_view_disagreement():
    rows,_=inputs();names=[r['physical_camera'] for r in rows]
    masks=np.ones((3,1080,1920),bool)
    v=np.array([[0,0,-2],[.01,0,-2],[0,.01,-2.]])
    triangles=np.array([[0,1,2]])
    support,outside=mask_votes(v,triangles,rows,masks,names)
    assert support.tolist()==[3] and outside.tolist()==[0]
    masks[1]=False
    support,outside=mask_votes(v,triangles,rows,masks,names)
    assert support.tolist()==[2] and outside.tolist()==[1]


def test_unavailable_mask_is_not_negative_evidence():
    rows,_=inputs();names=[r['physical_camera'] for r in rows]
    rows[2]['cx']=-1000
    masks=np.ones((3,1080,1920),bool)
    v=np.array([[0,0,-2],[.01,0,-2],[0,.01,-2.]])
    support,outside=mask_votes(v,np.array([[0,1,2]]),rows,masks,names)
    assert support.tolist()==[2] and outside.tolist()==[0]
