import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ordered_forearm_admission import point_votes
from study_confidence_depth_prior import unproject
from test_forearm_color_witnesses import setup_scene


def test_geometry_is_evaluated_before_semantic_admission():
    rows,depths,_=setup_scene();rows=rows[:3];depths=depths[:3];names=[r['physical_camera'] for r in rows]
    masks={n:np.ones((1080,1920),bool) for n in names};masks[names[1]][:,46:]=False
    data={n+'_trusted':np.zeros((1080,1920),bool) for n in names}
    identity=lambda row,points,inside:inside
    plane=unproject(rows[0],np.array([50]),np.array([50]),np.array([1.]))
    curve=unproject(rows[0],np.array([50]),np.array([50]),np.array([.8]))
    a=point_votes(plane,rows,names,masks,data,depths,identity)
    b=point_votes(curve,rows,names,masks,data,depths,identity)
    assert a[1][0]==1 and b[1][0]==0 and b[0][0]==3


def test_admission_option_requires_the_same_boundary_curvature_workflow(tmp_path):
    from study_forearm_production_delta import prepare
    with pytest.raises(ValueError,match='Admission order requires'):
        prepare(tmp_path/'not_created','001037',admission_shape='quadric')
    assert not (tmp_path/'not_created').exists()
