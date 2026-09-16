import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from study_face_angular_visibility import angular_proposals,transformed_render


def test_raster_control_cannot_admit_new_depth_footprint():
    valid=np.array([[1],[1],[1],[0]],bool);skin=np.ones_like(valid)
    w=np.array([[.2],[.1],[.5],[1.]])
    assert np.flatnonzero(angular_proposals(valid,skin,w,np.array([0]),'raster')[0]).tolist()==[2]
    assert np.flatnonzero(angular_proposals(valid,skin,w,np.array([0]),'consensus')[0]).tolist()==[2,3]


def test_both_controls_require_old_visibility_and_three_skin_witnesses():
    valid=np.array([[1],[1],[1],[0]],bool);skin=np.ones_like(valid);w=np.array([[.2],[.1],[.5],[1.]])
    for mode in ['raster','consensus']:
        for old in [-1,255,3]:assert not angular_proposals(valid,skin,w,np.array([old]),mode)[0].any()
        skin[1]=False
        assert not angular_proposals(valid,skin,w,np.array([0]),mode)[0].any()
        skin[1]=True


def test_only_quality_statement_changes_in_render_adapter():
    source=transformed_render()
    assert 'weights=np.broadcast_to(angle[:,None],length.shape)' in source
    assert 'scene.cast_rays' in source and 'sample_native' in source
