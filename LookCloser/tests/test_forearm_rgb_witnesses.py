import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from forearm_rgb_witnesses import color_errors,qualified_errors
from test_forearm_color_witnesses import setup_scene


def test_same_chroma_different_brightness_is_not_equivalent_rgb_evidence():
    rows,depths,images=setup_scene()
    for n in ['1','2','3']:images[n]=np.broadcast_to(np.array([240,160,80],np.uint8),(1080,1920,3))
    points=np.array([[0,0,-1.]])
    chroma,rgb=color_errors(points,rows[0],rows,depths,images)
    assert (chroma<=.04).sum()==3 and (rgb>.1).sum()==3
    assert not np.isfinite(qualified_errors(points,rows[0],rows,depths,images,.08)).any()


def test_identical_rgb_still_requires_geometric_witnesses():
    rows,depths,images=setup_scene();points=np.array([[0,0,-1.]])
    assert np.isfinite(qualified_errors(points,rows[0],rows,depths,images,.08)).sum()==3
    depths[3][:]=0
    assert np.isfinite(qualified_errors(points,rows[0],rows,depths,images,.08)).sum()==2


def test_rgb_limit_cannot_silently_disable_depth_color_guard(tmp_path):
    import pytest
    from study_forearm_production_delta import prepare
    for kwargs in [dict(witness_rgb_limit=.12),dict(photometric_free_space=True,witness_rgb_limit=0),
                   dict(photometric_free_space=True,witness_rgb_limit=float('nan'))]:
        with pytest.raises(ValueError,match='RGB witness limit requires'):
            prepare(tmp_path/'not_created','001037',**kwargs)
    assert not (tmp_path/'not_created').exists()
