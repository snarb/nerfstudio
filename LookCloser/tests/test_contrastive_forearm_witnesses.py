import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from contrastive_forearm_witnesses import comparison_votes,carving_decision
from study_confidence_depth_prior import project_integer
from test_forearm_color_witnesses import setup_scene


def test_flat_patch_is_not_decisive_free_space_evidence():
    rows,depths,images=setup_scene()
    r=comparison_votes(np.array([[0,0,-1.]]),np.array([[0,0,-.8]]),rows[0],rows,depths,images)
    assert r['rgb_qualified'][0]==3 and r['comparable'][0]==3 and r['decisive'][0]==0


def test_wrong_alternative_patch_is_distinguished_by_three_other_cameras():
    rows,depths,images=setup_scene();observed=np.array([[0,0,-1.]]);proposed=np.array([[0,0,-.8]])
    for row in rows[1:]:
        old=np.rint(project_integer(row,observed)[0][0]).astype(int)
        new=np.rint(project_integer(row,proposed)[0][0]).astype(int)
        image=images[row['physical_camera']].copy()
        for y in range(new[1]-2,new[1]+3):
            for x in range(new[0]-2,new[0]+3):
                if abs(x-old[0])>2 or abs(y-old[1])>2:image[y,x]=255
        images[row['physical_camera']]=image
    r=comparison_votes(observed,proposed,rows[0],rows,depths,images)
    assert r['rgb_qualified'][0]==3 and r['decisive'][0]==3


def test_missing_alternative_visibility_is_not_evidence_for_abstention():
    votes=dict(rgb_qualified=np.array([3,3,2,4]),comparable=np.array([3,2,2,4]),decisive=np.array([0,2,2,3]))
    assert carving_decision(votes).tolist()==[False,True,False,True]


def test_comparison_option_requires_explicit_compatible_rgb_guard(tmp_path):
    import pytest
    from study_forearm_production_delta import prepare
    for kwargs in [dict(witness_comparison_margin=.01),
                   dict(photometric_free_space=True,witness_rgb_limit=.12,witness_comparison_margin=float('nan'))]:
        with pytest.raises(ValueError,match='Comparison requires'):
            prepare(tmp_path/'not_created','001037',**kwargs)
    assert not (tmp_path/'not_created').exists()
