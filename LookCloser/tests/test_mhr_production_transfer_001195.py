import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_mhr_production_transfer_001195 as run
from review_mhr_production_patch_control import check_recipe
from localize_mhr_production_patch_side_effects import side_effect_masks


def test_actual_frame_and_distinct_base_binding():
    p=run.binding();assert p['frame']=='001195'
    assert p['production_mesh_sha256']!=p['raw_mesh_sha256']
    assert p['normalization_metadata_identical'] and not p['fresh_fit_performed']
    assert p['candidate_math_unchanged'] and p['admission_math_unchanged']


def test_policy_is_current_and_fails_old_power8():
    p=run.read(run.PARENT/'request.json');check_recipe(p)
    old=dict(p,recipe=dict(p['recipe'],source_incidence_power=8))
    with pytest.raises(AssertionError):check_recipe(old)


def test_black_and_lost_depth_are_separate():
    old=np.ones((1,3,3),np.uint8)*10;new=old.copy();new[0,:2]=0
    a=np.array([[0.,1.,1.]]);b=np.array([[1.,1.,0.]])
    result=side_effect_masks(old,a,new,b)
    np.testing.assert_array_equal(result['new_geometry_without_rgb'],[[True,False,False]])
    np.testing.assert_array_equal(result['newly_black_rgb'],[[True,True,False]])
    np.testing.assert_array_equal(result['lost_geometry'],[[False,False,True]])
