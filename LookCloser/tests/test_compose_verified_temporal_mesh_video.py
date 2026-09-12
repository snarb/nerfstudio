from copy import deepcopy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compose_verified_temporal_mesh_video import assert_texture_equivalent


def parent():
    return {'calibration_sha256':'c','profiles_sha256':'p','exposure_sha256':'e','uses_heldout_rgb':False,
            'recipe':{'frames':150,'fixed_profiles':True,'averages_rgb':False,'target_angle_sigma_degrees':4.},
            'script_hashes':{'renderer.py':'same'}}


def test_geometry_control_subset_preserves_texture_recipe():
    a=parent();b=deepcopy(a);b['recipe'].update(frames=1,geometry_control_variant='full-block')
    b['script_hashes']['control.py']='extra'
    assert_texture_equivalent(a,b)


@pytest.mark.parametrize('key',['calibration_sha256','profiles_sha256','exposure_sha256','uses_heldout_rgb'])
def test_changed_input_cannot_be_claimed_equivalent(key):
    a=parent();b=deepcopy(a);b[key]='changed'
    with pytest.raises(ValueError):assert_texture_equivalent(a,b)


def test_source_prior_cannot_silently_change():
    a=parent();b=deepcopy(a);b['recipe']['target_angle_sigma_degrees']=6.
    with pytest.raises(ValueError):assert_texture_equivalent(a,b)


def test_renderer_code_cannot_silently_change():
    a=parent();b=deepcopy(a);b['script_hashes']['renderer.py']='different'
    with pytest.raises(ValueError):assert_texture_equivalent(a,b)
