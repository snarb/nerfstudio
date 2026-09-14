import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_early_texture_prior import early_quality,transform_source


def test_early_prior_preserves_visible_target_that_incidence_alone_would_discard():
    quality=np.array([[.05,0.],[1.,1.]])
    weights=np.array([1.,.001])
    old=np.where(quality>=quality.max(0)*.12,quality,0)*weights[:,None]
    new=early_quality(quality,weights)
    assert old[0,0]==0 and new[0,0]>.0 and new.argmax(0)[0]==0
    assert new[0,1]==0  # invisible sources cannot be resurrected


def test_runtime_transform_is_exact_and_refuses_changed_renderer():
    old='quality=np.where(quality>=quality.max(0)*.12,quality,0)'
    source='prefix\n'+old+'\nsuffix'
    changed=transform_source(source)
    assert changed.startswith('prefix\n') and changed.endswith('\nsuffix')
    with pytest.raises(ValueError):transform_source(source.replace(old,'different'))
    with pytest.raises(ValueError):transform_source(source+'\n'+old)
