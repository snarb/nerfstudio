import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from review_temporal_face_angular_control import tracked_appearance


def test_same_baseline_motion_samples_expose_candidate_only_color_step():
    rng=np.random.default_rng(7);a=rng.integers(20,190,(480,270,3),dtype=np.uint8)
    ids=np.zeros((480,270),np.uint8)
    result=tracked_appearance((a,a,ids,ids),(a,a+10,ids,ids))
    assert result['common_samples']>50000
    assert result['baseline']['mean_rgb_step']<.01
    assert abs(result['candidate']['mean_rgb_step']-10)<.01
