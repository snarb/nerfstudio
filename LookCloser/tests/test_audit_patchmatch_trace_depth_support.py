from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_patchmatch_trace_depth_support import native_evidence,trace_points


def test_missing_depth_never_votes_and_not_bilinearly_mixed():
    depth=np.zeros((7,7));depth[3,3]=2.
    row=native_evidence(depth,3.49,3.49,1.)
    assert row['status']=='unknown_or_mixed' and row['positive_taps']==1
    assert row['native_taps'][2][2]==2.
    assert row['near_within_gap']==0


@pytest.mark.parametrize('depth,z,status',[(1.,1.,'near_surface'),(1.1,1.,'measured_free_space'),
                                         (.9,1.,'foreground_occlusion'),(0.,1.,'unknown_or_mixed')])
def test_native_layers(depth,z,status):
    assert native_evidence(np.full((7,7),depth),3.,3.,z)['status']==status


def test_unstable_far_layer_is_not_free_space_evidence():
    depth=np.ones((7,7))*1.1;depth[1:4]=1.5
    assert native_evidence(depth,3,3,1)['status']=='unknown_or_mixed'


def test_boundary_and_behind_camera_are_unknown():
    depth=np.ones((7,7))
    assert native_evidence(depth,0,0,1)['status']=='outside_native_footprint'
    assert native_evidence(depth,3,3,-1)['status']=='behind_camera_or_nonfinite'


def test_legacy_zero_depth_world_is_not_a_surface():
    valid,missing=trace_points({'pixels':[{'pixel':[2,3],'target_depth':0.,'world':[1,2,3]},
                                        {'pixel':[4,5],'target_depth':1.,'world':[2,3,4]}]})
    assert len(valid)==len(missing)==1 and valid[0]['index']==1
    with pytest.raises(ValueError):trace_points({'pixels':[{'pixel':[0,0],'target_depth':1.,'world':None}]})
