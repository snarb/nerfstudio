import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_marshawn_contour_sources_cpu import select_layers,project


def test_actual_weight_partition_peak_and_conditional_median():
    assert select_layers([.1,.2,.05,.4,.25],[False,True,False,True,False])==dict(removed_peak=4,removed_q50=4,retained_peak=3,retained_q50=3)


def test_empty_layer_and_bad_weights_rejected():
    for w,k in [([1,0],[True,True]),([1,-1],[True,False]),([np.nan,1],[True,False])]:
        with pytest.raises(ValueError):select_layers(w,k)


def test_projection_preserves_floor_frame_and_pixel_center():
    f=dict(transform_matrix=np.eye(4).tolist(),fl_x=100,fl_y=120,cx=50,cy=60)
    uv,z=project([.105,-.0875,-1],f)
    np.testing.assert_allclose(uv,[60.5,70.5]);assert z==1
