from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from trace_patchmatch_render_pixels import source_identity,seam_depth_statistics


def test_source_identity_distinguishes_single_source_and_average():
    colors=np.array([[230,25,75],[60,180,75]],np.uint8)
    labels=np.array([[0,1],[0,1]])
    warps=[np.full((2,2,3),20,np.uint8),np.full((2,2,3),100,np.uint8)]
    yy,xx=np.indices((2,2))
    prediction=np.stack(warps)[labels,yy,xx]
    rows=source_identity(prediction,warps,colors[labels],colors)
    assert all(row['max_rgb_difference_8bit']==0 for row in rows)
    prediction[0,1]=60
    rows=source_identity(prediction,warps,colors[labels],colors)
    assert rows[1]['pixels_differing_over_one_lsb']==1


def test_categorical_seam_does_not_imply_depth_boundary():
    colors=np.array([[230,25,75],[60,180,75]],np.uint8)
    labels=np.zeros((4,6),int);labels[:,3:]=1
    depth=np.ones((4,6),np.float32)
    stats=seam_depth_statistics(colors[labels],depth,colors,(0,0,6,4))
    assert stats['adjacent_pixel_pairs']==4 and stats['fraction_under_0_001']==1
    depth[:,3:]=2
    stats=seam_depth_statistics(colors[labels],depth,colors,(0,0,6,4))
    assert stats['median_abs_log_depth_jump']==pytest.approx(np.log(2))
    assert stats['fraction_under_0_001']==0
