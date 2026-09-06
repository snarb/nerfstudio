from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from trace_patchmatch_render_pixels import source_identity,seam_depth_statistics


@pytest.mark.parametrize('z',[0.,-1.,np.nan,np.inf])
def test_missing_depth_is_not_a_camera_center_point(z):
    from trace_patchmatch_render_pixels import unproject_valid_pixel
    assert unproject_valid_pixel(np.full((2,2),z),{},0,0,.5) is None


def test_native_pixel_unprojection_and_bounds():
    from trace_patchmatch_render_pixels import unproject_valid_pixel
    frame=dict(cx=.5,cy=.5,fl_x=2.,fl_y=2.,transform_matrix=np.eye(4).tolist())
    np.testing.assert_allclose(unproject_valid_pixel(np.full((2,2),2.),frame,1,0,.5),[1,0,-2])
    with pytest.raises(ValueError):unproject_valid_pixel(np.ones((2,2)),frame,-1,0,.5)


def test_native_footprint_reports_wrong_layer_contribution():
    from PIL import Image
    from trace_patchmatch_render_pixels import bilinear_footprint
    depth=np.ones((2,2));depth[1,1]=2
    rgb=Image.new('RGB',(2,2),(10,20,30))
    rows=bilinear_footprint(depth,.25,.5,1.,rgb)
    assert sum(r['weight'] for r in rows)==pytest.approx(1.)
    assert sum(r['weight'] for r in rows if not r['same_depth_layer'])==pytest.approx(.125)
    assert rows[-1]['native_rgb8']==[10,20,30] and rows[-1]['mesh_depth']==2.


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
