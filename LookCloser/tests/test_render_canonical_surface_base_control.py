from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_canonical_surface_base_control import unproject_mesh_depth,apply_selected_offsets


def inputs():
    target=dict(h=2,w=3,fl_x=2.,fl_y=4.,cx=.5,cy=.5,transform_matrix=np.eye(4).tolist())
    manifest=dict(dataparser_scale=1.,dataparser_transform=np.eye(4)[:3].tolist())
    return np.ones((2,3),np.float32)*2,target,manifest


def test_mesh_unprojection_uses_half_pixel_and_no_extra_scale():
    depth,target,manifest=inputs();world=unproject_mesh_depth(depth,target,manifest)
    np.testing.assert_array_equal(world[0],[[0,0,-2],[1,0,-2],[2,0,-2]])
    np.testing.assert_array_equal(world[1,0],[0,-.5,-2])
    target['transform_matrix'][0][3]=3
    np.testing.assert_array_equal(unproject_mesh_depth(depth,target,manifest),world+[3,0,0])


@pytest.mark.parametrize('bad',['scale','transform','distortion','nonfinite','shape'])
def test_coordinate_mismatch_fails_closed(bad):
    depth,target,manifest=inputs()
    if bad=='scale':manifest['dataparser_scale']=.1
    if bad=='transform':manifest['dataparser_transform'][0][3]=1
    if bad=='distortion':target['k1']=.01
    if bad=='nonfinite':depth[0,0]=float('nan')
    if bad=='shape':depth=depth[:1]
    with pytest.raises(ValueError):unproject_mesh_depth(depth,target,manifest)


def test_offset_zero_replays_and_invalid_support_is_unchanged():
    baseline=np.full((2,3,3),.3,np.float32);offsets=np.full_like(baseline,.2)
    labels=np.array([[-1,0,1],[2,3,7]])
    off,_=apply_selected_offsets(baseline,labels,offsets,strength=0)
    np.testing.assert_array_equal(off,baseline)
    out,stats=apply_selected_offsets(baseline,labels,offsets)
    np.testing.assert_array_equal(out[0,0],baseline[0,0]);np.testing.assert_allclose(out[labels>=0],.5)
    assert stats['changed_pixels']==5 and stats['clipped_channels']==0


def test_clipping_is_explicitly_counted():
    out,stats=apply_selected_offsets(np.full((1,1,3),.9),np.array([[0]]),np.full((1,1,3),.2))
    np.testing.assert_array_equal(out,1);assert stats['clipped_channels']==3


@pytest.mark.parametrize('bad',['nan','strength','labels'])
def test_invalid_color_or_labels_rejected(bad):
    rgb=np.zeros((2,3,3),np.float32);offset=rgb.copy();labels=np.zeros((2,3),np.int32);strength=1
    if bad=='nan':offset[0,0,0]=float('nan')
    if bad=='strength':strength=1.1
    if bad=='labels':labels[0,0]=8
    with pytest.raises(ValueError):apply_selected_offsets(rgb,labels,offset,strength=strength)
