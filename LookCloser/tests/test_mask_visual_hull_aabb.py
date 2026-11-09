"""Actual synthetic calibrated image formation, occlusion and slender geometry."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mask_visual_hull_aabb import (MaskCamera,derive_aabb,classify_cells,ray_box_hits,
    select_person_instances,temporal_envelope,apply_minimum_margins,rebuild_margins)

BODY=np.array([[-.28,-.2,-.3],[.28,.2,.55]])
LEG=np.array([[-.21,-.08,-.9],[-.17,.08,-.3]])
SECOND=np.array([[.18,-.1,-.65],[.36,.12,.2]])
CREW=np.array([[.85,.85,-.5],[1.15,1.15,.5]])

def camera(origin,size=128):
    origin=np.asarray(origin,float);forward=-origin/np.linalg.norm(origin)
    up=np.array([0.,0.,1.])
    if abs(forward@up)>.9:up=np.array([0.,1.,0.])
    right=np.cross(forward,up);right/=np.linalg.norm(right);up=np.cross(right,forward)
    pose=np.eye(4);pose[:3,:3]=np.stack([right,up,-forward],1);pose[:3,3]=origin
    yy,xx=np.mgrid[:size,:size];local=np.stack([(xx+.5-size/2)/100,-(yy+.5-size/2)/100,-np.ones_like(xx)],-1)
    dirs=local@pose[:3,:3].T
    masks=np.stack([ray_box_hits(origin,dirs,CREW),ray_box_hits(origin,dirs,BODY)|ray_box_hits(origin,dirs,LEG),ray_box_hits(origin,dirs,SECOND)])
    fg,unknown=select_person_instances(masks,{0:'crew',1:'fighter_black',2:'fighter_grey'},['fighter_black','fighter_grey'])
    return MaskCamera(str(origin),pose,100,100,size/2,size/2,fg,unknown,np.ones_like(fg))

def rig():
    # Enough independent views to reject a crew cone overlapping three targets.
    return [camera([3*np.cos(a),3*np.sin(a),z])for z in [0.,1.5]for a in np.arange(8)*np.pi/4]

def bounds(cameras,**kw):
    support=kw.pop('min_support',3)
    return derive_aabb(cameras,[[-1.4,-1.4,-1.2],[1.4,1.4,1.2]],leaf_size=.09,min_support=support,
        max_background=1,padding_native=1,margin_world=.03,snap_step=.01,max_cells=1000000,**kw)

def test_known_cuboids_slender_leg_and_false_crew_instance():
    report,cells=bounds(rig(),min_support=8);b=np.array(report['aabb'])
    assert np.all(b[0]<=np.minimum.reduce([BODY[0],LEG[0],SECOND[0]]))
    assert np.all(b[1]>=np.maximum.reduce([BODY[1],LEG[1],SECOND[1]]))
    assert (b[1,:2]<CREW[0,:2]).any() # High-rank crew instance excluded by stable ID.
    assert not report['requires_expanded_search']
    assert (cells['foreground_support']>=3).all()

def test_fully_occluded_camera_unknown_not_background():
    cameras=rig();cameras[0].foreground[:]=False;cameras[0].unknown[:]=True
    report,_=bounds(cameras);b=np.array(report['aabb'])
    assert b[0,2]<=LEG[0,2]and b[1,2]>=BODY[1,2]

def test_false_negative_thin_limb_is_protected_by_padding():
    cameras=rig()
    # Remove a small foreground strip in two cameras, as a segmentation FN.
    for c in cameras[:2]:c.foreground[:,60:62]=False
    report,_=bounds(cameras);assert report['aabb'][0][2]<=LEG[0,2]

def test_corner_footprint_not_center_only_and_unknown_vs_clear():
    c=camera([0,-3,0]);c.foreground[:]=False;c.foreground[64,55]=True;c.unknown[:]=False;c.prepare(0)
    lo=np.array([[-.4,-.1,-.1]]);hi=np.array([[.4,.1,.1]])
    fg,possible,bg,unknown=classify_cells([c],lo,hi)
    assert fg[0]==1 and bg[0]==0 # One edge pixel, not the center, supports cell.
    c.foreground[:]=False;c.unknown[:]=True;c.prepare(0)
    assert classify_cells([c],lo,hi)[2][0]==0

def test_unknown_duplicate_does_not_erase_selected_actor_support():
    c=camera([0,-3,0]);c.unknown|=c.foreground;c.prepare(0)
    lo=BODY[0][None];hi=BODY[1][None];fg,possible,bg,unknown=classify_cells([c],lo,hi)
    assert fg[0]==1 and bg[0]==0 and unknown[0]==1

def test_partial_nearplane_and_out_of_fov_are_unknown_not_background():
    c=camera([0,-3,0]);c.prepare(0)
    # Cell crosses camera plane and cannot be safely projected yet.
    lo=np.array([[-.1,-3.1,-.1]]);hi=np.array([[.1,-2.9,.1]])
    fg,possible,bg,unknown=classify_cells([c],lo,hi)
    assert bg[0]==0 and unknown[0]==1 and possible[0]==1
    lo=np.array([[10.,-.1,0.]]);hi=np.array([[11.,.1,.2]])
    assert classify_cells([c],lo,hi)[2][0]==0

def test_search_domain_clip_is_reported_not_silent():
    cameras=rig()
    r,_=derive_aabb(cameras,[[-.3,-.3,-.6],[.4,.3,.6]],leaf_size=.08,min_support=3,
        max_background=1,padding_native=2,margin_world=0,snap_step=.01)
    assert r['requires_expanded_search'] # True slender leg extends below domain.

def test_fixed_world_temporal_envelope_preserves_both_frames():
    a=np.array([[-.201,-.1,-.9],[.35,.21,.55]]);b=a+np.array([.013,0,.002])
    union=temporal_envelope([a,b],snap_step=.05)
    assert np.all(union[0]<=np.minimum(a[0],b[0]))and np.all(union[1]>=np.maximum(a[1],b[1]))
    np.testing.assert_allclose(union/.05,np.round(union/.05),atol=1e-12)

def test_minimum_xyz_bottom_margin_guaranteed_after_snap_and_monotone():
    source=np.array([[-.201,.013,-.143],[.137,.299,.227]])
    small,sm=apply_minimum_margins(source,.02,snap_step=.01)
    large,proof=apply_minimum_margins(source,.03,[.05,.01,.04],.09,snap_step=.01)
    assert np.all(large[0]<=small[0])and np.all(large[1]>=small[1])
    assert np.all(source[0]-large[0]>=np.array([.05,.03,.09])-1e-12)
    assert np.all(large[1]-source[1]>=np.array([.05,.03,.04])-1e-12)
    assert proof['verified_actual_margins_at_least_requested']
    assert proof['effective_lower_face_minimums']==pytest.approx([.05,.03,.09])

@pytest.mark.parametrize('common,axis,bottom',[
    (-.01,None,None),(float('nan'),None,None),(.01,[-1,0,0],None),
    (.01,[0,float('nan'),0],None),(.01,None,-.1),(.01,None,float('nan'))])
def test_invalid_margin_rejected(common,axis,bottom):
    with pytest.raises(ValueError,match='nonnegative'):
        apply_minimum_margins(np.array([[0.,0.,0.],[1.,1.,1.]]),common,axis,bottom)

def test_temporal_envelope_has_minimum_margin_to_all_actual_frame_extents():
    frames=np.array([[[-.2,0,-.1],[.2,.3,.5]],[[-.18,.02,-.13],[.24,.31,.51]]])
    result=temporal_envelope(frames,snap_step=.03,minimum_margin=.04,bottom_margin=.11)
    assert np.all(frames[:,0]-result[0]>=np.array([.04,.04,.11])-1e-12)
    assert np.all(result[1]-frames[:,1]>=.04-1e-12)

def test_margin_only_rebuild_uses_saved_voxel_extents_without_recarving(tmp_path):
    import json
    cells=tmp_path/'cells.npz';np.savez(cells,lower=np.array([[0.,0.,0.],[.3,0.,0.]]),
        upper=np.array([[.1,.1,.1],[.4,.1,.1]]))
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'retained_extent_aabb':[[0,0,0],[.4,.1,.1]],
        'padding_native_pixels':128,'source_bindings':{}}))
    result=rebuild_margins(receipt,cells,minimum_margin=.02,axis_margin=[.05,.03,.04],bottom_margin=.08,snap_step=.01)
    box=np.asarray(result['aabb'])
    assert np.all(box[0]<=np.array([-.05,-.03,-.08])+1e-12)
    assert np.all(box[1]>=np.array([.45,.13,.14])-1e-12)
    np.testing.assert_allclose(box/.01,np.round(box/.01),atol=1e-12)
    assert result['kind']=='MARGIN_ONLY_REBUILD_NO_RECARVING'
