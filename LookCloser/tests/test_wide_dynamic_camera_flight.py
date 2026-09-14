import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wide_dynamic_camera_flight import wide_path


def rig():
    rows=[]
    for col in 'DEFGHIJKL':
        for level in 'ABCDE':
            pose=np.eye(4)
            # Sideways sensor: native X is portrait-up, native Y portrait-left.
            pose[:3,3]=[(ord('C')-ord(level))*.06,-(ord(col)-ord('H'))*.06,1]
            name=f'{col}004_{level}005_test'
            if col=='H' and level=='C':name='H004_C005_1210SZ'
            rows.append(dict(physical_camera=name,transform_matrix=pose.tolist(),fl_x=100,fl_y=100,cx=50,cy=50,w=100,h=100))
    return rows


def test_wide_flight_has_horizontal_then_vertical_motion_and_real_limits():
    rows=rig();path,report=wide_path(rows,np.zeros(3))
    assert len(path)==150 and len({p['physical_camera'] for p in path})==150
    xy=np.array([p['rig_offset_xy'] for p in path])
    assert np.ptp(xy[:,0])>7.9 and np.ptp(xy[:,1])>3.95
    assert xy[:,0].max()<=4 and xy[:,0].min()>=-4
    assert xy[:,1].max()<=2 and xy[:,1].min()>=-2
    assert xy[0,0]<-3.8
    indices=report['extrema_indices'];assert indices['right']<indices['top']<indices['bottom']
    assert report['vertical_available_offsets']==[-2,2] and report['vertical_limit_disclosed']
    assert report['continuous_periodic_camera'] and not report['actor_clip_periodic']
    weights=np.array([r['convex_weights'] for r in path]);assert weights.min()>=0
    lookup={r['physical_camera']:r for r in rows}
    corners=np.array([lookup[n]['transform_matrix'] for n in report['anchors']])[:,:3,3]
    poses=np.array([r['transform_matrix'] for r in path])
    assert np.allclose(poses[:,:3,3],weights@corners)
    assert report['maximum_pairwise_view_angle_degrees']>20
    closed_steps=np.linalg.norm(np.roll(poses[:,:3,3],-1,axis=0)-poses[:,:3,3],axis=1)
    assert closed_steps.max()/closed_steps.min()<1.04
    assert np.allclose(np.linalg.det(poses[:,:3,:3]),1)
    # Verify changing rays on an actual asymmetric mesh, not only pose metadata.
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    mesh=o3d.geometry.TriangleMesh.create_box(.15,.2,.25)
    mesh.translate([-.08,-.05,-.1]);scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    a=camera_depth(scene,path[indices['left']])[0]
    b=camera_depth(scene,path[indices['right']])[0]
    assert np.count_nonzero(np.isfinite(a)!=np.isfinite(b))>20


def test_wide_flight_does_not_invent_missing_anchors():
    rows=[r for r in rig() if not r['physical_camera'].startswith('L004_E005')]
    with pytest.raises(KeyError):wide_path(rows,np.zeros(3))


def test_wide_audit_rejects_repeated_source_time(tmp_path,monkeypatch):
    import finalize_wide_dynamic_flight as module
    monkeypatch.setattr(module,'verify_request',lambda _:dict(ordered_frame_ids=['000973']*150,inventory=[]))
    monkeypatch.setattr(module,'sha',lambda _:'digest')
    with pytest.raises(ValueError,match='source-time inventory'):module.audit(tmp_path)


def test_wide_audit_rejects_partial_render_before_encoding(tmp_path,monkeypatch):
    import finalize_wide_dynamic_flight as module
    ids=[f'{899+2*i:06d}' for i in range(150)]
    monkeypatch.setattr(module,'verify_request',lambda _:dict(ordered_frame_ids=ids,inventory=[dict(frame_id=f) for f in ids]))
    monkeypatch.setattr(module,'sha',lambda _:'digest')
    with pytest.raises(ValueError,match='Incomplete renders'):module.audit(tmp_path)


def test_encoded_review_needs_explicit_findings(tmp_path):
    import finalize_wide_dynamic_flight as module
    with pytest.raises(ValueError,match='actual MP4 findings'):module.record_encoded_review(tmp_path,'')
