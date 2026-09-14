import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dynamic_grid_flythrough import open_grid_path
from train_foreground_guard import carve_candidates, mask_from_probability


def rows_for_grid():
    rows=[]
    for col in 'FGHI':
        for level in 'ABCD':
            pose=np.eye(4);pose[:3,3]=[(ord(col)-ord('H'))*.025,(ord(level)-ord('C'))*.025,1]
            name=f'{col}004_{level}005_test'
            if col=='H' and level=='C':name='H004_C005_1210SZ'
            rows.append(dict(physical_camera=name,transform_matrix=pose.tolist(),fl_x=100,fl_y=100,cx=50,cy=50,w=100,h=100))
    return rows


@pytest.mark.parametrize('size',[3,4])
def test_dynamic_path_has_full_extent_convexity_and_no_jumps(size):
    rows=rows_for_grid();path,report=open_grid_path(rows,np.zeros(3),size)
    assert len(path)==150 and len({r['physical_camera'] for r in path})==150
    assert np.allclose(report['achieved_grid_interval_extent_xy'],[.96*(size-1)]*2)
    weights=np.array([r['convex_weights'] for r in path]);assert weights.min()>0
    by_name={r['physical_camera']:r for r in rows}
    corners=np.array([by_name[n]['transform_matrix'] for n in report['anchors']])[:,:3,3]
    positions=np.array([r['transform_matrix'] for r in path])[:,:3,3]
    assert np.allclose(positions,weights@corners)
    assert report['raw_calibration_motion']['speed_max_min_ratio']<1.002
    assert np.linalg.norm(positions[0]-positions[-1])>.05
    assert not report['continuous_periodic_loop']


def test_silhouette_requires_every_vertex_and_ignores_out_of_frame():
    row=dict(transform_matrix=np.eye(4).tolist(),fl_x=10,fl_y=10,cx=5,cy=5)
    vertices=np.array([[0,0,-1],[.1,0,-1],[0,.1,-1],[20,0,-1]],np.float32)
    triangles=np.array([[0,1,2],[0,1,3]])
    masks=np.zeros((6,10,10),np.uint8)
    remove,votes=carve_candidates(vertices,triangles,[row]*6,masks,6)
    assert remove.tolist()==[True,False]
    assert votes.tolist()==[6,6,6,0]
    masks[0,:,:]=1
    assert not carve_candidates(vertices,triangles,[row]*6,masks,6)[0].any()


def test_invalid_semantic_seed_fails_closed():
    with pytest.raises(ValueError,match='Unusable semantic seeds'):
        mask_from_probability(np.zeros((100,100,3),np.uint8),np.zeros((100,100),np.float32))


def test_audit_rejects_static_time_inventory(tmp_path,monkeypatch):
    import finalize_dynamic_grid_video as audit
    request=dict(ordered_frame_ids=['000973']*150,inventory=[dict(frame_id='000973')]*150)
    monkeypatch.setattr(audit,'verify_request',lambda _:request)
    monkeypatch.setattr(audit,'sha',lambda _:'digest')
    monkeypatch.setattr(audit,'read',lambda _:dict(frames=[]))
    with pytest.raises(ValueError,match='different chronological'):
        audit.audit(tmp_path)


def test_audit_rejects_partial_outputs(tmp_path,monkeypatch):
    import finalize_dynamic_grid_video as audit
    ids=[f'{899+2*i:06d}' for i in range(150)]
    request=dict(ordered_frame_ids=ids,inventory=[dict(frame_id=f) for f in ids])
    monkeypatch.setattr(audit,'verify_request',lambda _:request)
    monkeypatch.setattr(audit,'sha',lambda _:'digest')
    monkeypatch.setattr(audit,'read',lambda _:dict(frames=[]))
    with pytest.raises(ValueError,match='Incomplete or extra'):
        audit.audit(tmp_path)


def test_publication_requires_actual_encoded_review(tmp_path,monkeypatch):
    import finalize_dynamic_grid_video as audit
    monkeypatch.setattr(audit,'sha',lambda _:'digest')
    monkeypatch.setattr(audit,'read',lambda _:dict(video_sha256='digest',review_status='pending_actual_encoded_review'))
    with pytest.raises(ValueError,match='Actual encoded video review'):
        audit.verify_video_review(tmp_path)


def test_publication_rejects_changed_encoded_evidence(tmp_path,monkeypatch):
    import finalize_dynamic_grid_video as audit
    video=dict(video_sha256='digest',review_status='reviewed_with_failures',review_notes='Observed holes',encoded_overview_sha256='digest')
    crop=dict(video_sha256='digest',status='reviewed_with_failures',notes='Observed holes',
              evidence=[dict(path='crop.png',sha256='old')]*4)
    monkeypatch.setattr(audit,'sha',lambda _:'digest')
    monkeypatch.setattr(audit,'read',lambda p:video if Path(p).name=='video_manifest.json' else crop)
    with pytest.raises(ValueError,match='Changed decoded crop evidence'):
        audit.verify_video_review(tmp_path)
