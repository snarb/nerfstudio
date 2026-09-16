from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from study_temporal_source_retention import CLIPS,FRAMES,subset
import numpy as np
from review_temporal_source_retention import exact_geometry,temporal_sources


def test_contiguous_actual_times():
    assert len(FRAMES)==len(set(FRAMES))==48
    for frames in CLIPS.values():
        assert len(frames)==24
        assert all(int(b)-int(a)==2 for a,b in zip(frames,frames[1:]))
        assert all((int(f)-899)//2<118 for f in frames)


def test_subset_preserves_camera_mesh_and_sources():
    q={'inventory':[{'frame_id':'000973','index':37,'camera':{'a':1},'mesh':'f'}],
       'source_rows':[{'source_dataset':'/data/000973','source_images':[1]}],'ordered_frame_ids':['000973']}
    out=subset(q,'000973');assert out==q
    out['inventory'][0]['camera']['a']=2
    assert q['inventory'][0]['camera']['a']==1
    with pytest.raises(AssertionError):subset(q,'000975')
    q['inventory'][0]['index']=118
    with pytest.raises(AssertionError):subset(q,'000973')


def test_geometry_audit_detects_changes(tmp_path):
    roots=[tmp_path/'a',tmp_path/'b']
    for root in roots:
        root.mkdir();np.savez(root/'target_depth.npz',depth=np.ones((3,4),np.float32))
        np.save(root/'face_source_labels.npy',np.arange(5))
    exact_geometry(*roots)
    np.save(roots[1]/'face_source_labels.npy',np.arange(5)+1)
    with pytest.raises(AssertionError):exact_geometry(*roots)


def test_identical_static_sequence_has_zero_switches():
    rng=np.random.default_rng(7)
    rgb=rng.integers(1,254,(480,270,3),dtype=np.uint8)
    ids=np.zeros((480,270),np.uint8)
    d=temporal_sources((rgb,rgb,ids,ids),(rgb,rgb,ids,ids))
    assert d['common_tracked_samples']>10000
    assert d['baseline_source_switches']==d['candidate_source_switches']==0
    new=ids.copy();new[100:200,100:200]=1
    d=temporal_sources((rgb,rgb,ids,ids),(rgb,rgb,ids,new))
    assert d['baseline_source_switches']==0
    assert d['candidate_source_switches']==10000
