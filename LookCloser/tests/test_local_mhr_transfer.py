from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from transfer_local_mhr_prior import settings
from review_local_mhr_transfer import enclosed_misses
from run_local_mhr_completion import adapt


def test_settings_never_reuses_previous_pose(tmp_path):
    s=settings(dict(root=str(tmp_path/'fresh'),inputs=str(tmp_path/'inputs'),frame='001083'))
    assert s['FRAME']=='001083'
    assert s['CANONICAL']==tmp_path/'fresh/canonical_pose'
    assert s['FINAL']==tmp_path/'fresh/silhouette100'
    assert s['RGB']==tmp_path/'inputs/rgb'


def test_enclosed_misses_excludes_crop_border_not_ground_truth():
    d=np.ones((12,12));d[4:6,4:6]=0;d[:3,2]=0
    mask=enclosed_misses(d,[0,0,12,12])
    assert mask.sum()==4 and mask[4:6,4:6].all()
    assert not mask[:3,2].any()
    assert enclosed_misses(np.ones((12,12)),[0,0,12,12]).sum()==0


def test_anchor_path_adapter_keeps_confidence_rules():
    import study_mhr_local_head_prior as head
    before="Path('/mnt/data/dec5_jaw_measured_depth/analysis')"
    _,proof=adapt(head,'anchors',[(before,'DEPTH_ROOT',1)],dict(DEPTH_ROOT=Path('/explicit/depth')))
    assert proof['generated_source'].replace('DEPTH_ROOT',before)==proof['original_source']
    assert 'other>=3' in proof['generated_source'] and 'distance<=.006' in proof['generated_source']


def test_actual_fresh_landmarks_are_same_time_and_train_only():
    from study_multiview_face_prior import read
    from joint_temporal_texture import HELD_CAMERAS
    root=Path('/mnt/data/dec5_mhr_transfer_001083_inputs/rgb')
    if not (root/'inference.json').exists():pytest.skip('DEC5 integration assets not installed')
    q=read(root/'inference.json');r=q['records']
    assert len(r)==62 and {x['frame'] for x in r}=={'001083'}
    assert len({x['camera'] for x in r})==62 and not {x['camera'] for x in r}&HELD_CAMERAS
    assert read(root/'request.json')['model_z_or_transform_used_as_metric'] is False
