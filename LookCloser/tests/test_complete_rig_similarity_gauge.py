from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from complete_rig_similarity_gauge import transform_camera_similarity,complete_query_gauge


def test_similarity_preserves_query_projection_with_scene_transform():
    pose=np.eye(4);pose[:3,3]=[1,2,3]
    pose[:3,:3]=Rotation.from_rotvec([-.2,.1,.3]).as_matrix()
    frame={'transform_matrix':pose.tolist(),'fl_x':9000,'physical_camera':'query'}
    rotation=Rotation.from_rotvec([.03,-.02,.01]).as_matrix();scale=1.03;translation=np.array([.1,-.2,.03])
    point=np.array([1.1,1.8,2.3])
    transformed=transform_camera_similarity(frame,scale,rotation,translation)
    moved=np.asarray(transformed['transform_matrix'])
    old_camera=(point-pose[:3,3])@pose[:3,:3]
    new_camera=(scale*(rotation@point)+translation-moved[:3,3])@moved[:3,:3]
    np.testing.assert_allclose(new_camera,scale*old_camera,atol=1e-12)
    assert transformed['fl_x']==frame['fl_x'] and frame['transform_matrix']==pose.tolist()


def fixture():
    frames=[{'physical_camera':name,'transform_matrix':np.eye(4).tolist()} for name in ['train','query']]
    payload={'frames':frames}
    manifest=dict(camera_changes=[{'physical_camera':'train'}],uses_eval_rgb=False,per_time_camera_optimization=False,
        gauge_alignment=dict(scale=1.,rotation=np.eye(3).tolist(),translation=[1.,0.,0.]))
    return payload,manifest


def test_only_untouched_queries_receive_gauge():
    payload,manifest=fixture();result=complete_query_gauge(payload,payload,manifest)
    assert result['frames'][0]==payload['frames'][0]
    assert result['frames'][1]['transform_matrix'][0][3]==1
    assert payload['frames'][1]['transform_matrix'][0][3]==0


def test_already_changed_query_is_rejected():
    from copy import deepcopy
    payload,manifest=fixture();changed=deepcopy(payload)
    changed['frames'][1]['transform_matrix'][0][3]=.1
    with pytest.raises(ValueError):complete_query_gauge(payload,changed,manifest)


def test_negative_scale_is_rejected():
    payload,_=fixture()
    with pytest.raises(ValueError):transform_camera_similarity(payload['frames'][0],-1,np.eye(3),np.zeros(3))
