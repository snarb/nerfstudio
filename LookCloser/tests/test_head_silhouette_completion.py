from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_head_silhouette_completion import sample_field
from joint_temporal_texture import project
from local_silhouette_volume import signed_pixels,combine_silhouettes
from silhouette_domain_surface import stable_domain_faces


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_field_matches_cpu_with_62_known_camera_bits(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    row=dict(transform_matrix=np.eye(4).tolist(),fl_x=20.,fl_y=20.,cx=16.,cy=16.)
    rows=[row]*62;mask=np.zeros((32,32),bool);mask[8:24,8:24]=True
    fields=[signed_pixels(mask)]*62;points=np.array([[0.,0.,-1],[.4,0,-1],[4,0,-1]])
    field,count,bits=sample_field(points,rows,fields,margin=1.,device=device)
    uv,z=project(points,rows);expected,n,_=combine_silhouettes(uv,z,fields,[np.ones((32,32),bool)]*62,3,-1.)
    np.testing.assert_allclose(field,expected,atol=1e-5);np.testing.assert_array_equal(count,n)
    assert bits[0]==bits[1]==(1<<62)-1 and bits[2]==0
    assert field[0]>0 and field[1]<0


def test_unknown_to_known_transition_is_not_a_surface():
    bits=np.full((2,2,2),(1<<62)-1,dtype=np.uint64)
    v=np.array([[.1,.1,.1],[.2,.1,.1],[.1,.2,.1]]);tri=np.array([[0,1,2]])
    assert stable_domain_faces(v,tri,np.zeros(3),1.,bits)[0]
    bits[1,1,1]=0
    assert not stable_domain_faces(v,tri,np.zeros(3),1.,bits)[0]
