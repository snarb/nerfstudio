import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from local_silhouette_volume import signed_pixels, combine_silhouettes, remove_box_caps
from silhouette_domain_surface import availability_bits,stable_domain_faces


def test_unknown_camera_is_not_background_or_support():
    mask=np.zeros((9,9),bool);mask[2:7,2:7]=True
    f=signed_pixels(mask);d=np.ones_like(mask)
    uv=np.array([[[4.,4.]],[[4.,4.]],[[4.,4.]],[[50.,4.]]])
    out,n,p=combine_silhouettes(uv,np.ones((4,1)),[f]*4,[d]*4)
    assert out[0]>0 and n[0]==3 and p[0]==3
    out,n,p=combine_silhouettes(uv,np.ones((4,1)),[f]*4,[d]*4,minimum_views=4)
    assert out[0]<0


def test_available_disagreement_vetoes_and_margin_is_explicit():
    mask=np.zeros((9,9),bool);mask[2:7,2:7]=True
    f=signed_pixels(mask);d=np.ones_like(mask)
    uv=np.array([[[4.,4.]],[[4.,4.]],[[1.,4.]]])
    out,n,p=combine_silhouettes(uv,np.ones((3,1)),[f]*3,[d]*3)
    assert out[0]<0 and n[0]==3 and p[0]==2
    out,_,_=combine_silhouettes(uv,np.ones((3,1)),[f]*3,[d]*3,margin_pixels=2)
    assert out[0]>0


def test_annotation_footprint_and_behind_camera_are_unknown():
    mask=np.zeros((9,9),bool);mask[2:7,2:7]=True
    f=signed_pixels(mask);d=np.ones_like(mask);d[4,5]=False
    uv=np.array([[[4.5,4.]],[[4.,4.]]])
    out,n,_=combine_silhouettes(uv,np.array([[1.],[-1.]]),[f]*2,[d]*2,minimum_views=1)
    assert n[0]==0 and out[0]<0


def test_box_caps_removed_and_bad_masks_rejected():
    v=np.array([[0,0,0],[.5,.5,.5],[.5,.6,.5],[.5,.5,.6]])
    t=np.array([[0,1,2],[1,2,3]])
    np.testing.assert_array_equal(remove_box_caps(v,t,[0]*3,[1]*3,.01),[False,True])
    with pytest.raises(ValueError):signed_pixels(np.zeros((3,3)))


def test_availability_edge_cannot_be_an_anatomical_cap():
    bits=np.full((3,3,3),7,np.uint16)
    vertices=np.array([[.2,.2,.2],[.6,.2,.2],[.2,.6,.2]])
    tri=np.array([[0,1,2]])
    assert stable_domain_faces(vertices,tri,[0,0,0],1,bits)[0]
    bits[1,1,1]=3
    assert not stable_domain_faces(vertices,tri,[0,0,0],1,bits)[0]
    # Even when both sides have >=3 views, a new fourth view is a domain edge.
    bits[1,1,1]=15
    assert not stable_domain_faces(vertices,tri,[0,0,0],1,bits)[0]


def test_availability_bits_track_each_camera():
    d=np.ones((9,9),bool)
    uv=np.array([[[4.,4.]],[[10.,4.]],[[4.,4.]]])
    bits=availability_bits(uv,np.array([[1.],[1.],[-1.]]),[d]*3)
    assert bits[0]==1
