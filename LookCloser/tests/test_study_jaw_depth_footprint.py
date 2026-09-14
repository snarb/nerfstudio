import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_train_confidence import enclosing_samples


def inputs():
    rows=[]
    for i,x in enumerate([0,.1,-.1]):
        pose=np.eye(4);pose[0,3]=x
        rows.append(dict(physical_camera=str(i),transform_matrix=pose.tolist(),fl_x=100,fl_y=100,cx=50,cy=50))
    return rows,[np.full((1080,1920),2,dtype=np.float32) for _ in rows]


def test_votes_use_unique_train_views_and_deterministic_ties():
    rows,depths=inputs();points=np.array([[0,0,-2.]])
    votes,refs=train_reference_votes(points,rows,depths)
    assert votes.tolist()==[3] and rows[refs[0]]['physical_camera']=='0'
    votes2,refs2=train_reference_votes(points,rows[::-1],depths[::-1])
    assert votes2.tolist()==[3] and rows[::-1][refs2[0]]['physical_camera']=='0'


def test_no_anchor_from_missing_depth():
    rows,depths=inputs()
    for d in depths:d.fill(0)
    votes,refs=train_reference_votes(np.array([[0,0,-2.]]),rows,depths)
    assert votes.tolist()==[0] and refs.tolist()==[-1]


def test_fractional_footprint_does_not_veto_on_one_far_corner():
    depth=np.ones((4,4));depth[2,2]=1.004
    _,_,far=enclosing_samples(np.array([[1.9,1.6]]),np.array([1.]),depth)
    assert not far[0]
    depth[1:3,1:3]=1.004
    _,_,far=enclosing_samples(np.array([[1.9,1.6]]),np.array([1.]),depth)
    assert far[0]
    depth[1,1]=0
    _,_,far=enclosing_samples(np.array([[1.9,1.6]]),np.array([1.]),depth)
    assert not far[0]
