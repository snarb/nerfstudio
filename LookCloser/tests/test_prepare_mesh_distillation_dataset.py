import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_mesh_distillation_dataset import camera_plan, triangle_keys, support_products


def test_plan_has_exact_splits_and_convex_positions():
    rows=[]
    for x in range(14):
        for y in range(5):
            pose=np.eye(4); pose[:3,3]=[x,y,1]
            rows.append(dict(physical_camera=f'{chr(65+x)}004_{chr(65+y)}005_test',
                             transform_matrix=pose.tolist(),fl_x=100.,fl_y=101.,cx=50.,cy=30.))
    # Real production input has 62 rows; preserve a sufficiently rich middle rig.
    rows=rows[4:66]
    plan=camera_plan(rows)
    assert len(plan)==324
    assert sum(p['split']=='train' for p in plan)==300
    lookup={r['physical_camera']:np.array(r['transform_matrix'])[:3,3] for r in rows}
    for p in plan:
        w=np.array(p['weights'])
        assert np.all(w>=0) and np.isclose(w.sum(),1)
        np.testing.assert_allclose(np.array(p['camera']['transform_matrix'])[:3,3],w@np.array([lookup[n] for n in p['parents']]))
    assert plan==camera_plan(rows)
    positions=np.array([p['camera']['transform_matrix'] for p in plan])[:,:3,3]
    assert len(np.unique(positions,axis=0))==324


def test_triangle_provenance_ignores_reindex_and_winding():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    a=triangle_keys(v,np.array([[0,1,2],[0,1,3]]))
    b=triangle_keys(v[::-1],np.array([[1,3,2]]))
    assert np.isin(a,b).tolist()==[True,False]


def test_unknown_black_not_supervised_and_prior_depth_excluded():
    d=np.ones((9,9),np.float32);hit=np.ones((9,9),bool)
    count=np.full((9,9),3,np.uint8);count[4,4]=0
    prior=np.zeros((9,9),bool);prior[1,1]=True
    valid,trusted,weight=support_products(d,hit,count,prior,cv2)
    assert not valid[4,4] and not trusted[4,4] and weight[4,4]==0
    assert valid[1,1] and not trusted[1,1] and weight[1,1]==.25
    assert trusted[7,7] and weight[7,7]==1


def test_single_view_rgb_is_allowed_but_depth_is_not_trusted():
    depth=np.ones((7,7),np.float32);hit=np.ones((7,7),bool)
    count=np.ones((7,7),np.uint8);prior=np.zeros((7,7),bool)
    valid,trusted,weight=support_products(depth,hit,count,prior,cv2)
    assert valid.all() and not trusted.any()
    np.testing.assert_allclose(weight,1/3)


def test_depth_jump_is_not_geometry_supervision():
    depth=np.ones((7,7),np.float32);depth[:,4:]=2
    valid,trusted,weight=support_products(depth,np.ones((7,7),bool),
        np.full((7,7),5,np.uint8),np.zeros((7,7),bool),cv2)
    assert valid.all()
    assert not trusted[:,3:5].any()
    assert trusted[:,:3].all() and trusted[:,5:].all()
    np.testing.assert_allclose(weight[:,3:5],.25)
