import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
sp=importlib.util.spec_from_file_location('fourk_pilot',SCRIPTS/'build_marshawn_4k_pilots.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)


def test_exact_array_replay_rejects_single_pixel_or_dtype():
    a={'black':np.zeros((3,4),bool),'W64':np.ones((3,4),np.float32)}
    assert all(r['exact'] for r in m.exact_arrays(a,a).values())
    b={k:v.copy() for k,v in a.items()};b['W64'][1,1]=np.nextafter(np.float32(1),np.float32(0))
    with pytest.raises(ValueError,match='W64'):m.exact_arrays(a,b)
    b={k:v.copy() for k,v in a.items()};b['W64']=b['W64'].astype(np.float64)
    with pytest.raises(ValueError):m.exact_arrays(a,b)


def test_scaled_camera_preserves_pose_and_ray_geometry():
    r={'w':1920,'h':1012,'fl_x':1300.,'fl_y':1298.,'cx':960.,'cy':506.,'transform_matrix':np.eye(4).tolist(),'k1':0.}
    high=m.scaled_camera(r,2)
    assert high['w']==3840 and high['h']==2024
    assert high['transform_matrix']==r['transform_matrix'] and high['k1']==r['k1']
    uv=np.array([1223.5,421.5]);assert np.array_equal((uv-[r['cx'],r['cy']])/[r['fl_x'],r['fl_y']],(uv*2-[high['cx'],high['cy']])/[high['fl_x'],high['fl_y']])


def test_semantics_veto_overlap_hair_and_physical_kernels(monkeypatch):
    shapes=[];original=m.cv2.getStructuringElement
    monkeypatch.setattr(m.cv2,'getStructuringElement',lambda kind,size: (shapes.append(size) or original(kind,size)))
    black=np.zeros((180,220),bool);grey=black.copy();black[30:100,30:70]=True;grey[30:100,65:110]=True
    hard=(black|grey).astype(np.float32);hard[30,29]=1;veto=np.zeros_like(black);veto[60:65,40:45]=True;valid=np.ones_like(black)
    arrays=m.semantics_from_projected({'black':black,'grey':grey},hard,{'W32':hard,'W64':hard},veto,valid,2)
    assert shapes==[(33,33),(65,65),(129,129)]
    assert arrays['black'][30,29] and not (arrays['black']&arrays['grey']).any()
    assert not (arrays['core']&arrays['band']).any()
    assert all(not arrays[k][veto].any() for k in ['black','grey','W32','W64','core','band','nominal'])


def test_photometry_matches_frozen_order_and_not_gain_alone():
    row={'physical_camera':'AA','w':4,'h':3,'fl_x':3.,'fl_y':3.,'cx':2.,'cy':1.5}
    linear=np.full((3,4,3),.025,np.float32);contract={'global_base_gain':4};fit={'k':4.,'radial_global_center_EV':.457,'camera_rows':[{'camera':'AA','scalar_correction_EV':.136}]}
    expected=m.srgb(((linear*4).astype(np.float32).astype(np.float64)*np.exp2(m.radial_ev(row,4.)-.457+.136)[...,None]).astype(np.float32))
    actual=m.corrected(linear,row,contract,fit)
    assert np.array_equal(actual,expected)
    assert not np.array_equal(actual,m.srgb(linear*4))


def test_reject_more_than_two_workers_before_export(tmp_path):
    with pytest.raises(ValueError,match='two CPU'):m.run(tmp_path/'missing',tmp_path/'export',3)
    assert not (tmp_path/'export').exists()
    assert set(m.PILOTS)=={'AA','NE','JA','MC'}
