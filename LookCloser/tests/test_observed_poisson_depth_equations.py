"""Synthetic checks of depth sign/normalization; RGB qualification is stubbed."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import align_poisson_to_measured_depth as alignment


def test_opengl_farther_observation_moves_along_negative_camera_z(monkeypatch):
    rows=[dict(physical_camera=str(i),transform_matrix=np.eye(4).tolist(),cx=960.,cy=540.,fl_x=1000.,fl_y=1000.) for i in range(4)]
    depth=np.full((1080,1920),1.004,np.float32);points=np.array([[0.,0.,-1.]])
    monkeypatch.setattr(alignment,'color_errors',lambda p,*args:(np.zeros((4,len(p))),np.zeros((4,len(p)))))
    h,g,count,_=alignment.depth_equations(points,rows,[depth]*4,{})
    np.testing.assert_allclose(h[0],np.diag([0,0,1]),rtol=0,atol=0)
    np.testing.assert_allclose(g[0],[0,0,-(float(depth[0,0])-1)],rtol=0,atol=1e-12)
    assert count.tolist()==[4]
    delta,_=alignment.solve_displacement(h,g,[],maximum_step=.006,prior_weight=1e-6)
    assert delta[0,2]<0 and delta[0,0]==0 and delta[0,1]==0


def test_far_or_unqualified_observation_is_not_a_constraint(monkeypatch):
    rows=[dict(physical_camera=str(i),transform_matrix=np.eye(4).tolist(),cx=960.,cy=540.,fl_x=1000.,fl_y=1000.) for i in range(4)]
    depth=np.full((1080,1920),1.004,np.float32);points=np.array([[0.,0.,-1.]])
    monkeypatch.setattr(alignment,'color_errors',lambda p,*args:(np.full((4,len(p)),np.nan),np.full((4,len(p)),np.nan)))
    h,g,count,_=alignment.depth_equations(points,rows,[depth]*4,{})
    assert not h.any() and not g.any() and not count.any()
    monkeypatch.setattr(alignment,'color_errors',lambda p,*args:(np.zeros((4,len(p))),np.zeros((4,len(p)))))
    depth.fill(1.02);h,g,count,_=alignment.depth_equations(points,rows,[depth]*4,{})
    assert not h.any() and not g.any() and not count.any()
