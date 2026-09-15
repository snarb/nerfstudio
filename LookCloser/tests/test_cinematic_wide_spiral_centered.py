from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cinematic_wide_spiral_centered as spiral


def test_physical_arc_length_retains_wide_full_turn_and_hold(monkeypatch):
    rows=[]
    for name,x,y in [('C004_B005',-5,1),('K004_B005',3,1),
                     ('K004_D005',3,-1),('C004_D005',-5,-1),('H004_C005',0,0)]:
        pose=np.eye(4);pose[:3,3]=[x,y,2]
        rows.append(dict(physical_camera=name,transform_matrix=pose.tolist()))
    monkeypatch.setattr(spiral,'cameras',lambda _: (rows,None,None))
    spiral._arc_length_map.cache_clear()
    try:
        u,angle,radius,xy,settle=spiral.spiral_parameters(np.arange(150))
        first=np.flatnonzero(u>=5/6)[0]
        assert first<100 and angle[0]-angle[first]>=2*np.pi
        assert radius[first]>.7
        assert np.ptp(xy[:first+1,0])>4.8 and np.ptp(xy[:first+1,1])>1.5
        assert xy[0,0]>1 and xy[0,1]>.5
        assert xy[:,0].min()>-2.7 and xy[:,0].max()<2.7
        # Arc table drops sub-1e-12 length steps; actual saved camera matrices
        # are explicitly snapped to exact H/C at118 and independently audited.
        np.testing.assert_allclose(xy[118:],0,atol=1e-9)
        np.testing.assert_allclose(settle[118:],1,atol=1e-5)
        speed=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        assert speed.max()<1.04*np.median(speed[15:70])
        assert speed[0]<.02*speed.max() and speed[117]<.01*speed.max()
    finally:
        spiral._arc_length_map.cache_clear()
