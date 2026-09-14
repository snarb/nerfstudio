import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_camera_grid_flight import spline_weights,grid_path


def test_spline_containment_and_full_span():
    w,uv=spline_weights(1000)
    assert (w>=0).all()
    np.testing.assert_allclose(w.sum(1),1)
    np.testing.assert_allclose(np.ptp(uv,axis=0),[.96,.96],atol=1e-5)
    a,b=spline_weights(1001,endpoint=True)
    np.testing.assert_allclose(a[0],a[-1],atol=1e-12)


def test_grid_path_includes_closing_arc_without_velocity_jump():
    rows=[]
    for h in 'FGHI':
        for v in 'ABCD':
            p=np.eye(4);p[:3,3]=[(ord(h)-ord('H'))*.02,(ord(v)-ord('C'))*.03,1]
            rows.append({'physical_camera':h+'004_'+v+'005_'+('1210SZ' if (h,v)==('H','C') else 'test'),
                         'transform_matrix':p.tolist()})
    for size,count in [(3,480),(4,720)]:
        path,report=grid_path(rows,np.zeros(3),size,count)
        assert len(path)==count
        assert report['linear_speed_ratio']<1.001
        assert min(report['achieved_grid_interval_extent_xy'])>.95*(size-1)
