import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_temporal_full_block_control import transfer_mesh_gauge


def test_gauge_transfer_preserves_underlying_calibration_points():
    raw=np.array([[1.,2.,3.],[-3.,1.,4.]])
    a=np.eye(4);a[:3,3]=[.1,.2,.3]
    b=np.eye(4);b[:3,:3]=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]];b[:3,3]=[-2.,3.,1.]
    source={'dataparser_transform':a[:3].tolist(),'dataparser_scale':.1}
    target={'dataparser_transform':b[:3].tolist(),'dataparser_scale':.3}
    v=(raw@a[:3,:3].T+a[:3,3])*.1
    expected=(raw@b[:3,:3].T+b[:3,3])*.3
    actual=transfer_mesh_gauge(v,source,target)
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    np.testing.assert_allclose(transfer_mesh_gauge(actual,target,source),v,atol=1e-12)
