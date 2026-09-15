"""Check the independent path audit's per-frame gauge inversion."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('independent_cinematic_audit', SCRIPTS/'audit_cinematic_path_requests.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def test_gauge_inverse_does_not_scale_rotation_or_miss_applied_transform():
    theta=.23
    transform=np.eye(4);transform[:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
    transform[:3,3]=[1.3,-6.1,.4]
    applied=np.eye(4);applied[:3,:3]=[[0,1,0],[1,0,0],[0,0,-1]]
    applied[:3,3]=[.1,.2,.3]
    raw=np.eye(4);raw[:3,3]=[4,5,6]
    normalized=transform@np.linalg.inv(applied)@raw
    normalized[:3,3]*=.123
    metadata=dict(dataparser_scale=.123,dataparser_transform=transform[:3].tolist())
    calibration=dict(applied_transform=applied[:3].tolist())
    saved=normalized.copy()
    np.testing.assert_allclose(audit.raw_matrix(normalized,metadata,calibration),raw,atol=1e-12)
    np.testing.assert_array_equal(saved,normalized)
