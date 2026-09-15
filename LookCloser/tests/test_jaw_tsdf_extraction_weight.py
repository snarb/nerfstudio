import importlib.util
from pathlib import Path
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from audit_jaw_tsdf_extraction_weight import check_command


def test_only_declared_control_parameters_change():
    base=['python','fuse.py','--output','old.ply','--tensor-weight-threshold','2','--voxel-length','.0005']
    actual=['python','fuse.py','--output','new.ply','--tensor-weight-threshold','0.5','--voxel-length','.0005']
    check_command(base,actual,Path('new.ply'),.5)
    actual[-1]='.001'
    with pytest.raises(AssertionError): check_command(base,actual,Path('new.ply'),.5)


def test_duplicate_control_flag_rejected():
    base=['--output','a','--tensor-weight-threshold','2']
    with pytest.raises(AssertionError):
        check_command(base,base+['--output','b'],'b',2.)
