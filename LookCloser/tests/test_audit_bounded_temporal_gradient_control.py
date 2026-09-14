import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from joint_temporal_texture import atomic_json, sha
from bound_temporal_gradient_offset import run
from audit_bounded_temporal_gradient_control import audit


def test_independent_reproduction_and_tamper_rejection(tmp_path):
    source=tmp_path/'raw'; source.mkdir()
    before=np.full((16,20,3),120,np.uint8)
    Image.fromarray(before).save(source/'baseline.png')
    np.savez(source/'offset.npz',offset=np.full((3,16,20),-.9,np.float32),
             depth=np.ones((16,20)),selection=np.zeros((16,20)))
    atomic_json(source/'result.json',dict(hashes={p.name:sha(p) for p in source.iterdir()}))
    dest=tmp_path/'bounded'; run(source,dest); audit(dest)
    assert (dest/'bounded_audit.json').exists()
    Image.fromarray(np.zeros_like(before)).save(dest/'corrected.png')
    with pytest.raises(ValueError,match='Changed bounded artifact'):audit(dest)
