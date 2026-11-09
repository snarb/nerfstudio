import importlib.util
from pathlib import Path
import sys
import numpy as np

scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('free_audit',scripts/'audit_marshawn_free_space.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_free_prefix_excludes_witness_surface_and_clips_partial_bin():
    starts=np.array([0.,1.,2.,3.]);ends=starts+1;sigma=np.array([1.,2.,1000.,1000.])
    assert audit.prefix_tau(starts,ends,sigma,1.5)==2.
    assert audit.prefix_tau(starts,ends,sigma,2.)==3.
    assert audit.prefix_tau(starts,ends,sigma,0.)==0.
