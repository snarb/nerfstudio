"""Saved-code tolerance cannot conceal changed image shape, dtype or >1-code drift."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('frprobe',Path(__file__).parents[1]/'scripts/probe_marshawn_FR_boundary.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
def test_single_code_rounding(dtype):
 a=np.full((1000,1000),10,dtype=dtype);b=a.copy();b[0,1]+=1
 r=m.compare_code_images(a,b);assert r['restored_max_code_error']==1 and r['restored_changed_values']==1
 assert m.compare_code_images(a,a)['restored_bitwise_equal']
@pytest.mark.parametrize('other',[np.array([[10,22]],np.uint8),np.array([10,20],np.uint8),np.array([[10,20]],np.uint16)])
def test_drift_or_shape_dtype_rejected(other):
 with pytest.raises(ValueError):m.compare_code_images(np.array([[10,20]],np.uint8),other)

def test_changed_fraction_rejected():
 for dtype in [np.uint8,np.uint16]:
  a=np.zeros((10,10),dtype=dtype);b=a.copy();b[0,0]=1
  with pytest.raises(ValueError):m.compare_code_images(a,b)
