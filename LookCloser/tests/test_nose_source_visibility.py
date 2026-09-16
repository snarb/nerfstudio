import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from diagnose_nose_source_visibility import direct_visible


def test_continuous_visibility_excludes_occluders_and_misses():
    t=np.array([1,1-1e-6,1+1e-6,.999,1.001,np.inf,np.nan])
    assert direct_visible(t).tolist()==[True,True,True,False,False,False,False]
