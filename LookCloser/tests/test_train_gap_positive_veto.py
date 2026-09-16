import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_train_gap_positive_veto import positive_samples


def test_any_foreground_neighbour_protects_but_background_depth_does_not():
    mask=np.zeros((4,4),bool);mask[2,2]=True
    xy=np.array([[1.2,1.2],[1.2,1.2],[0.,0.],[np.nan,2.],[2.,2.]])
    z=np.array([1.,2.,1.,1.,1.])
    np.testing.assert_array_equal(positive_samples(mask,xy,z,[.9,1.1]),[True,False,False,False,True])
