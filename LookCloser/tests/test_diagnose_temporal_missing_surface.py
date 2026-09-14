import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_temporal_missing_surface import classify


def test_stage_attribution_does_not_equate_background_with_carving():
    original=np.array([[np.inf,1.,1.,1.,0.]])
    final=np.array([[np.inf,np.inf,1.,1.,0.]])
    ids=np.array([[255,255,255,0,255]])
    masks=classify(original,final,ids)
    assert masks['removed_surface'].tolist()==[[False,True,False,False,False]]
    assert masks['geometry_without_rgb'].tolist()==[[False,False,True,False,False]]
    assert masks['original_miss'].tolist()==[[True,False,False,False,True]]
    assert masks['supported_render'].tolist()==[[False,False,False,True,False]]
