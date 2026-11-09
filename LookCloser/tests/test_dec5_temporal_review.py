import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_temporal_review import residual_metrics


def test_translation_and_real_appearance_change_are_not_flicker():
    rng=np.random.RandomState(4)
    gt=rng.uniform(.1,.7,(32,48,3)).astype(np.float32)
    next_gt=np.roll(gt,1,axis=1)+.05
    forward=np.zeros((32,48,2),np.float32);forward[...,0]=1
    backward=-forward
    # A constant model bias remains constant despite true motion/lighting change.
    result=residual_metrics(gt+.1,next_gt+.1,gt,next_gt,forward,backward)
    assert result['valid_pixels']>1000
    assert result['error_change_mae']<1e-7
    assert result['gt_change_mae']>.04


def test_flicker_is_detected_and_inconsistent_flow_is_excluded():
    gt=np.full((32,48,3),.4,np.float32)
    prediction=gt.copy();prediction[8:24,12:36]+=.2
    zero=np.zeros((32,48,2),np.float32)
    roi=np.zeros((32,48),bool);roi[8:24,12:36]=True
    result=residual_metrics(gt,prediction,gt,gt,zero,zero,roi)
    assert abs(result['error_change_mae']-.2)<1e-6
    assert result['gt_change_mae']==0
    inconsistent=zero.copy();inconsistent[...,0]=5
    result=residual_metrics(gt,prediction,gt,gt,zero,inconsistent,roi)
    assert result['valid_pixels']==0
    assert result['error_change_mae'] is None
