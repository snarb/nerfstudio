import importlib.util
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('eval4k',Path(__file__).parents[1]/'scripts/build_marshawn_4k_eval_diagnostic.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_eval_gain_uses_frozen_prediction_only():
 fit=dict(uses_eval_RGB=False,k=4,camera_rows=[{'camera':'AA'}],eval_predictions={'DC':{'ev':-.1331}})
 assert m.eval_gain(fit,'DC')==-.1331
 with pytest.raises(ValueError):m.eval_gain(fit,'AA')
 fit['uses_eval_RGB']=True
 with pytest.raises(ValueError):m.eval_gain(fit,'DC')
def test_pixel_centres_scale_before_raster_offset():
 a=dict(w=1920,h=1012,fl_x=1355.3,fl_y=1353.7,cx=960.,cy=506.)
 m.verify_rays(a,{k:v*2 for k,v in a.items()})
 wrong={k:v*2 for k,v in a.items()};wrong['cx']-=.5
 with pytest.raises(ValueError):m.verify_rays(a,wrong)
