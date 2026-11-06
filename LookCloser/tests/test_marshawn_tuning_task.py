import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_tuning_task import winner,resolve

def result(lpips,psnr=25,step=106316,**config):
    return dict(selection=dict(masked_lpips=lpips,psnr=psnr,step=step),config=config)

def test_selector_tolerance_and_skips():
    a=result(.20,25);b=result(.2005,26)
    assert winner([a,b,None,dict(skipped=True)]) is b
    assert winner([a,result(.202,30)]) is a

def test_resolution_does_not_inherit_remote_paths():
    parent=result(.2,objective='actor',sampling='body80',data='/remote/data',output='/remote/run')
    spec=resolve(dict(inherit='actor',overrides=dict(width=32)),{'p':parent},Path('/local/data'),Path('/local/run'))
    assert spec['data']=='/local/data' and spec['output']=='/local/run'
    assert spec['width']==32 and spec['sampling']=='body80'

def test_control_is_not_best_actor():
    a=result(.2,objective='actor',sampling='body80',radius=1)
    b=result(.3,objective='full',sampling='baseline',radius=0)
    spec=resolve(dict(inherit='control',overrides=dict(seed=43)),dict(a=a,b=b),Path('/data'),Path('/out'))
    assert spec['objective']=='full' and spec['seed']==43
