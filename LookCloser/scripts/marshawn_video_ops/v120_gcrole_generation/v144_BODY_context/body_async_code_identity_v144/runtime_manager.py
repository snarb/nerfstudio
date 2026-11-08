"""Durable additive adopter for the existing G1 BODY configs and controller locks."""
from pathlib import Path
import argparse,os,sys,time,subprocess,traceback
O=Path(__file__).parent;G=O.parent.parent;sys.path.insert(0,str(O))
from marshawn_native_async_collect import read,write,sha,identity,alive,lock,base_module

def verify():
 r=read(O/'release.json');assert r['authorized'] and r['parent_source_release_sha256']=='f74e81c7df9f000b790701371b978034dc3b69ea241d8806cecd0a9578b98153'
 for p,d in r['files_sha256'].items():assert sha(p)==d,('Released operational code changed',p)
 return r

def run():
 with lock(O/'manager.lock')as held:
  if held is None:return
  write(O/'manager_identity.json',identity());r=verify()
  while True:
   verify();records=[]
   for frame in range(1715,2016,5):
    c=G/'operations/sam3_assist/v5'/f'frame{frame:06d}'/'all66_config.json'
    if not c.exists():continue
    cfg=read(c);assert cfg['campaign']==str(G)and cfg['frame']==frame and len(cfg['records'])==66
    root=Path(cfg['root']);assert root==c.parent/'all66_controller'
    if(root/'all66_review_ready.json').exists():continue
    base=base_module(cfg);base.guard(cfg);ip=root/'watchdog_identity.json';old=read(ip)if ip.exists()else None
    if not alive(old):
     with(root/'watchdog_v144.log').open('ab')as out:p=subprocess.Popen([sys.executable,str(O/'marshawn_sam3_union_batch_async_g1_v144.py'),'watchdog','--config',str(c)],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
     records.append(dict(frame=frame,started=p.pid,configuration_sha256=sha(c),same_existing_controller_locks=True))
    else:records.append(dict(frame=frame,adopted_watchdog=old,configuration_sha256=sha(c)))
   write(O/'manager_state.json',dict(pid=os.getpid(),records=records,time=time.time(),no_config_task_or_source_mutation=True));time.sleep(5)

def watchdog():
 with lock(O/'manager_watchdog.lock')as held:
  if held is None:return
  write(O/'manager_watchdog_identity.json',identity())
  while True:
   verify();p=O/'manager_identity.json';old=read(p)if p.exists()else None
   if not alive(old):
    with(O/'manager.log').open('ab')as out:subprocess.Popen([sys.executable,__file__,'run'],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   time.sleep(10)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','watchdog']);a=p.parse_args()
 try:run()if a.action=='run'else watchdog()
 except Exception as e:write(O/'manager_error.json',dict(error=str(e),traceback=traceback.format_exc(),time=time.time()));raise
