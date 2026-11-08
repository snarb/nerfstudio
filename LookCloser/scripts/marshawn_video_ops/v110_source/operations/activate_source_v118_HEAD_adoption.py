from pathlib import Path
import json,sys,os,fcntl,signal,subprocess,time
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations';Q=O/'containment_v118_qualification';sys.path[:0]=[str(O),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_tuning_controller import alive,identity
import marshawn_source_producer_v118 as new
new.verify();lock=open(O/'source_CPU_handoff.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX);state=json.loads((O/'source_producer_v115_state.json').read_text());assert state['frame']==1705 and state['stage']=='current_HEAD_source_DATA_QA';receipt=O/'source_producer_v115_logs/1705_current_HEAD_source_DATA_QA.identity.json';worker=json.loads(receipt.read_text());assert alive(worker);assert b'marshawn_source_factory_v115.py'in Path(f"/proc/{worker['pid']}/cmdline").read_bytes();os.kill(worker['pid'],signal.SIGSTOP)
try:
 s=json.loads((O/'source_factory_v74_state.json').read_text());assert s['frame']==1705 and s['pid']==worker['pid']and s['stage']=='CURRENT_native_head_NN';head=O/'1705_head_ALL66_ROI_v73';jobs=json.loads((head/'controller/state.json').read_text())['jobs'];assert any(x['status']!='complete'for x in jobs.values());assert not(head/'all66_current_consensus_application.json').exists()and not list((head/'application_by_camera').glob('*.json'));f=V/'frames/001705';assert not(f/'native_semantic_ready.json').exists();before={p.name:sha(p)for p in(f/'native_masks').glob('*.npz')};head_owner=json.loads((head/'launch.json').read_text());assert Path(f"/proc/{head_owner['pid']}/cmdline").exists();retired=[]
 for p in [O/'source_producer_v115_watchdog_identity.json',O/'source_producer_v115_identity.json']:
  i=json.loads(p.read_text())
  if alive(i):assert b'marshawn_source_producer_v115.py'in Path(f"/proc/{i['pid']}/cmdline").read_bytes();os.kill(i['pid'],signal.SIGTERM);retired.append(i)
 os.kill(worker['pid'],signal.SIGTERM);os.kill(worker['pid'],signal.SIGCONT);retired.append(worker)
 for i in retired:
  for _ in range(100):
   if not alive(i):break
   time.sleep(.1)
  assert not alive(i)
 held=open(O/'source_factory_v74.lock','a+');fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB);after={p.name:sha(p)for p in(f/'native_masks').glob('*.npz')};assert before==after;assert Path(f"/proc/{head_owner['pid']}/cmdline").exists();env=dict(os.environ,PYTHONPATH=str(O)+':'+str(V/'code/LookCloser/scripts'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
 with(O/'source_producer_v118_watchdog.log').open('ab')as out:p=subprocess.Popen([sys.executable,str(O/'marshawn_source_producer_v118.py'),'watchdog'],stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
 write(Q/'activation.json',dict(status='pass',frame=1705,stage='safe_CURRENT_HEAD_NN_wait_CPU_handoff',retired_exact_CPU_owners=retired,preserved_live_HEAD_controller=head_owner,preserved_native_arrays_sha256=before,new_watchdog_pid=p.pid,new_source_release_sha256=sha(O/'source_producer_v118_release.json'),current1700_published_inputs_untouched=True,all_NN_field_frequency_and_prefetch_workers_preserved=True,created=time.time()));print('SOURCE118_WATCHDOG',p.pid,sha(O/'source_producer_v118_release.json'),flush=True)
except BaseException:
 if alive(worker):os.kill(worker['pid'],signal.SIGCONT)
 raise
