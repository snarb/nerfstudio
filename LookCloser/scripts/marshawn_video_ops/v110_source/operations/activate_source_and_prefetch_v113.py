from pathlib import Path
import sys,json,os,fcntl,signal,subprocess,time,shutil
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations';sys.path[:0]=[str(O),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_tuning_controller import alive
import marshawn_source_producer_v113 as p
import marshawn_native_prefetch_v113 as n
p.verify();n.check_release()
lock=open(O/'source_factory_v74.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
retired=[]
for receipt,expected in [(O/'source_producer_v112_watchdog_identity.json','marshawn_source_producer_v112.py'),(O/'source_producer_v112_identity.json','marshawn_source_producer_v112.py'),(O/'native_prefetch_v112/watchdog_identity.json','marshawn_native_prefetch_v112.py'),(O/'native_prefetch_v112/controller_identity.json','marshawn_native_prefetch_v112.py')]:
 if receipt.exists():
  i=json.loads(receipt.read_text())
  if alive(i):
   assert expected.encode()in Path(f"/proc/{i['pid']}/cmdline").read_bytes();os.kill(i['pid'],signal.SIGTERM);retired.append(dict(receipt=str(receipt),identity=i))
for row in retired:
 for _ in range(100):
  if not alive(row['identity']):break
  time.sleep(.1)
 assert not alive(row['identity'])
adopted=[];newlogs=O/'source_producer_v113_logs';newlogs.mkdir(exist_ok=True)
for path in (O/'source_producer_v112_logs').glob('*.identity.json'):
 i=json.loads(path.read_text())
 if alive(i):
  assert 'native_body_current' in path.name,('unexpected source writer active under owned lock',path)
  shutil.copyfile(path,newlogs/path.name);adopted.append(dict(namespace='source',receipt=str(path),identity=i))
for path in (O/'native_prefetch_v112').glob('*_worker_identity.json'):
 r=json.loads(path.read_text())
 if alive(r['identity']):shutil.copyfile(path,O/'native_prefetch_v113'/path.name);adopted.append(dict(namespace='prefetch',receipt=str(path),identity=r['identity']))
env=dict(os.environ,PYTHONPATH=str(O)+':'+str(V/'code/LookCloser/scripts'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2');started=[]
for cli,log in [(O/'marshawn_source_producer_v113.py',O/'source_producer_v113_watchdog.log'),(O/'marshawn_native_prefetch_v113.py',O/'native_prefetch_v113/watchdog.log')]:
 with log.open('ab')as out:q=subprocess.Popen([sys.executable,str(cli),'watchdog'],stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
 started.append(dict(cli=str(cli),pid=q.pid))
write(O/'source_prefetch_v113_handoff.json',dict(kind='JOINT_SOURCE_PREFETCH_NEXT_UNSTARTED_BODY_CPU8_HANDOFF',retired_exact_CPU_parents=retired,adopted_live_native_waiters=adopted,started=started,source_release_sha256=sha(O/'source_producer_v113_release.json'),prefetch_release_sha256=sha(O/'native_prefetch_v113/release.json'),canonical_source_writer_lock_held=True,all_existing_NN_fields_and_inputs_preserved=True,created=time.time()));print(json.dumps(started))
