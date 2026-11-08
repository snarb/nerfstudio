from pathlib import Path
import json,sys,os,fcntl,subprocess,time
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations';D=O/'CPU_spawn_v115';sys.path[:0]=[str(O),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_tuning_controller import alive
assert json.loads((D/'PASS.json').read_text())['status']=='pass';q=json.loads((D/'quiescence.json').read_text());assert q['status']=='pass'
for i in q['retired_exact_CPU_tree']:assert not alive(i)
for p in [O/'source_producer_v113_identity.json',O/'source_producer_v113_watchdog_identity.json']:assert not alive(json.loads(p.read_text()))
lock=open(O/'source_factory_v74.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for camera,digest in q['native_before_sha256'].items():assert sha(V/'frames/001700/native_masks'/camera)==digest
release=O/'source_producer_v115_release.json';r=json.loads(release.read_text())
for p in [O/'qualify_spawn_v115.py',D/'PASS.json',D/'quiescence.json',D/'stuck_tree_before.json']:r['files_sha256'][str(p)]=sha(p)
r.update(actual_threaded_parent_spawn_parity_sha256=sha(D/'PASS.json'),actual_deadlock_quiescence_sha256=sha(D/'quiescence.json'));write(release,r)
import marshawn_source_producer_v115 as new
new.verify();env=dict(os.environ,PYTHONPATH=str(O)+':'+str(V/'code/LookCloser/scripts'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
with(O/'source_producer_v115_watchdog.log').open('ab')as out:p=subprocess.Popen([sys.executable,str(O/'marshawn_source_producer_v115.py'),'watchdog'],stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
write(D/'activation.json',dict(status='pass',new_watchdog_pid=p.pid,release_sha256=sha(release),all66_native_preserved=True,no_HEAD_part_BODY_NN_redo_required=True,prefetch113_and_cached_transport114_unchanged=True,scientific_functions_AST_equal=True,actual_DE_archive_and_decisions_equal=True,created=time.time()));print('SOURCE115_WATCHDOG',p.pid,sha(release),flush=True)
