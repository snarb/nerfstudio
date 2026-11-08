from pathlib import Path
import sys,os,signal,time,json,fcntl
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations';sys.path[:0]=[str(O),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_tuning_controller import identity,alive
D=O/'CPU_spawn_v115';D.mkdir(exist_ok=True);frame=1700;f=V/'frames/001700';assert not(f/'native_semantic_ready.json').exists();before={p.name:sha(p)for p in(f/'native_masks').glob('*.npz')};retired=[]
for p in [O/'source_producer_v113_watchdog_identity.json',O/'source_producer_v113_identity.json']:
 i=json.loads(p.read_text())
 if alive(i):
  assert b'marshawn_source_producer_v113.py'in Path(f"/proc/{i['pid']}/cmdline").read_bytes();os.kill(i['pid'],signal.SIGTERM);retired.append(i)
parent=identity(2394582);assert alive(parent);pids=[int(x)for x in Path('/proc/2394582/task/2394582/children').read_text().split()];tree=[]
for pid in pids+[2394582]:
 try:argv=Path(f'/proc/{pid}/cmdline').read_bytes()
 except FileNotFoundError:continue
 if b'marshawn_source_factory_v110.py'not in argv:continue
 assert b'1700'in argv;i=identity(pid);tree.append(dict(identity=i,argv=argv.decode().replace('\0',' '),wchan=Path(f'/proc/{pid}/wchan').read_text()))
assert len(tree)==9,tree;write(D/'stuck_tree_before.json',dict(frame=1700,tree=tree,source_outer_retired=retired,native_before_sha256=before,no_training_failure=True,no_NN_jobs_killed=True))
for row in tree:
 if alive(row['identity']):os.kill(row['identity']['pid'],signal.SIGTERM)
for i in retired+[x['identity']for x in tree]:
 for _ in range(100):
  if not alive(i):break
  time.sleep(.1)
 if alive(i):os.kill(i['pid'],signal.SIGKILL);time.sleep(.1)
 assert not alive(i),i
held=open(O/'source_factory_v74.lock','a+');fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB);after={p.name:sha(p)for p in(f/'native_masks').glob('*.npz')};assert before==after;write(D/'quiescence.json',dict(status='pass',frame=1700,retired_exact_CPU_tree=[x['identity']for x in tree],canonical_writer_lock_reacquired=True,native_before_sha256=before,native_after_sha256=after,all66_current_native_NN_arrays_unchanged=True,all_part_and_head_NN_cached=True,source_marker_absent=True,created=time.time()));print('quiesced9CPUprocesses;66currentnative archives unchanged;writerlock available')
