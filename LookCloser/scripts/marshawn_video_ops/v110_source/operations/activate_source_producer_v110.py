from pathlib import Path
import sys,json,os,fcntl,signal,subprocess,time
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations';sys.path[:0]=[str(O),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_tuning_controller import alive
old=json.loads((O/'source_producer_v108_release.json').read_text());files=dict(old['files_sha256'])
for name in ['marshawn_source_producer_v110.py','marshawn_source_factory_v110.py','marshawn_visibility_unknown_v110.py','prepare_FD1695_deep27_uncertainty_v110.py']:files[str(O/name)]=sha(O/name)
approval=O/'root_1695_FD2_deep27_unresolved_proposal_approval.json';assert sha(approval)=='e923832a415601cf103a89c6898f898b1335438bbbe5421e45972ddf1c155409'
write(O/'source_producer_v110_release.json',dict(old,kind='ROOT_SCOPED_FD1695_RAW_PROPOSAL_UNCERTAINTY_V110',files_sha256=files,previous_release_sha256=sha(O/'source_producer_v108_release.json'),ROOT_scope_approval_sha256=sha(approval),generic32_threshold_unchanged=True,all_actual_source_arrays_unchanged_by_guard_accounting=True,created=time.time()))
lock=open(O/'source_factory_v74.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
from prepare_FD1695_deep27_uncertainty_v110 import prepare
proof=prepare();retired=[];state=json.loads((O/'source_producer_v108_state.json').read_text());ip=O/'source_producer_v108_logs'/f"{state['frame']}_{state['stage']}.identity.json"
if ip.exists():
 identity=json.loads(ip.read_text())
 if alive(identity):
  assert b'marshawn_source_factory_v108.py'in Path(f"/proc/{identity['pid']}/cmdline").read_bytes();os.kill(identity['pid'],signal.SIGTERM);retired.append(identity)
for name in ['source_producer_v108_watchdog_identity.json','source_producer_v108_identity.json']:
 p=O/name
 if p.exists():
  identity=json.loads(p.read_text())
  if alive(identity):assert b'marshawn_source_producer_v108.py'in Path(f"/proc/{identity['pid']}/cmdline").read_bytes();os.kill(identity['pid'],signal.SIGTERM);retired.append(identity)
for identity in retired:
 for _ in range(100):
  if not alive(identity):break
  time.sleep(.1)
 assert not alive(identity)
env=dict(os.environ,PYTHONPATH=str(O)+':'+str(V/'code/LookCloser/scripts'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
with(O/'source_producer_v110_watchdog.log').open('ab')as log:p=subprocess.Popen([sys.executable,str(O/'marshawn_source_producer_v110.py'),'watchdog'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
write(O/'source_producer_v110_handoff.json',dict(retired_exact_CPU_owners=retired,new_watchdog_pid=p.pid,release_sha256=sha(O/'source_producer_v110_release.json'),scope_guard_proof_sha256=sha(V/'frames/001695/guard_proposal_uncertainty_inputs/FD/proof.json'),no_NN_field_frequency_or_published_source_modified=True,canonical_writer_lock_held=True,created=time.time()));print('SOURCE_V110_WATCHDOG',p.pid,flush=True)
