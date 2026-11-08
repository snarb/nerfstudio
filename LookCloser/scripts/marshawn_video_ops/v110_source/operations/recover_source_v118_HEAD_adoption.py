from pathlib import Path
import json,sys,os,fcntl,subprocess,time
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations';Q=O/'containment_v118_qualification';sys.path[:0]=[str(O),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_tuning_controller import alive
import marshawn_source_producer_v118 as new
new.verify();retired=[]
for p in [O/'source_producer_v115_watchdog_identity.json',O/'source_producer_v115_identity.json',O/'source_producer_v115_logs/1705_current_HEAD_source_DATA_QA.identity.json']:
 i=json.loads(p.read_text());assert not alive(i);retired.append(i)
held=open(O/'source_factory_v74.lock','a+')
for n in range(100):
 try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB);break
 except BlockingIOError:time.sleep(.1)
else:raise RuntimeError('Source mutex still genuinely held')
f=V/'frames/001705';assert not(f/'native_semantic_ready.json').exists();installed=json.loads((f/'union_v5_installed.json').read_text());native={}
for r in installed['records']:
 p=f/'native_masks'/(r['camera']+'.npz');assert sha(p)==r['new'];native[r['camera']]=sha(p)
head=O/'1705_head_ALL66_ROI_v73';owner=json.loads((head/'launch.json').read_text());assert Path(f"/proc/{owner['pid']}/cmdline").exists();assert not(head/'all66_current_consensus_application.json').exists()
env=dict(os.environ,PYTHONPATH=str(O)+':'+str(V/'code/LookCloser/scripts'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
with(O/'source_producer_v118_watchdog.log').open('ab')as out:p=subprocess.Popen([sys.executable,str(O/'marshawn_source_producer_v118.py'),'watchdog'],stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
write(Q/'activation.json',dict(status='pass',frame=1705,stage='CURRENT_HEAD_NN_wait_CPU_handoff',retired_exact_CPU_owners=retired,preserved_live_HEAD_controller=owner,all66_native_match_original_BODY_installation=native,body_installation_receipt_sha256=sha(f/'union_v5_installed.json'),canonical_writer_lock_reacquired=True,initial_activation_source_lock_release_transient='FLOCK EAGAIN immediately after exact old CPU exit; retry recovered, no source data or NN failure',new_watchdog_pid=p.pid,new_source_release_sha256=sha(O/'source_producer_v118_release.json'),current1700_published_inputs_untouched=True,all_NN_field_frequency_and_prefetch_workers_preserved=True,created=time.time()));print('SOURCE118_WATCHDOG',p.pid,sha(O/'source_producer_v118_release.json'))
