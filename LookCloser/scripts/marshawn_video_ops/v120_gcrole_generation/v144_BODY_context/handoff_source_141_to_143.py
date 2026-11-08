"""Exact pre-NN CPU handoff; preserves every source/model/cache artifact."""
from pathlib import Path
import os,sys,json,time,signal,subprocess,hashlib,fcntl,shutil,traceback
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');O=G/'operations/body_g1_v140';R=G/'operations/sam3_assist/v5/frame001715';PY='/home/pavlo/.conda/envs/nerfstudio/bin/python'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);t=p.with_suffix('.atomic_tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def ident(pid):
 p=Path(f'/proc/{pid}')
 if not p.exists():return None
 s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split();return dict(pid=pid,ppid=int(f[1]),state=f[0],start=f[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
def allp():
 out={}
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:r=ident(int(p.name));out[r['pid']]=r
   except Exception:pass
 return out
def same(a,b):return all(a[k]==b[k]for k in['pid','start','boot','argv'])
def main():
 held=open(O/'handoff143.lock','a+');fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB);release=G/'operations/source_producer_g1_v143_release.json';assert sha(release)=='f74e81c7df9f000b790701371b978034dc3b69ea241d8806cecd0a9578b98153'and load(release)['authorized'];marker=G/'frames/001710/native_semantic_ready.json';assert sha(marker)=='275a2b5a746b8ef4bc7a89f357897758bc08f3dae99d9a03d05895b63271fc0d'
 assert not(R/'all66_config.json').exists()and not(R/'all66_controller/state.json').exists();task_inventory={str(p):sha(p)for folder in['tasks','tasks_local']for p in(R/folder).glob('*/*')if p.is_file()and not p.is_symlink()};assert len(list((R/'tasks').glob('*/task.json')))==66 and len(list((R/'tasks_local').glob('*/task.json')))==66
 roots=[]
 for name in ['source_producer_g1_v141_watchdog_identity.json','source_producer_g1_v141_identity.json','source_producer_g1_v141_logs/1715_native_body_current.identity.json']:
  expected=load(G/'operations'/name);actual=ident(expected['pid']);assert actual and actual['start']==expected['start']and actual['boot']==expected['boot'];assert any(x in actual['argv']for x in['marshawn_source_producer_g1_v141.py','marshawn_mask_assist_entry_g1_v140.py']);os.kill(actual['pid'],signal.SIGSTOP);roots.append(actual)
 # Freeze the complete exact local BODY CPU descendant tree before retirement.
 targets={x['pid']:x for x in roots}
 for _ in range(5):
  scan=allp();changed=False
  for r in scan.values():
   if r['ppid']in targets and r['pid']not in targets:
    assert r['state']=='Z'or any(x in r['argv']for x in['marshawn_mask_assist_full_g1_v140.py','marshawn_mask_assist_entry_g1_v140.py','multiprocessing.spawn','multiprocessing.resource_tracker','marshawn_native_frequency_drain.py']),r['argv']
    targets[r['pid']]=r;changed=True
    if r['state']!='Z':os.kill(r['pid'],signal.SIGSTOP)
  if not changed:break
 for r in targets.values():
  now=ident(r['pid'])
  if now and now['state']!='Z':assert same(r,now)and now['state']=='T';os.kill(r['pid'],signal.SIGKILL)
 for _ in range(100):
  alive=[r for r in targets.values()if ident(r['pid'])and ident(r['pid'])['state']!='Z']
  if not alive:break
  time.sleep(.1)
 else:raise RuntimeError('Exact old CPU owners did not exit')
 survivors=[r for r in allp().values()if r['state']!='Z'and str(G)in r['argv']and'--frame 1715'in r['argv']and any(x in r['argv']for x in['marshawn_mask_assist_full_g1_v140.py','marshawn_mask_assist_entry_g1_v140.py','marshawn_mask_assist_entry_g1_v140.py'])];assert not survivors,survivors
 for path in [G/'operations/source_factory_v74.lock',R/'factory.lock']:
  l=open(path,'a+');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(l,fcntl.LOCK_UN);l.close()
 archive=O/('retired_preNN_1715_controls_143_'+str(time.time_ns()));archive.mkdir()
 for name in['factory_identity.json','factory_watchdog_identity.json','factory.log','factory_watchdog.log','BODY_pack_CPU8_execution.json']:
  path=R/name
  if path.exists():shutil.move(str(path),str(archive/name))
 for path,digest in task_inventory.items():assert sha(path)==digest
 write(archive/'all132_task_input_hashes_preserved.json',task_inventory)
 assert sha(marker)=='275a2b5a746b8ef4bc7a89f357897758bc08f3dae99d9a03d05895b63271fc0d'
 receipt=dict(kind='ROOT_AUTHORIZED_ADDITIVE_SOURCE_BODY_IMPLEMENTATION_IDENTITY_HANDOFF',old_exact_CPU_owners=list(targets.values()),all_old_CPU_owners_dead_or_zombie=True,no_1715_NN_config_or_execution_existed=True,all132_existing_task_input_hashes_preserved=task_inventory,archived_failed_preNN_factory=str(archive),source1710_marker_unchanged=sha(marker),new_release_sha256=sha(release),parent_release_sha256=sha(G/'operations/source_producer_g1_v141_release.json'),no_fields_NN_frequency_cache_workers_stopped=True,time=time.time());write(O/'handoff143_retirement.json',receipt)
 with(G/'operations/source_producer_g1_v143_watchdog.log').open('ab')as out:p=subprocess.Popen([PY,str(G/'operations/marshawn_source_producer_g1_v143.py'),'watchdog'],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 write(O/'handoff143_launched.json',dict(watchdog_pid=p.pid,release_sha256=sha(release),retirement_receipt_sha256=sha(O/'handoff143_retirement.json'),time=time.time()));print('LAUNCHED_SOURCE143_WATCHDOG',p.pid,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write(O/'handoff143_error.json',dict(error=str(e),traceback=traceback.format_exc(),time=time.time()));raise
