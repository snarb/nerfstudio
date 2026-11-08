"""Durable finite SOURCE transaction; resumes exact held CPU parents only on PASS."""
from pathlib import Path
import os,sys,json,signal,subprocess,time,hashlib,fcntl,shutil,traceback
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');O=G/'operations/1715_FD_components_current_positive';F=G/'frames/001715';PY='/home/pavlo/.conda/envs/nerfstudio/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);t=p.with_suffix('.atomic_tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def ident(pid):
 p=Path(f'/proc/{pid}')
 if not p.exists():return None
 s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split();return dict(pid=pid,start=f[19],state=f[0],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),ppid=int(f[1]))
def descendants(pid):
 allp={}
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:r=ident(int(p.name));allp[r['pid']]=r
   except(Exception):continue
 found=set();todo=[pid]
 while todo:
  n=todo.pop()
  for r in allp.values():
   if r['ppid']==n and r['pid']not in found:found.add(r['pid']);todo.append(r['pid'])
 return[allp[n]for n in sorted(found)]
def state(stage,**kw):write(O/'transaction_state.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))
def main():
 O.mkdir(exist_ok=True);lock=open(O/'transaction.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a=load(O/'ROOT_application_authorization.json');assert a['authorized']and sha(G/'operations/apply_FD1715_current_photo_components.py')==a['applier_sha256'];assert not(F/'native_semantic_ready.json').exists()
 held=load(G/'operations/1715_FD8_current_source_quality_hold.json');worker=next(r for r in held['records']if r['pid']==2596253);r=ident(worker['pid']);children=descendants(worker['pid'])if r else[]
 phase=load(G/'operations/source_factory_v74_state.json');assert phase['pid']==worker['pid']and phase['stage']=='current_source_identity_and_visibility_guard'
 for child in children:
  if child['state']=='Z':continue
  assert child['ppid']==worker['pid']and ('multiprocessing.spawn' in child['argv']or'multiprocessing.resource_tracker'in child['argv']),'Unexpected non-CPU-guard descendant'
  now=ident(child['pid']);assert now and all(now[k]==child[k]for k in['start','boot','argv']);os.kill(child['pid'],signal.SIGKILL)
 for _ in range(100):
  if not[x for x in descendants(worker['pid'])if x['state']!='Z']:break
  time.sleep(.1)
 else:raise RuntimeError('Exact CPU guard descendants did not exit')
 if r and r['state']!='Z':
  assert all(r[k]==worker[k]for k in['start','boot','argv'])and r['state']=='T';os.kill(r['pid'],signal.SIGKILL)
  for _ in range(100):
   check=ident(r['pid'])
   if check is None or check['state']=='Z':break
   time.sleep(.1)
  else:raise RuntimeError('Exact SOURCE worker did not exit')
 write(O/'worker_retirement.json',dict(worker=worker,descendants=children,all_live_writing_descendants_absent=True,time=time.time(),no_fields_NN_frequency_cache_stopped=True))
 # Prove the canonical lock released naturally before any correction or archive.
 l=open(G/'operations/source_factory_v74.lock','a+');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(l,fcntl.LOCK_UN);l.close()
 backup=O/'unpublished_downstream_before';backup.mkdir(exist_ok=True);archived=[]
 for p in [F/'data/admission_unknown_hair',F/'crew_guard',F/'visibility_unknown_inputs',F/'native_review',F/'source_mask_binding.json',G/'operations/1715_source_factory_v74_REVIEW_READY.json']+[F/'data'/n for n in['prepared_ready.json','ready.json','final_v5_source_QA.json','visibility_unknown_QA_v74.json','transforms.json','box.json','semantics','masks','review']]:
  if p.exists():
   dest=backup/(str(len(archived))+'_'+p.name)
   if dest.exists():raise RuntimeError('Archived path already exists: '+str(dest))
   shutil.move(str(p),str(dest));archived.append(dict(source=str(p),archived=str(dest)))
 write(O/'unpublished_downstream_archive.json',dict(records=archived,time=time.time(),no_RGB_or_frequency_files_changed=True))
 state('applying_exact_current_NN_scope')
 with(O/'application.log').open('ab')as out:p=subprocess.Popen([PY,str(G/'operations/apply_FD1715_current_photo_components.py'),'--apply'],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2'))
 state('applying_exact_current_NN_scope',child_pid=p.pid);rc=p.wait();assert rc==0,'Finite applier failed; source parents remain held'
 done=load(O/'application_complete.json');assert done['status']=='pass'and done['exact_new_actor_pixels']==19881  and sha(F/'native_masks/FD.npz')==done['new_native_sha256'];assert done['all_native_arrays_exact_expected']and done['physical_UNKNOWN_unchanged'];assert not(F/'native_semantic_ready.json').exists()
 for path,digest in done['all65_other_native_archives_unchanged'].items():assert sha(path)==digest
 resumed=[]
 for parent in held['records']:
  if parent['pid']==2596253:continue
  r=ident(parent['pid']);assert r is not None and all(r[k]==parent[k]for k in['start','boot','argv'])and r['state']=='T';os.kill(r['pid'],signal.SIGCONT);resumed.append(r)
 state('application_PASS_exact_SOURCE_parents_resumed',application_complete_sha256=sha(O/'application_complete.json'),resumed=resumed);write(O/'transaction_complete.json',dict(status='pass',application_complete_sha256=sha(O/'application_complete.json'),resumed=resumed,time=time.time(),fresh_all66_SOURCE_QA_and_grid_still_required=True))
if __name__=='__main__':
 try:main()
 except Exception as e:state('FAILED_SOURCE_HELD',error=str(e),traceback=traceback.format_exc());raise
