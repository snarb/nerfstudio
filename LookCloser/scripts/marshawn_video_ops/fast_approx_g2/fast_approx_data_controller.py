"""Simple four-frame CPU DATA queue for user-accepted approximate actor masks."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from types import SimpleNamespace
ROOT=Path('/fsx/tmp/marshawn_video_fast_masks_20261108');OPS=ROOT/'operations';FRAMES=list(range(1720,2016,5));PY='/home/pavlo/.conda/envs/nerfstudio/bin/python'
sys.path.insert(0,str(ROOT/'code/LookCloser/scripts'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def identity(pid=None):
 pid=os.getpid()if pid is None else pid;p=Path('/proc')/str(pid);s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split();return dict(pid=pid,start=f[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
def alive(r):
 try:
  now=identity(r['pid']);s=Path('/proc',str(r['pid']),'stat').read_text();return all(now[k]==r[k]for k in['start','boot','argv'])and s[s.rfind(')')+2:].split()[0]!='Z'
 except(Exception):return False

def prepare(frame):
 import numpy as np
 assert frame in FRAMES;F=ROOT/'frames'/f'{frame:06d}';data=F/'data';lock=open(F/'simple_DATA.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(F/'simple_DATA_identity.json',identity())
 marker=F/'native_semantic_ready.json'
 if marker.exists():
  m=load(marker);assert m['kind']=='FAST_APPROX_MASK_SOURCE_READY_V1'and sha(data/'prepared_ready.json')==m['prepared_ready_sha256'];return
 batch=F/'native_masks/batch_ready.json';b=load(batch);cameras=load(ROOT/'source_inventory.json')['cameras'];assert b['frame']==frame and sorted(b['cameras'])==sorted(cameras)and len(cameras)==66
 policy=load(OPS/'fast_approx_policy.json');code=OPS/'fast_approx_data_math.py';assert sha(code)==policy['data_math_code_sha256'];assert sha(policy['original_data_code'])==policy['original_data_code_sha256']
 native_inventory={}
 for camera in cameras:
  p=F/'native_masks'/(camera+'.npz');meta=p.with_suffix('.json');r=load(meta);source=ROOT/'sources'/str(frame)/camera/'receipt.json';assert r['frame']==frame and r['archive_sha256']==sha(p)and r['source_receipt_sha256']==sha(source)
  if 'records'in b:assert b['records'][camera]==sha(meta)
  native_inventory[camera]=dict(archive_sha256=sha(p),metadata_sha256=sha(meta),source_receipt_sha256=sha(source))
 # The unchanged AREA/remap/native-falloff implementation runs once per frame.
 # Empty actor handling and approximation labels are the only DATA behavior change.
 if(data/'prepared_ready.json').exists():
  binding=load(F/'source_mask_binding.json');assert all(next(x for x in binding['records']if x['camera']==c)['native_mask_sha256']==native_inventory[c]['archive_sha256']for c in cameras),'Unpublished DATA stale: refuse silent reuse'
 spec=importlib.util.spec_from_file_location('fast_approx_data_math',code);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.main(SimpleNamespace(root=ROOT,frame=frame,box=ROOT/'common_box.json',diagnostic_partial=False,cameras=[]))
 ready=load(data/'ready.json');prepared=load(data/'prepared_ready.json');tf=load(data/'transforms.json');assert ready['train']==66 and len(ready['records'])==66 and len(tf['train_filenames'])==66
 records=[];unavailable=[]
 for row in ready['records']:
  camera=row['camera'];source=load(ROOT/'sources'/str(frame)/camera/'receipt.json');image=data/row['image'];assert sha(image)==source['image_sha256']==row['rgb_sha256'];semantic=Path(row['semantic']);assert sha(semantic)==row['semantic_sha256']
  with np.load(semantic,allow_pickle=False)as z:
   for key in ['black','grey','core','band','valid','nominal','unknown','source_ineligible']:assert z[key].shape==(1012,1920)and z[key].dtype==np.bool_
   for key in ['W32','W64']:assert z[key].shape==(1012,1920)and np.isfinite(z[key]).all()and z[key].min()>=0 and z[key].max()<=1.000001
  for actor in ['black','grey']:
   assert row['visibility'][actor]in['present','approx_unavailable']
   if row['visibility'][actor]=='approx_unavailable':assert row['crops'][actor]is None;unavailable.append(dict(camera=camera,actor=actor))
  records.append(dict(camera=camera,source_rgb_sha256=row['rgb_sha256'],semantic_sha256=row['semantic_sha256'],native_archive_sha256=native_inventory[camera]['archive_sha256']))
 qa=data/'fast_approx_source_QA.json';write(qa,dict(kind='FAST_APPROX_SOURCE_ALIGNMENT_QA_V1',frame=frame,status='pass',cameras=66,checks=['current source RGB hashes','66 cameras and native source bindings','semantic shape and finite W32/W64','same original calibration/remap/box'],mask_accuracy='approximate; no geometric or semantic perfection claim',no_HEAD_CREW_PRE_dense_or_human_quality_claim=True,approx_unavailable=unavailable,records=records,ready_sha256=sha(data/'ready.json'),prepared_ready_sha256=sha(data/'prepared_ready.json'),policy_sha256=sha(OPS/'fast_approx_policy.json')))
 write(marker,dict(kind='FAST_APPROX_MASK_SOURCE_READY_V1',frame=frame,cameras=66,source_generation=2,source_ready_sha256=sha(ROOT/'sources'/str(frame)/'source_ready.json'),prepared_ready_sha256=sha(data/'prepared_ready.json'),semantic_ready_sha256=sha(data/'ready.json'),source_QA_sha256=sha(qa),source_QA=str(qa),policy_sha256=sha(OPS/'fast_approx_policy.json'),native_masks=native_inventory,common_box_sha256=sha(ROOT/'common_box.json'),mask_accuracy='APPROXIMATE_USER_ACCEPTED',approx_unavailable=unavailable,no_cross_frame_NeRF_state=True,original_W32_W64_AREA_single_remap=True,no_old_HEAD_CREW_PRE_or_dense_PASS_claims=True,created=time.time()))
 print(frame,'SOURCE_READY',sha(marker),flush=True)

def controller(workers):
 lock=open(OPS/'data_controller.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(OPS/'data_controller_identity.json',identity());jobs={};logs=OPS/'data_logs';logs.mkdir(exist_ok=True)
 while True:
  for frame,job in list(jobs.items()):
   if not alive(job):jobs.pop(frame)
  complete=[f for f in FRAMES if(ROOT/'frames'/f'{f:06d}'/'native_semantic_ready.json').exists()]
  if len(complete)==len(FRAMES):write(OPS/'data_complete.json',dict(frames=complete,time=time.time()));return
  for frame in FRAMES:
   if len(jobs)>=workers:break
   F=ROOT/'frames'/f'{frame:06d}'
   if frame in complete or frame in jobs or not(F/'native_masks/batch_ready.json').exists():continue
   ident=F/'simple_DATA_identity.json';old=load(ident)if ident.exists()else None
   if old and alive(old):jobs[frame]=old;continue
   with(logs/f'{frame}.log').open('ab')as out:p=subprocess.Popen([PY,__file__,'frame','--frame',str(frame)],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
   for _ in range(20):
    try:jobs[frame]=identity(p.pid);break
    except FileNotFoundError:time.sleep(.01)
  write(OPS/'data_state.json',dict(pid=os.getpid(),complete=len(complete),running=jobs,pending=len(FRAMES)-len(complete)-len(jobs),workers=workers,time=time.time()));time.sleep(10)

def watchdog(workers):
 lock=open(OPS/'data_watchdog.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(OPS/'data_watchdog_identity.json',identity())
 while not(OPS/'data_complete.json').exists():
  p=OPS/'data_controller_identity.json';old=load(p)if p.exists()else None
  if not old or not alive(old):
   with(OPS/'data_controller.log').open('ab')as out:subprocess.Popen([PY,__file__,'controller','--workers',str(workers)],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  time.sleep(10)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['frame','controller','watchdog']);parser.add_argument('--frame',type=int);parser.add_argument('--workers',type=int,default=4);a=parser.parse_args()
 try:prepare(a.frame)if a.action=='frame'else controller(a.workers)if a.action=='controller'else watchdog(a.workers)
 except Exception as e:
  if a.action=='frame':write(ROOT/'frames'/f'{a.frame:06d}'/'simple_DATA_error.json',dict(frame=a.frame,error=str(e),traceback=traceback.format_exc(),time=time.time()))
  raise
