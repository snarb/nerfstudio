"""One-frame-ahead isolated current NN; no live-mask installation/publication."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time,traceback
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_tuning_controller import identity,alive
from marshawn_source_producer_v76 import continuous_gate
O=V/'operations/native_prefetch_v113';O.mkdir(exist_ok=True)
def load(p):return json.loads(Path(p).read_text())
def state(frame,stage,**kw):write(O/'state.json',dict(frame=frame,stage=stage,pid=os.getpid(),time=time.time(),**kw))
def proof(previous):
 f=V/'frames'/f'{previous:06d}';guard=f/'crew_guard/plan_receipt.json';head=f/'data/admission_unknown_hair/manifest.json'
 if not guard.exists()or not head.exists():return None
 g=load(guard);h=load(head)
 if g['status']!='pass'or g['cameras']!=66 or h['frame']!=previous or len(h['records'])!=66:return None
 rows=[]
 for row in h['records']:
  camera=row['camera'];p=f/'native_masks'/(camera+'.npz');meta=p.with_suffix('.json');m=load(meta);r=load(f/'crew_guard'/(camera+'_people.json'));digest=sha(p)
  if r['status']!='pass'or r['current_native_archive_sha256']!=digest or m['archive_sha256']!=digest or row['native_actor_archive_sha256']!=digest:return None
  source=V/'sources'/str(previous)/camera/'receipt.json'
  if row['source_receipt_sha256']!=sha(source)or m['source_receipt_sha256']!=sha(source):return None
  rows.append(dict(camera=camera,native_sha256=digest,native_metadata_sha256=sha(meta),guard_sha256=sha(f/'crew_guard'/(camera+'_people.json')),source_receipt_sha256=sha(source)))
 return dict(previous_frame=previous,guard_sha256=sha(guard),current_head_manifest_sha256=sha(head),records=rows,all66_current_native_and_head_guard_proven=True,previous_pixels_only_2D_identity_not_final_FG=True)
def check_release():
 r=load(O/'release.json');assert r['authorized']
 for p,digest in r['files_sha256'].items():assert sha(p)==digest,p
 return r

def controller():
 lock=open(O/'controller.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(O/'controller_identity.json',identity());release=check_release()
 for frame in range(1690,2016,5):
  while True:
   try:
    check_release();ready=V/'operations/sam3_assist/v5'/f'frame{frame:06d}'/'all66_controller/all66_review_ready.json'
    if ready.exists():break
    if not continuous_gate():state(frame,'wait_ROOT_continuous_release');time.sleep(10);continue
    p=proof(frame-5)
    if p is None:state(frame,'wait_previous_actual66native_HEAD_guard_PASS');time.sleep(10);continue
    write(O/f'{frame}_current_2D_identity_qualification.json',p);argv=[sys.executable,str(V/'operations/sam3_assist/code/marshawn_native_route_v113.py'),'--campaign',str(V),'--frame',str(frame)];log=O/f'{frame}_isolated_native.log';state(frame,'isolated_CURRENT_native_NN',argv=argv,previous_guard_proof_sha256=sha(O/f'{frame}_current_2D_identity_qualification.json'))
    adoption=O/f'{frame}_worker_identity.json'
    previous=load(adoption) if adoption.exists() else None
    if previous and alive(previous['identity']):
     while alive(previous['identity']):state(frame,'adopt_existing_CURRENT_native_owner',worker_pid=previous['identity']['pid']);time.sleep(10)
     if ready.exists():break
    with log.open('ab')as out:worker=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    write(O/f'{frame}_worker_identity.json',dict(identity=identity(worker.pid),argv=argv))
    while worker.poll()is None:state(frame,'isolated_CURRENT_native_NN',worker_pid=worker.pid,previous_guard_proof_sha256=sha(O/f'{frame}_current_2D_identity_qualification.json'));time.sleep(10)
    if worker.returncode:raise RuntimeError('Native prefetch worker exited '+str(worker.returncode))
    assert ready.exists();state(frame,'isolated_native_complete_sourceproducer_owns_alias_install_HEAD_and_DATA',ready_sha256=sha(ready));break
   except Exception as e:write(O/'error.json',dict(frame=frame,error=str(e),traceback=traceback.format_exc(),time=time.time()));time.sleep(30)
 write(O/'complete.json',dict(frames=list(range(1690,2016,5)),no_live_install_DATA_source_marker_field_or_learned_state=True,finished=time.time()))

def watchdog():
 lock=open(O/'watchdog.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(O/'watchdog_identity.json',identity())
 while not(O/'complete.json').exists():
  check_release();p=O/'controller_identity.json';old=load(p)if p.exists()else None
  if not alive(old):
   with(O/'controller.log').open('ab')as out:subprocess.Popen([sys.executable,__file__,'controller'],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  time.sleep(10)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['controller','watchdog']);a=p.parse_args();controller()if a.action=='controller'else watchdog()
