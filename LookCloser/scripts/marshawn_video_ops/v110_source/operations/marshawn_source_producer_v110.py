"""Chronological crash-adopting source queue; no learned model state transfer."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time,traceback
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_video_producer_v3 import native_inventory
from marshawn_v5_input_install import install
from marshawn_tuning_controller import identity,alive

def load(p):return json.loads(Path(p).read_text())
def status(frame,stage,**kw):write(V/'operations/source_producer_v110_state.json',dict(pid=os.getpid(),frame=frame,stage=stage,checked_at=time.time(),**kw))
def verify():
 release=load(V/'operations/source_producer_v110_release.json');assert release['authorized']
 for p,digest in release['files_sha256'].items():
  if sha(p)!=digest:raise ValueError('Released source helper changed '+p)
 policy=load(V/'operations/head_hair_policy_v74.json')
 required=['maximum_overlap_expansion_rounds','geometry_UNKNOWN_radius_native','positive_min_score','weak_min_score','minimum_existing_actor_overlap_fraction','minimum_semantic_pair_IoU','checkpoint_sha256','frames','authorized']
 if any(k not in policy for k in required):raise ValueError('Head policy schema incomplete BEFORE NN: '+str(required))
 assert policy['maximum_overlap_expansion_rounds']==2 and policy['geometry_UNKNOWN_radius_native']==128
 return release

def command(frame,stage,argv):
 root=V/'operations/source_producer_v110_logs';root.mkdir(exist_ok=True);receipt=root/f'{frame}_{stage}.identity.json';log=root/f'{frame}_{stage}.log';status(frame,stage,argv=argv)
 old=load(receipt)if receipt.exists()else None
 if old and alive(old):
  while alive(old):status(frame,stage,adopted_worker=old);time.sleep(10)
  return
 env=dict(os.environ,PYTHONPATH=str(V/'operations')+':'+str(V/'code/LookCloser/scripts'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
 with log.open('ab')as out:p=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(50):
  try:
   stat=Path(f'/proc/{p.pid}/stat').read_text();start=stat[stat.rfind(')')+2:].split()[19];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();write(receipt,dict(pid=p.pid,start=start,boot=boot,argv=argv));break
  except FileNotFoundError:time.sleep(.01)
 while p.poll()is None:status(frame,stage,worker_pid=p.pid);time.sleep(10)
 if p.returncode:raise RuntimeError(f'{stage} worker failed {p.returncode}; {log}')

def continuous_gate():
 p=V/'operations/v5_1680_final_visual_release.json'
 if not p.exists():return False
 g=load(p)
 if not g.get('authorized')or not g.get('continuous_authorized')or g.get('kind')!='ROOT_REVIEWED_V5_1680_AUTONOMOUS_SOURCE_RELEASE':return False
 for path,key in [(V/'frames/001680/native_semantic_ready.json','native_semantic_ready_sha256'),(V/'frames/001680/data/prepared_ready.json','prepared_ready_sha256')]:assert sha(path)==g[key]
 for row in g['visual_evidence']:assert sha(row['path'])==row['sha256']
 return True

def publish(frame,release):
 f=V/'frames'/f'{frame:06d}';data=f/'data';marker=f/'native_semantic_ready.json'
 if marker.exists():
  m=load(marker);assert sha(data/'prepared_ready.json')==m['prepared_ready_sha256'];return
 q=load(data/'final_v5_source_QA.json');guard=f/'crew_guard/plan_receipt.json';assert q['status']=='pass'and q['frame']==frame and q['cameras']==66 and load(guard)['status']=='pass';assert q['prepared_ready_sha256']==sha(data/'prepared_ready.json')and q['ready_sha256']==sha(data/'ready.json');visibility=data/'visibility_unknown_QA_v74.json';assert load(visibility)['status']=='pass';prepared=load(data/'prepared_ready.json');hair=data/'admission_unknown_hair/manifest.json';assert prepared['admission_unknown_hair_manifest_sha256']==sha(hair)
 inventory=native_inventory(V,frame,load(V/'source_inventory.json')['cameras']);bindings=load(f/'source_mask_binding.json')['records'];ready=load(data/'ready.json')
 for camera,i in inventory.items():
  b=next(x for x in bindings if x['camera']==camera);r=next(x for x in ready['records']if x['camera']==camera);g=load(f/'crew_guard'/(camera+'_people.json'));assert b['native_mask_sha256']==i['archive_sha256']and b['native_mask_receipt_sha256']==i['metadata_sha256'];assert g['current_native_archive_sha256']==i['archive_sha256']and g['current_candidate_archive_sha256']==i['person_candidates_sha256'];assert sha(r['semantic'])==r['semantic_sha256']and sha(data/r['image'])==r['rgb_sha256']
 write(marker,dict(kind='SOURCE_BOUND_NATIVE_ACTOR_SEMANTICS_READY_V1',frame=frame,cameras=66,source_ready_sha256=sha(V/'sources'/str(frame)/'source_ready.json'),prepared_ready_sha256=sha(data/'prepared_ready.json'),semantic_ready_sha256=sha(data/'ready.json'),crew_guard_receipt_sha256=sha(guard),native_masks=inventory,common_box_sha256=sha(V/'common_box.json'),no_cross_frame_NeRF_state=True,visibility_grid_dispatch_deferred=True,operations_release_sha256=sha(V/'operations/source_producer_v110_release.json'),source_QA_sha256=sha(data/'final_v5_source_QA.json'),visibility_UNKNOWN_QA_sha256=sha(visibility),admission_unknown_hair_manifest_sha256=sha(hair),required_router='NEW_C_F10_140000_CAP_ONLY',source_revision='ORDERED_CURRENT_NATIVE_SEMANTIC_COMPONENT_UNION_AND_CONTOUR_UNCERTAINTY_V107',NO_FIELD_BEFORE_NEW_GRID_AND_DENSE=True,human_source_review_claim=False,routine_source_policy_authorized_by_ROOT=True,created=time.time()))
 status(frame,'source_published',native_semantic_ready_sha256=sha(marker))

def prepare(frame,release):
 f=V/'frames'/f'{frame:06d}'
 if(f/'native_semantic_ready.json').exists():publish(frame,release);return
 if frame>1680:
  while not continuous_gate():status(frame,'wait_ROOT_1680_continuous_canary_release');time.sleep(10)
 if not(f/'union_v5_installed.json').exists():
  entry=V/'operations/sam3_assist/v5'/f'frame{frame:06d}'/'native_inputs_ready.json'
  if not entry.exists():command(frame,'native_body_current',[sys.executable,str(V/'operations/sam3_assist/code/marshawn_mask_assist_entry_v7.py'),'--campaign',str(V),'--frame',str(frame)])
  status(frame,'install_current_body_union');install(V,frame,entry,V/'operations/union_v5_future_input_install_v74.json')
 review=V/'operations'/f'{frame}_source_factory_v74_REVIEW_READY.json'
 # Adopt the already-running finite1680 factory; never start a second writer.
 if frame==1680:
  live=V/'operations/source_factory_v74_state.json'
  while not review.exists()and live.exists():
   s=load(live);pid=s.get('pid',0);cmd=Path(f'/proc/{pid}/cmdline')
   if s.get('frame')!=frame or not cmd.exists()or not any(name in cmd.read_bytes()for name in [b'marshawn_source_factory_v75.py',b'marshawn_source_factory_v110.py']):break
   status(frame,'adopt_actual_1680_source_factory',worker_pid=pid);time.sleep(10)
 if not review.exists():command(frame,'current_HEAD_source_DATA_QA',[sys.executable,str(V/'operations/marshawn_source_factory_v110.py'),'--frame',str(frame),'--previous',str(frame-5),'--prepare-only'])
 if not review.exists():raise RuntimeError('Source factory exited without actual QA receipt')
 publish(frame,release)

def controller():
 lock=open(V/'operations/source_producer_v110.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(V/'operations/source_producer_v110_identity.json',identity());release=verify()
 for frame in range(1680,2016,5):
  while True:
   try:verify();prepare(frame,release);break
   except Exception as e:
    write(V/'operations/source_producer_v110_error.json',dict(frame=frame,error=str(e),traceback=traceback.format_exc(),created=time.time(),source_publication_withheld=True));status(frame,'retry_source_quality_or_operational_failure',error=str(e));time.sleep(30)
 write(V/'operations/source_producer_v110_complete.json',dict(frames=list(range(1680,2016,5)),finished=time.time(),all_fields_independent_cold_owned_by_router=True))

def watchdog():
 lock=open(V/'operations/source_producer_v110_watchdog.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(V/'operations/source_producer_v110_watchdog_identity.json',identity())
 while not(V/'operations/source_producer_v110_complete.json').exists():
  verify();p=V/'operations/source_producer_v110_identity.json';old=load(p)if p.exists()else None
  if not alive(old):
   with(V/'operations/source_producer_v110_controller.log').open('ab')as out:subprocess.Popen([sys.executable,__file__,'controller'],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  time.sleep(10)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['controller','watchdog']);a=p.parse_args();controller()if a.action=='controller'else watchdog()
