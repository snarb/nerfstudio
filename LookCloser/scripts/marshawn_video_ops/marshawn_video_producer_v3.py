"""Source-owned chronological native preparation; publish only guarded semantics.

This producer does not seal frequency inventories, carve a 3D grid, dispatch a
field job, or modify scientific code. The visibility dispatcher owns those.
"""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from prepare_marshawn_tuning import sha,write
from crew_guard_plan import candidates

def load(path):return json.loads(Path(path).read_text())
def verify_release(a):
 r=load(a.release);root=a.root;code=root/'code/LookCloser/scripts';original=load(root/'prep_release.json')
 if not r.get('authorized') or r['common_box_sha256']!=sha(root/'common_box.json'):raise ValueError('Unqualified operational release/box')
 if r['original_prep_release_sha256']!=sha(root/'prep_release.json'):raise ValueError('Original preparation release changed')
 if original['source_inventory_sha256']!=sha(root/'source_inventory.json'):raise ValueError('RAW source inventory changed')
 for path,digest in original.get('source_files_sha256',{}).items():
  if sha(path)!=digest:raise ValueError('Frozen photometry/calibration/firstinput source changed '+path)
 for name,digest in original['scripts_sha256'].items():
  if sha(code/name)!=digest:raise ValueError('Frozen scientific helper changed '+name)
 for name,digest in r['operations_sha256'].items():
  if sha(root/'operations'/name)!=digest:raise ValueError('Operational helper changed '+name)
 for path,digest in r.get('external_files_sha256',{}).items():
  if sha(path)!=digest:raise ValueError('Released external helper/owner changed '+path)
 return r

def command(a,frame,name,argv):
 log=a.root/'operations'/'producer_v3_logs'/f'{frame:06d}_{name}.log';log.parent.mkdir(exist_ok=True)
 state=dict(producer_pid=os.getpid(),frame=frame,stage=name,command=argv,started=time.time());write(a.root/'operations/producer_v3_state.json',state)
 env=dict(os.environ,PYTHONPATH=str(a.root/'operations')+':'+str(a.root/'code/LookCloser/scripts'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
 with log.open('a')as stream:
  child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);state['worker_pid']=child.pid;write(a.root/'operations/producer_v3_state.json',state)
  while child.poll()is None:state['checked_at']=time.time();write(a.root/'operations/producer_v3_state.json',state);time.sleep(10)
  if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}; {log}')
 write(log.with_suffix('.done.json'),dict(state,finished=time.time()))

def wait_source(a,frame):
 path=a.root/'sources'/str(frame)/'source_ready.json';owner=load(a.root/'operations/future_sources/ownership.json')
 if frame not in owner['frame_ids']:raise ValueError('Frame not owned by exclusive source decoder')
 while not path.is_file():write(a.root/'operations/producer_v3_state.json',dict(producer_pid=os.getpid(),frame=frame,stage='wait_exclusive_source_owner',checked_at=time.time()));time.sleep(20)
 marker=load(path)
 if marker['config_sha256']!=owner['config_sha256']:raise ValueError('Source-ready configuration differs from exclusive owner')
 if {r['camera']for r in marker['records']}!=set(owner['cameras']):raise ValueError('Source camera membership differs')
 assert marker['kind']=='ALL66_FROZEN_RGB_NATIVE_SOURCE_READY_V1'and marker['frame']==frame and marker['source_verified_all66']and len(marker['records'])==66
 for row in marker['records']:
  if sha(row['receipt'])!=row['receipt_sha256']:raise ValueError('Source receipt changed')
  receipt=load(row['receipt'])
  if receipt['frame']!=frame or receipt['camera']!=row['camera']:raise ValueError('Foreign frame/camera source receipt')
  for key in ['image','raw_native_image']:
   if sha(row[key])!=row[key+'_sha256']or str(receipt[key])!=str(row[key]):raise ValueError('Source image changed '+row['camera'])
 return sha(path)

def native_inventory(root,frame,cameras):
 f=root/'frames'/f'{frame:06d}';result={}
 for camera in cameras:
  p=f/'native_masks'/f'{camera}.npz';meta=p.with_suffix('.json');m=load(meta)
  if sha(p)!=m['archive_sha256']:raise ValueError('Native actor archive changed '+camera)
  cp,cm=candidates(root,frame,camera)
  result[camera]=dict(archive_sha256=sha(p),metadata_sha256=sha(meta),person_candidates_sha256=sha(cp))
 return result

def prepare(a,release,frame,cameras):
 root=a.root;f=root/'frames'/f'{frame:06d}';marker=f/'native_semantic_ready.json'
 if marker.exists():
  r=load(marker)
  if sha(f/'data/prepared_ready.json')!=r['prepared_ready_sha256']:raise ValueError('Published prepared source changed')
  return
 if frame==1660 or(f/'dispatch_ready.json').exists():return
 source_sha=None
 if frame>=1675:
  source_sha=wait_source(a,frame)
  assist=release['native_assist_command'];values=dict(frame=frame,root=str(root),python=sys.executable)
  command(a,frame,'native_assist',[str(s).format(**values)for s in assist])
 native_inventory(root,frame,cameras)
 command(a,frame,'crew_current_person_guard',[sys.executable,str(root/'operations/crew_guard_autonomous.py'),'--root',str(root),'--frame',str(frame),'--policy',str(root/'operations/actor_fragment_policy.json')])
 guard=f/'crew_guard/plan_receipt.json';g=load(guard)
 if g['status']!='pass':
  write(f/'crew_guard/publication_blocked.json',dict(frame=frame,reason='Current-frame person identity or component requires fresh native qualification',guard_sha256=sha(guard),created=time.time()))
  raise ValueError('UNKNOWN current-frame person overlap: publication blocked')
 # Existing ready files may come from RGB-only map prefit. The frozen builder
 # validates existing image content, retains frequency files, and recomputes W64.
 command(a,frame,'data',[sys.executable,str(root/'code/LookCloser/scripts/marshawn_video_prep_data.py'),'--root',str(root),'--frame',str(frame),'--box',str(root/'common_box.json')])
 command(a,frame,'native_visual_receipts',[sys.executable,str(root/'operations/marshawn_native_review.py'),'--root',str(root),'--frame',str(frame)])
 inv=native_inventory(root,frame,cameras)
 bindings=load(f/'source_mask_binding.json')['records'];ready=load(f/'data/ready.json')
 for camera in cameras:
  guardrow=load(f/'crew_guard'/(camera+'_people.json'));binding=next(r for r in bindings if r['camera']==camera);row=next(r for r in ready['records']if r['camera']==camera)
  if guardrow['current_native_archive_sha256']!=inv[camera]['archive_sha256']or guardrow['current_candidate_archive_sha256']!=inv[camera]['person_candidates_sha256']:raise ValueError('Stale person guard '+camera)
  if binding['native_mask_sha256']!=inv[camera]['archive_sha256']or binding['native_mask_receipt_sha256']!=inv[camera]['metadata_sha256']:raise ValueError('Stale data mask binding '+camera)
  if sha(f/'data'/row['image'])!=row['rgb_sha256']or sha(row['semantic'])!=row['semantic_sha256']:raise ValueError('Prepared source content changed '+camera)
 result=dict(kind='SOURCE_BOUND_NATIVE_ACTOR_SEMANTICS_READY_V1',frame=frame,cameras=66,source_ready_sha256=source_sha,prepared_ready_sha256=sha(f/'data/prepared_ready.json'),semantic_ready_sha256=sha(f/'data/ready.json'),crew_guard_receipt_sha256=sha(guard),native_masks=inv,common_box_sha256=sha(root/'common_box.json'),no_cross_frame_NeRF_state=True,visibility_grid_dispatch_deferred=True,operations_release_sha256=sha(a.release),created=time.time())
 write(marker,result)

def main(a):
 lock=open(a.root/'operations/producer_v3.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);release=verify_release(a);cameras=load(a.root/'source_inventory.json')['cameras'];assert len(cameras)==66
 frames=[a.only_frame]if a.only_frame else list(range(1670,2016,5))
 for frame in frames:
  while True:
   try:verify_release(a);prepare(a,release,frame,cameras);break
   except Exception as e:
    write(a.root/'operations/producer_v3_error.json',dict(frame=frame,error=str(e),traceback=traceback.format_exc(),created=time.time(),policy='UNKNOWN blocks native_semantic_ready; no training dispatch'))
    if a.fail_fast:raise
    time.sleep(40)
 write(a.root/'operations/producer_v3_complete.json',dict(frames=frames,finished=time.time(),no_field_state_transferred=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--only-frame',type=int);p.add_argument('--fail-fast',action='store_true');main(p.parse_args())
