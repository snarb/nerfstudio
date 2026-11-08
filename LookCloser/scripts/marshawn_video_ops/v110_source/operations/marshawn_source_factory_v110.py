"""Durable CURRENT-source factory. Fields, maps and 3D grids are separate owners."""
from pathlib import Path
import argparse,concurrent.futures,fcntl,json,os,shutil,subprocess,sys,time,traceback,uuid
import numpy as np
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_head_queue_v90 import make
from marshawn_head_hair_qualify_v72 import qualify
from marshawn_apply_head_consensus_v88 import apply
from marshawn_hair_admission_layer_v88 import build,bind
from marshawn_union_guard_v53 import validate_union
from marshawn_visibility_unknown_v110 import resolve,validate_DATA
from marshawn_video_producer_v3 import native_inventory
from marshawn_cpu_worker_init_v108 import initialize as cpu_worker_initialize

def load(p):return json.loads(Path(p).read_text())
def state(frame,stage,**extra):write(V/'operations/source_factory_v74_state.json',dict(frame=frame,stage=stage,pid=os.getpid(),checked_at=time.time(),**extra))
def call(frame,name,argv):
 state(frame,name,argv=argv);log=V/'operations/source_factory_v74_logs'/f'{frame}_{name}.log';log.parent.mkdir(exist_ok=True)
 with log.open('ab')as out:subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,PYTHONPATH=str(V/'operations')+':'+str(V/'code/LookCloser/scripts'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2'))
def controller_done(O):
 p=O/'controller/state.json'
 return p.exists()and all(x['status']=='complete'for x in load(p)['jobs'].values())
def wait_head(frame,O):
 while not controller_done(O):
  state(frame,'CURRENT_native_head_NN',root=str(O));launch=load(O/'launch.json');pid=launch['pid'];alive=Path(f'/proc/{pid}/cmdline').exists()
  if alive:alive=b'native_ROI_pool_controller_v6.py'in Path(f'/proc/{pid}/cmdline').read_bytes()
  if not alive:
   with(O/'controller.log').open('ab')as out:p=subprocess.Popen(launch['argv'],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONPATH=str(V/'operations/sam3_assist/code')+':'+str(V/'code/LookCloser/scripts'),PYTHONDONTWRITEBYTECODE='1'))
   launch.update(pid=p.pid,restarted=time.time());write(O/'launch.json',launch)
  time.sleep(10)
 r=load(V/'operations/sam3_assist/v5/full_release.json');remote=str(Path(r['remote_root'])/'operations'/O.name)+'/'
 call(frame,O.name+'_collect',['rsync','-rt','--include=*/','--include=*.json','--include=*.npz','--exclude=*',r['remote_host']+':'+remote,str(O/'remote_results')+'/'])
def observe(frame,O,policy):
 observations={}
 for cam in load(V/'source_inventory.json')['cameras']:
  sp=O/cam/'spec.json';C=O/'remote_results'/cam/'candidates';q,fg,geom=qualify(V,frame,cam,sp,C,policy);dest=O/'qualification'/cam;dest.mkdir(parents=True,exist_ok=True);p=dest/'tile_proposal.npz';np.savez_compressed(p,qualified_actor_hair=fg,geometry_hair_proposal=geom,crop_xyxy=np.array(q['crop_xyxy']));q.update(source_spec=str(sp),candidates=str(C),tile_proposal_sha256=sha(p));qp=dest/'qualification.json';write(qp,q);observations[cam]=[dict(proposal=str(p),qualification=str(qp))]
 return observations

def expanded(frame,observations,policy,round_number):
 r=load(V/'operations/sam3_assist/v5/full_release.json');SV=Path(r['remote_root']);O=V/'operations'/f'{frame}_head_overlap_round{round_number}_v74';SO=SV/'operations'/O.name;pending=[]
 for cam,tiles in observations.items():
  last=[x for x in tiles if load(x['qualification']).get('expansion_round',0)==round_number-1]
  for tile in last:
   q=load(tile['qualification']);spec=load(q['source_spec']);x0,y0,x1,y1=spec['crop_xyxy']
   for edge in q['required_native_overlap_edges']:
    dx={'left':-672,'right':672}.get(edge,0);dy={'top':-672,'bottom':672}.get(edge,0);nx=max(0,min(8192-1008,x0+dx));ny=max(0,min(4320-1008,y0+dy));roi=[nx,ny,nx+1008,ny+1008]
    if any(load(x['qualification'])['crop_xyxy']==roi for x in tiles):continue
    key=f'{cam}_{edge}_{nx}_{ny}';
    if not any(item[1]==key for item in pending):pending.append((cam,key,spec,roi))
 if not pending:return observations
 O.mkdir(exist_ok=True);jobs=[]
 for cam,key,spec,roi in pending:
  local=O/key;local.mkdir(exist_ok=True);remote=SO/key;spec=dict(spec,crop_xyxy=roi,output=str(remote/'candidates'),expansion_round=round_number,ROI_expansion_reason='Strong genuinelyNEWcurrenthair outside qualified actor64 or originalUNKNOWN touches tileedge; native overlap only');write(local/'spec.json',spec);jobs.append(dict(name=f'current{frame}_{key}_head_expansion_v74',host='silver',command=[r['remote_python'],str(SV/'operations/native_text_actor_parts_v6.py'),'--spec',str(remote/'spec.json')],env={'CREW_PILOT_LEASE_RECEIPT':'{job_root}/receipt.json'},metrics_file=str(remote/'candidates/receipt.json')))
 if not(O/'launch.json').exists():
  host=dict(address=r['remote_host'],python=r['remote_python'],controller=str(SV/'code/LookCloser/scripts/marshawn_tuning_controller.py'),root=str(SO/'dispatch'),gpus=list(range(8)));write(O/'plan.json',dict(schema_version=1,poll_seconds=5,hosts={'silver':host},jobs=jobs));call(frame,'stage_head_expansion',['rsync','-rt','--include=*/','--include=spec.json','--exclude=*',str(O)+'/',r['remote_host']+':'+str(SO)+'/']);request=V/'operations/native10_requests'/f'{frame}_head_overlap_round{round_number}_v74.json';write(request,dict(status='pending',host_gpus={'silver':list(range(8))},source_bound_plan=str(O/'plan.json'),source_bound_plan_sha256=sha(O/'plan.json'),frame=frame,nonce=uuid.uuid4().hex,created=time.time()));write(O/'launch.json',dict(pid=-1,argv=[sys.executable,str(V/'operations/native_ROI_pool_controller_v6.py'),str(O/'controller'),str(O/'plan.json'),str(request)]))
 wait_head(frame,O)
 for cam,key,spec,roi in pending:
  sp=O/key/'spec.json';C=O/'remote_results'/key/'candidates';q,fg,geom=qualify(V,frame,cam,sp,C,policy);dest=O/'qualification'/key;dest.mkdir(parents=True,exist_ok=True);p=dest/'tile_proposal.npz';np.savez_compressed(p,qualified_actor_hair=fg,geometry_hair_proposal=geom,crop_xyxy=np.array(roi));q.update(source_spec=str(sp),candidates=str(C),tile_proposal_sha256=sha(p),expansion_round=round_number);qp=dest/'qualification.json';write(qp,q);observations[cam].append(dict(proposal=str(p),qualification=str(qp)))
 return observations

def camera_guard(args):
 frame,camera=args
 import crew_guard_plan_v53 as m
 from marshawn_visibility_unknown_v110 import resolve
 policy=load(V/'operations/visibility_unknown_policy_v107.json');policy['policy_sha256']=sha(V/'operations/visibility_unknown_policy_v107.json');original=m.write
 def output(path,value):
  if Path(path).name in ['plan_receipt.json','identity_novelty_pass.json']:return
  if Path(path).name==camera+'_people.json':value=resolve(V,frame,value,policy)
  original(path,value)
 code=Path(m.__file__).read_text().replace('qualified_actor=np.zeros((270,512),bool);qualified_instances={};fragment_proofs=[]','qualified_actor=small(qualified_actor_rgb);qualified_instances={};fragment_proofs=[]').replace("print(camera,report['status'],unknown,residual_n,flush=True)","print(camera,report['status'],report['unknown_person_instances'],residual_n,flush=True)");scope=dict(m.__dict__);exec(compile(code,str(V/'operations/crew_guard_plan_source_H_v74_runtime.py'),'exec'),scope);scope['write']=output;scope['run'](V,frame,[camera],True);return load(V/'frames'/f'{frame:06d}'/'crew_guard'/(camera+'_people.json'))

def rebuild_if_stale(frame):
 f=V/'frames'/f'{frame:06d}';data=f/'data';prepared=data/'prepared_ready.json';binding=f/'source_mask_binding.json'
 if not prepared.exists():return
 bindings=load(binding)['records']if binding.exists()else[];fresh=len(bindings)==66
 for row in bindings:
  p=f/'native_masks'/(row['camera']+'.npz');fresh=fresh and row['native_mask_sha256']==sha(p)and row['native_mask_receipt_sha256']==sha(p.with_suffix('.json'))and row['source_receipt_sha256']==sha(V/'sources'/str(frame)/row['camera']/'receipt.json')
 if fresh:return
 if(f/'native_semantic_ready.json').exists():raise ValueError('Published source DATA cannot be rebuilt')
 archive=f/f'stale_derived_DATA_before_v74_{time.time_ns()}';archive.mkdir();records={}
 for p in list((data/'semantics').glob('*.npz'))+[data/x for x in ['ready.json','prepared_ready.json','semantic_ownership_v5.json','final_v5_source_QA.json','transforms.json','box.json']]+[binding]:
  if p.exists():
   q=archive/p.relative_to(f);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p);records[str(p.relative_to(f))]=sha(q)
 write(archive/'archive_receipt.json',dict(frame=frame,files_sha256=records,reason='Current source/native hashes differ from cached DATA bindings',RGB_and_frequency_preserved=True))
 for p in [prepared,data/'ready.json',binding]:
  if p.exists():p.unlink()

def finish(frame,previous,prepare_only):
 f=V/'frames'/f'{frame:06d}';f.mkdir(exist_ok=True);marker=f/'native_semantic_ready.json'
 if marker.exists():return
 if frame==1685 and not(V/'operations/repair_1685_v82/complete.json').exists():raise RuntimeError('Durable source correction transaction not complete; publication withheld')
 from marshawn_identity_refresh_v79 import refresh
 refresh(V,frame)
 if not(f/'union_v5_installed.json').exists():raise RuntimeError('Fresh body NN not yet installed by source owner')
 cameras=load(V/'source_inventory.json')['cameras'];O=make(frame,previous);wait_head(frame,O);policy_path=V/'operations/head_hair_policy_v74.json';policy=load(policy_path);assert policy['authorized']and frame in policy['frames'];observations=observe(frame,O,policy)
 for round_number in range(1,policy['maximum_overlap_expansion_rounds']+1):observations=expanded(frame,observations,policy,round_number)
 corrections=V/'operations/weak_head_geometry_v86/1690_corrected_ROI_bindings.json'
 if frame==1690:
  if not corrections.exists():raise RuntimeError('Finite corrected current 1690 HEAD ROI results not yet collected')
  correction=load(corrections);assert correction['frame']==1690
  for row in correction['records']:
   cam=row['camera'];sp=Path(row['spec']);C=Path(row['candidates']);q,fg,geom=qualify(V,frame,cam,sp,C,policy);dest=O/'corrected_weak_ROI'/cam;dest.mkdir(parents=True,exist_ok=True);p=dest/'tile_proposal.npz';np.savez_compressed(p,qualified_actor_hair=fg,geometry_hair_proposal=geom,crop_xyxy=np.array(q['crop_xyxy']));q.update(source_spec=str(sp),candidates=str(C),tile_proposal_sha256=sha(p),corrected_weak_ROI_seed=True);qp=dest/'qualification.json';write(qp,q);observations[cam].append(dict(proposal=str(p),qualification=str(qp)))
  receipt=O/'corrected_weak_ROI/application.json'
  if not receipt.exists():
   rows=[]
   for row in correction['records']:
    rows.append(apply(V,frame,row['camera'],[dict(spec=row['spec'],candidates=row['candidates'])],policy_path));validate_union(V,frame,row['camera'])
   write(receipt,dict(frame=frame,records=rows,correction_bindings_sha256=sha(corrections)))

 application=O/'all66_current_consensus_application.json'
 if not application.exists():
  state(frame,'apply_current_head_consensus');rows=[]
  for cam,tiles in observations.items():
   specs=[dict(spec=load(x['qualification'])['source_spec'],candidates=load(x['qualification'])['candidates'])for x in tiles];receipt=O/'application_by_camera'/(cam+'.json');receipt.parent.mkdir(exist_ok=True)
   if receipt.exists():
    item=load(receipt);assert item['actual_native_sha256']==sha(f/'native_masks'/(cam+'.npz'));rows.append(item['application'])
   else:
    item=apply(V,frame,cam,specs,policy_path);validate_union(V,frame,cam);write(receipt,dict(camera=cam,actual_native_sha256=sha(f/'native_masks'/(cam+'.npz')),application=item));rows.append(item)
  write(application,dict(frame=frame,cameras=66,records=rows,policy_sha256=sha(policy_path)))
 from marshawn_ordered_current_source_stage_v108 import run as ordered_source_stage
 state(frame,'ordered_current_semantic_and_component_source');ordered_source_stage(frame)
 manifest=f/'data/admission_unknown_hair/manifest.json'
 stale_geometry=not manifest.exists()
 if manifest.exists():
  gm=load(manifest);stale_geometry=gm.get('weak_none_rule')!='OLD_UNKNOWN_UNION_ENTIRE_BOUNDED_CURRENT_QUERIED_HEAD_ROI'or any(r['native_actor_archive_sha256']!=sha(f/'native_masks'/(r['camera']+'.npz'))for r in gm['records'])
 if stale_geometry:
  if manifest.exists():shutil.move(str(manifest.parent),str(f/'data'/('admission_unknown_hair_before_v90_'+str(time.time_ns()))))
  state(frame,'build_current_hair_geometry_UNKNOWN');build(V,frame,observations,policy_path)
 from crew_reference_advance import advance
 advance(V,frame);state(frame,'current_source_identity_and_visibility_guard')
 prior_visibility=f/'visibility_unknown_inputs'
 if prior_visibility.exists():shutil.move(str(prior_visibility),str(f/('visibility_unknown_inputs_before_v107_'+str(time.time_ns()))))
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,initializer=cpu_worker_initialize)as pool:reports=list(pool.map(camera_guard,[(frame,c)for c in cameras]))
 status='pass'if all(x['status']=='pass'for x in reports)else'unknown';guard=f/'crew_guard/plan_receipt.json';write(guard,dict(frame=frame,cameras=66,status=status,reports=[dict(camera=x['camera'],status=x['status'],unknown=x['unknown_person_instances'],actor_novelty=x['actor_novelty_trigger'])for x in reports],current_H_novelty_qualified=True,conservative_PRE0_existing_visibility_UNKNOWN_policy_sha256=sha(V/'operations/visibility_unknown_policy_v107.json'),no_prior_bitmap_final_pixels=True))
 if status!='pass':raise RuntimeError('Unresolved actual current PRE/identity conflict: source remains withheld')
 write(f/'crew_guard/autonomous_guard_ready.json',dict(frame=frame,status='pass',guard_sha256=sha(guard),policy_sha256=sha(V/'operations/visibility_unknown_policy_v107.json')))
 rebuild_if_stale(frame);call(frame,'frozen_DATA',[sys.executable,str(V/'code/LookCloser/scripts/marshawn_video_prep_data.py'),'--root',str(V),'--frame',str(frame),'--box',str(V/'common_box.json')]);call(frame,'ownership',[sys.executable,str(V/'operations/marshawn_semantic_ownership_v5.py'),'--root',str(V),'--frame',str(frame)]);bind(V,frame);validate_DATA(V,frame);call(frame,'native_visual',[sys.executable,str(V/'operations/marshawn_native_review.py'),'--root',str(V),'--frame',str(frame)]);call(frame,'source_QA',[sys.executable,str(V/'operations/marshawn_final_source_QA_v7.py'),'--root',str(V),'--frame',str(frame)])
 review=V/'operations'/f'{frame}_source_factory_v74_REVIEW_READY.json';write(review,dict(frame=frame,cameras=66,prepared_ready_sha256=sha(f/'data/prepared_ready.json'),ready_sha256=sha(f/'data/ready.json'),hair_manifest_sha256=sha(manifest),source_QA_sha256=sha(f/'data/final_v5_source_QA.json'),visibility_UNKNOWN_QA_sha256=sha(f/'data/visibility_unknown_QA_v74.json'),native_visual_receipt_sha256=sha(f/'native_review/receipt.json'),native_semantic_ready_withheld=True,created=time.time()))
 if prepare_only:return
 raise RuntimeError('Continuous publication release not yet activated; prepared source is review-ready')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--frame',type=int,required=True);p.add_argument('--previous',type=int,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();lock=open(V/'operations/source_factory_v74.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:finish(a.frame,a.previous,a.prepare_only)
 except Exception as e:write(V/'operations/source_factory_v74_error.json',dict(frame=a.frame,error=str(e),traceback=traceback.format_exc(),created=time.time()));raise
