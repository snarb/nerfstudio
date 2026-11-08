"""Finite first-six preview; exact released shaders, native CP/YAML, separate CPU outputs."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,os,shlex,shutil,subprocess,sys,time,fcntl
C=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107');V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');F=Path('/fsx/tmp/marshawn_fleet10_140k_20261107');O=Path(__file__).parent;P=C/'operations/video_geometry_cpu_params_v2';D=C/'operations/video_geometry_cpu_durable_v3'
sys.path[:0]=[str(C/'code/LookCloser/scripts'),str(D)]
import marshawn_tuning_controller as fleet
import cpu_exec
from bridge_v3 import Bridge

def read(p):return json.loads(Path(p).read_text())
def sha(p):return cpu_exec.sha(p)
def write(p,v):return cpu_exec.atomic(p,v)
def immutable(p,v):
 p=Path(p);blob=json.dumps(v,indent=2)+'\n';p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_text()!=blob:raise ValueError('Refuse different existing partial input '+str(p))
 else:p.write_text(blob)
def call(h,args,**kw):return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',h['address'],shlex.join(list(map(str,args)))]if h.get('address')else list(map(str,args)),check=True,capture_output=True,text=True,**kw)
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def setup():
 plan=read(F/'video_plan.json');export=read(V/'video/export_pipeline.json');request=read(export['path']);request['source_transforms']='source_transforms.json';immutable(O/'path.json',request)
 assert sha(O/'path.json')=='35953efcc54ce0464bc45091c0a999936bd1a12c65c6a55b735c313d972d60be'
 release=read(P/'root_release.json');assert release['approved']and release['weak_grid_guard']is False
 assert sha(P/'worker.py')==release['native_R0_GPU_worker_sha256']and sha(P/'worker_params.py')==release['CPU_worker_sha256']and sha(P/'hybrid_params.json')==release['profile_sha256']
 bindings={str(p):sha(p)for p in [Path(__file__),F/'video_plan.json',V/'video/export_pipeline.json',Path(export['path']),P/'root_release.json',P/'hybrid_params.json',P/'worker.py',P/'worker_params.py',D/'cpu_exec.py',D/'bridge_v3.py']};assignments={};hosts={};controls={};jobs=[];completion={}
 base=read(export['spec']);tf=C/'frames/001660/data/transforms.json';assert sha(tf)==request['source_transforms_sha256'];helpernames=['worker.py','worker_params.py','one.py','cap_loader.py','render_gate_loader.py','focal_scale.py','hybrid_params.json','root_release.json']
 for name,h in plan['hosts'].items():
  root=Path(h['root']);local=O/('stage_'+name);local.mkdir(exist_ok=True);remote=root/'operations/video_partial_first6_v1'/('stage_'+name);controls[name]=str(remote)
  spec=dict(base,output=str(C/'video/geometry_cpu_delivery_params_v2'),geometry_output=str(root/'video/geometry_cpu_float_v1'),gathered_geometry=str(C/'video/geometry_cpu_float_v1'),frame_root_template=str(root/'frames/{frame_id}'),data_template=str(C/'frames/{frame_id}/data'),render_profile='frozen_R0_s8_g1',ray_chunk=2048,source_visibility_adapter_sha256=sha(P/'one.py'))
  if name=='silver':
   sv=Path('/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107');spec.update(legacy_hybrid_root=str(sv/'video/reference/legacy'),photometry_fit=str(sv/'video/reference/photometry_fit.json'))
  immutable(local/'spec.json',spec);immutable(local/'path.json',request)
  for p in [tf,*[P/n for n in helpernames]]:
   dest=local/('source_transforms.json'if p==tf else p.name)
   if dest.exists():assert sha(dest)==sha(p)
   else:shutil.copy2(p,dest)
  call(h,['mkdir','-p',str(remote)])
  if remote!=local:
   if root==C:raise ValueError('Unexpected shared-FSX staging path')
   subprocess.run(['rsync','-a',str(local)+'/',h['address']+':'+str(remote)+'/'],check=True)
  hosts[name]=dict(h,root=str(root/'operations/video_partial_first6_v1'/('dispatch_'+name)))
  if root==C:
   for directory in [Path(hosts[name]['root']),Path(hosts[name]['root'])/'jobs']:
    directory.mkdir(parents=True,exist_ok=True);directory.chmod(0o777)
 for i,r in enumerate(request['records'][:6]):
  a=Path(plan['assignments'])/(r['frame_id']+'.json');assignment=read(a);host=assignment['host'];h=plan['hosts'][host];assert assignment['host_root']==h['root'];assignments[str(i)]=dict(path=str(a),sha256=sha(a),assignment=assignment)
  src=Path(h['root'])/'frames'/r['frame_id']/'frame_completion.json';done=json.loads(call(h,['cat',str(src)]).stdout);assert done['completed_iterations']==140000;completion[str(i)]=dict(path=str(src),document=done)
  jobs.append(dict(name='partial6_geometry_%04d'%i,host=host,command=[h['python'],str(Path(controls[host])/'worker.py'),'gpu','--code',str(Path(h['root'])/'code'),'--spec',str(Path(controls[host])/'spec.json'),'--path',str(Path(controls[host])/'path.json'),'--index',str(i)],cwd=controls[host],env={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}))
 cpu=O/'cpu';cpu.mkdir(exist_ok=True)
 for n in helpernames+['path.json','source_transforms.json']:
  src=O/'stage_olive'/n;dst=cpu/n
  if dst.exists():assert sha(dst)==sha(src)
  else:shutil.copy2(src,dst)
 spec=read(O/'stage_olive/spec.json');spec['output']=str(O/'images');immutable(cpu/'spec.json',spec)
 immutable(O/'inputs.json',dict(bindings=bindings,assignments=assignments,completions=completion,hosts=hosts,controls=controls,jobs=jobs,profile_sha256=sha(P/'hybrid_params.json'),path_sha256=sha(O/'path.json'),frames=6,fps=24,duration_seconds=.25,final_hook_unchanged=True))
 return read(O/'inputs.json')

def validate_package(i,inputs):
 r=read(O/'path.json')['records'][i];a=inputs['assignments'][str(i)]['assignment'];h=inputs['hosts'][a['host']];src=Path(a['host_root'])/'video/geometry_cpu_float_v1/frames'/f'{i:04d}';dest=C/'video/geometry_cpu_float_v1/frames'/f'{i:04d}'
 if a['host']=='silver':
  # Fetch only if producer's source-bound completion exists; absence is not failure.
  try:remote=json.loads(call(h,['cat',str(src/'geometry_complete.json')],timeout=10).stdout)
  except subprocess.SubprocessError:return False
  dest.mkdir(parents=True,exist_ok=True);subprocess.run(['rsync','-a','--exclude=GPU.lock',h['address']+':'+str(src)+'/',str(dest)+'/'],check=True)
 if not(dest/'geometry_complete.json').exists():return False
 d=read(dest/'geometry_complete.json');b=d['binding'];assert d['kind']=='NATIVE_MODEL_BOUND_FLOAT_GEOMETRY_V1'and b['worker_sha256']==inputs['bindings'][str(P/'worker.py')]
 assert b['path_sha256']==inputs['path_sha256']and b['record']==r and d['checkpoint_binding']['completed_iterations']==140000 and d['checkpoint_binding']['active_hard140k_revision']
 assert d['pipeline_tensor_SHA_before']==d['pipeline_tensor_SHA_after']and d['native_HC_parity']['max_RGB_byte_difference']<=1
 assert Path(d['checkpoint_binding']['production_campaign'])==Path(a['host_root'])
 for n,v in d['files_sha256'].items():assert sha(dest/n)==v,(i,n)
 return True

def allowed_slots(inputs,state):
 # Mirror existing operational priority, excluding ONLY this finite preview's own intent.
 broker=module(V/'operations/frequency10/code/marshawn_frequency13.py','live_frequency_priority');cfg=read(V/'operations/frequency10/config_v13.json');_,io,legacy,_=broker.imports(V);reserved,demand=broker.field_reservations(V,io)
 out={h:[g for g in x['gpus']if(h,g)not in reserved]for h,x in inputs['hosts'].items()}
 if demand:out={h:[]for h in out}
 old=legacy.priority(V)
 if old['yield_new_maps']:out['silver']=[g for g in out['silver']if g not in {1,5,6,7}]
 for r in legacy.material_requests(V,cfg):out['silver']=[g for g in out['silver']if g!=r['gpu']]
 for p in(V/'operations/native10_requests').glob('*.json'):
  if p.name=='partial_first6_released_R0.json':continue
  x=read(p)
  if not isinstance(x,dict)or x.get('status')=='complete'or not isinstance(x.get('host_gpus'),dict):continue
  for h,gs in x['host_gpus'].items():
   if h in out and isinstance(gs,list):out[h]=[g for g in out[h]if g not in gs]
 # At most one physical GPU per host. Running holders retain ownership naturally.
 result={};wanted={};snapshots={}
 for h,row in inputs['hosts'].items():
  try:snap=json.loads(fleet.host_call(row,['_probe','--root',row['root']]).stdout)
  except Exception:result[h]=[];continue
  snapshots[h]=snap;live=[j.get('gpu')for j in state['jobs'].values()if j.get('host')==h and j['status']not in fleet.TERMINAL|{'pending'}]
  if live:result[h]=[live[0]];continue
  if not any(j['host']==h and state['jobs'][j['name']]['status']=='pending'for j in inputs['jobs']):result[h]=[];continue
  candidates=[g for g in out[h]if str(g)in snap.get('gpus',{})and not snap['gpus'][str(g)]['processes']and snap['gpus'][str(g)]['used_mib']<1024]
  # Intent may request a slot occupied by an own short frequency fit, never NN/FIELD/unknown.
  if not candidates:
   for g in out[h]:
    d=snap.get('gpus',{}).get(str(g),{});owners=[j for j in read(V/'operations/frequency10/state.json')['jobs'].values()if j.get('host')==h and j.get('gpu')==g and j.get('status')in ['running','dispatching','unknown']]
    owned_pids={j['worker']['pid']for j in owners if j.get('worker')}
    if owned_pids and d.get('processes')and all(p['pid']in owned_pids for p in d['processes']):candidates=[g];break
  result[h]=candidates[:1]
  if result[h]:wanted[h]=result[h]
 request=V/'operations/native10_requests/partial_first6_released_R0.json';write(request,dict(status='pending'if wanted else'complete',host_gpus=wanted or{'dev3':[0]},frames=list(range(1660,1690,5)),purpose='Finite6framepreview exactreleasedR0/profile; no scientific preemption',source_code_sha256=sha(__file__),time=time.time()))
 write(O/'priority.json',dict(allowed=out,selected=result,field_demand=demand,time=time.time()));return result

def main():
 held=fleet.lock(O/'controller.lock')
 if held is None:return
 inputs=setup();state=read(O/'state.json')if(O/'state.json').exists()else dict(jobs={j['name']:dict(status='pending')for j in inputs['jobs']});write(O/'controller_identity.json',fleet.identity());gathered=set()
 # Reuse exact canonical0 and any later completed own packages; never restamp private variants.
 for i,j in enumerate(inputs['jobs']):
  if validate_package(i,inputs):state['jobs'][j['name']].update(status='complete',completion_source='exact_source_bound_existing_canonical_geometry');gathered.add(i)
 bridge=Bridge(module(P/'route.py','read_only_router_for_CPU_inputs'),O/'CPU_exec',D/'cpu_exec.py',shader_sha=inputs['bindings'][str(P/'worker_params.py')],max_concurrent=2)
 cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');cpu_jobs={}
 while len(gathered)<6 or sum((O/'images/frames'/f'{i:04d}'/'complete.json').exists()for i in range(6))<6:
  for p,v in inputs['bindings'].items():assert sha(p)==v,p
  selected=allowed_slots(inputs,state);hosts={h:dict(x,gpus=selected[h])for h,x in inputs['hosts'].items()};fleet.tick(dict(schema_version=1,poll_seconds=5,hosts=hosts,jobs=inputs['jobs']),state,O)
  for i,j in enumerate(inputs['jobs']):
   if i not in gathered and state['jobs'][j['name']]['status']=='complete'and validate_package(i,inputs):gathered.add(i)
   if state['jobs'][j['name']]['status']in ['failed','blocked','orphaned']:raise RuntimeError('PartialGPU job needs review '+j['name'])
  for i in sorted(gathered):
   out=O/'images/frames'/f'{i:04d}'
   if(out/'complete.json').exists():continue
   command=[inputs['hosts']['olive']['python'],str(O/'cpu/worker_params.py'),'cpu','--code',str(C/'code'),'--spec',str(O/'cpu/spec.json'),'--path',str(O/'cpu/path.json'),'--index',str(i)];directory=O/'CPU_exec'/f'{i:04d}';intent=cpu_exec.prepare(directory,command,bridge.inputs(command),index=i,max_concurrent=2);runtime=cpu_exec.runtime_for(directory,intent)
   if runtime and cpu_exec.live(runtime):continue
   if runtime and runtime['exec_attempt']>=2:raise RuntimeError('CPUshader failed twice '+str(i))
   out.mkdir(parents=True,exist_ok=True)
   with(out/'CPU_stdout.log').open('ab')as log:subprocess.Popen([command[0],str(D/'cpu_exec.py'),'--intent',str(directory/'intent.json'),'--expected-sha256',sha(directory/'intent.json')],cwd=O/'cpu',env=cpuenv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(O/'progress.json',dict(gathered=sorted(gathered),CPU_complete=[i for i in range(6)if(O/'images/frames'/f'{i:04d}'/'complete.json').exists()],time=time.time()));time.sleep(5)
 request=V/'operations/native10_requests/partial_first6_released_R0.json';write(request,dict(status='complete',host_gpus={'dev3':[0]},no_active_GPU_requirement=True,time=time.time()))
 encoder=module(C/'operations/video_geometry_cpu_encoding_v4/encoder.py','qualified_encoder_helpers');movies={};pngs={}
 for i in range(6):
  p=O/'images/frames'/f'{i:04d}'/'complete.json';d=read(p)
  for n,v in d['images'].items():assert sha(p.parent/n)==v;pngs[str(p.parent/n)]=v
 for kind in ['lookcloser','hybrid']:
  out=O/('Marshawn_first6_'+kind+'_24fps_0p25s_4K.mp4');receipt=out.with_suffix('.json')
  if not receipt.exists():
   if out.exists():raise ValueError('Refuse foreign unbound preview movie')
   temporary=O/(out.stem+'.pending.mp4');command=encoder.command('/home/pavlo/.conda/envs/lookcloser-jpeg/bin/ffmpeg',O/'images/frames/%04d'/(kind+'.png'),6,temporary);subprocess.run(command,check=True);proof=encoder.probe('/home/pavlo/.conda/envs/lookcloser-jpeg/bin/ffprobe',temporary,6);decode=encoder.decode_check('/home/pavlo/.conda/envs/lookcloser-jpeg/bin/ffmpeg',temporary,6,O/'images/frames',kind);temporary.replace(out);write(receipt,dict(sha256=sha(out),PNG_sha256=pngs,ffprobe=proof,decode=decode,profile_sha256=inputs['profile_sha256'],count=6,duration_seconds=.25))
  movies[str(out)]=sha(out)
 for p,v in pngs.items():assert sha(p)==v
 write(O/'complete.json',dict(frames=6,fps=24,duration_seconds=.25,profile_sha256=inputs['profile_sha256'],path_sha256=inputs['path_sha256'],native_R0=True,PNG_sha256=pngs,movies_sha256=movies,final_delivery_markers_untouched=True,finished=time.time()))
if __name__=='__main__':main()
