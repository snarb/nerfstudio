"""Durable transport-only render router: native GPU geometry → shared-FSX CPU.

Default frozen R0/s8/g1 rendering. No native TIFF bulk copy and no checkpoint
or YAML relocation. Scientific training snapshots remain untouched.
"""
from pathlib import Path
import argparse,json,hashlib,subprocess,shlex,sys,time,os,fcntl,shutil
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');C=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107');SV=Path('/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.pending');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def call(h,args,**kw):return subprocess.run(['ssh','-o','BatchMode=yes',h['address'],shlex.join(args)]if h.get('address')else args,check=True,capture_output=True,text=True,**kw)
def identity(pid):
 p=Path('/proc')/str(pid)
 try:s=(p/'stat').read_text().split(') ',1)[1].split();return dict(pid=pid,start=s[19],state=s[0],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=(p/'cmdline').read_bytes().decode().split('\0'))
 except FileNotFoundError:return None

def main(a):
 O=Path(__file__).parent;lock=(O/'postprocess.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);sys.path[:0]=[str(C/'code'),str(C/'code/LookCloser/scripts')]
 import marshawn_tuning_controller as fleet
 import marshawn_video_render as renderer
 if(O/'delivery_complete.json').exists():return
 plan=read(a.fleet/'video_plan.json');hosts=plan['hosts'];export=read(a.export_plan);request=read(export['path']);assert request['version']==4 and request['paused_GT_authorized']is True;request=dict(request,source_transforms='source_transforms.json');sourceTF=C/'frames/001660/data/transforms.json';assert sha(sourceTF)==request['source_transforms_sha256'];canonical=O/'canonical_path.json';write(canonical,request)
 out=C/'video/geometry_cpu_delivery_params_v2';gather=C/'video/geometry_cpu_float_v1';(out/'frames').mkdir(parents=True,exist_ok=True);(gather/'frames').mkdir(parents=True,exist_ok=True)
 for p in [out,out/'frames',gather,gather/'frames']:os.chmod(p,0o777)
 release=read(O/'root_release.json')
 if release.get('approved')is not True or release['profile_sha256']!=sha(O/'hybrid_params.json')or release['CPU_worker_sha256']!=sha(O/'worker_params.py')or release['route_sha256']!=sha(__file__)or release['native_R0_GPU_worker_sha256']!=sha(O/'worker.py'):raise ValueError('ROOT parameter-only release missing/changed')
 helperfiles=[O/n for n in ['worker.py','worker_params.py','one.py','cap_loader.py','render_gate_loader.py','focal_scale.py','hybrid_params.json','root_release.json']];sourcebindings={str(p):sha(p)for p in [Path(__file__),*helperfiles,a.export_plan,a.fleet/'video_plan.json',Path(export['path']),C/'code_seal.json']};specs={};controls={};generic={};base=read(export['spec'])
 for name,h in hosts.items():
  root=Path(h['root']);local=O/('control_'+name);local.mkdir(exist_ok=True);control=root/'operations/video_geometry_cpu_params_v2'/('control_'+name);controls[name]=control
  spec=dict(base,output=str(out),geometry_output=str(root/'video/geometry_cpu_float_v1'),gathered_geometry=str(gather),frame_root_template=str(root/'frames/{frame_id}'),data_template=str(C/'frames/{frame_id}/data'),render_profile='frozen_R0_s8_g1',ray_chunk=2048)
  if name=='silver':spec.update(legacy_hybrid_root=str(SV/'video/reference/legacy'),photometry_fit=str(SV/'video/reference/photometry_fit.json'))
  # CPU receives its own LOCAL spec; no foreign absolute data/model paths are rewritten.
  write(local/'spec.json',spec);write(local/'path.json',request);shutil.copy2(sourceTF,local/'source_transforms.json')
  for p in helperfiles:shutil.copy2(p,local/p.name)
  if name!='silver':
   Path(root/'video/geometry_cpu_float_v1/frames').mkdir(parents=True,exist_ok=True);os.chmod(root/'video/geometry_cpu_float_v1',0o777);os.chmod(root/'video/geometry_cpu_float_v1/frames',0o777)
  call(h,['mkdir','-p',str(control)])
  if h.get('address')and root!=C:subprocess.run(['rsync','-a',str(local)+'/',h['address']+':'+str(control)+'/'],check=True)
  elif local!=control:raise ValueError('LOCAL controls differ')
  generic[name]=dict(h,root=str(root/'operations/video_geometry_cpu_params_v2'/('dispatch_'+name)));specs[name]=local/'spec.json'
  if name!='silver':
   p=Path(generic[name]['root']);p.mkdir(parents=True,exist_ok=True);os.chmod(p,0o777)
 gpu=read(O/'GPU_state.json')if(O/'GPU_state.json').exists()else dict(jobs={});jobs=[];assignments={}
 for i,record in enumerate(request['records'][:72]):
  assignment_path=Path(plan['assignments'])/('%06d.json'%int(record['frame_id']));assignment=read(assignment_path);host=assignment['host'];h=hosts[host]
  if Path(assignment['host_root'])!=Path(h['root']):raise ValueError('Model-host assignment mismatch')
  assignments[str(i)]=dict(assignment=assignment,path=str(assignment_path),sha256=sha(assignment_path));job=dict(name=f'geometry_cpu_v1_{i:04d}',host=host,command=[h['python'],str(controls[host]/'worker.py'),'gpu','--code',str(Path(h['root'])/'code'),'--spec',str(controls[host]/'spec.json'),'--path',str(controls[host]/'path.json'),'--index',str(i)],cwd=str(controls[host]),env={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'});jobs.append(job);gpu['jobs'].setdefault(job['name'],dict(status='pending'))
 write(O/'render_plan.json',dict(schema_version=1,poll_seconds=5,hosts=generic,jobs=jobs));write(O/'bindings.json',dict(source_files_sha256=sourcebindings,assignments=assignments,path_sha256=sha(canonical),raw_bulk_copy_required=False,default_profile='qualified_sigma4_gain0p75_smoothfocal_nativeR0',model_CP_YAML_native_host_preserved=True))
 cpu=read(O/'CPU_state.json')if(O/'CPU_state.json').exists()else{};gathered=set(read(O/'gathered.json')['indices'])if(O/'gathered.json').exists()else set();cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 while True:
  for name,digest in sourcebindings.items():
   if sha(name)!=digest:raise ValueError('Live renderer source/input changed '+name)
  fleet.tick(dict(schema_version=1,poll_seconds=5,hosts=generic,jobs=jobs),gpu,O);write(O/'GPU_state.json',gpu)
  for i,job in enumerate(jobs):
   if i in gathered or gpu['jobs'][job['name']]['status']!='complete':continue
   assignment=assignments[str(i)]['assignment'];h=hosts[assignment['host']];src=Path(h['root'])/'video/geometry_cpu_float_v1/frames'/f'{i:04d}';dest=gather/'frames'/f'{i:04d}'
   if assignment['host']=='silver':dest.mkdir(exist_ok=True);subprocess.run(['rsync','-a','--exclude=GPU.lock',h['address']+':'+str(src)+'/',str(dest)+'/'],check=True)
   receipt=read(dest/'geometry_complete.json')
   if receipt.get('kind')!='NATIVE_MODEL_BOUND_FLOAT_GEOMETRY_V1':raise ValueError('Unqualified native geometry package')
   b=receipt['binding'];cp=receipt['checkpoint_binding']
   if b['path_sha256']!=sha(canonical)or b['record']!=request['records'][i]or Path(cp['production_campaign'])!=Path(h['root'])or cp['completed_iterations']>140000:raise ValueError('Mixed geometry/model revision')
   for name,digest in receipt['files_sha256'].items():
    if sha(dest/name)!=digest:raise ValueError('Gathered geometry changed')
   gathered.add(i);write(O/'gathered.json',dict(indices=sorted(gathered),gathered_only_float_geometry_and_receipts=True))
  for i in gathered:
   key=str(i);p=out/'frames'/f'{i:04d}'/'complete.json'
   if p.exists():
    receipt=read(p)
    if receipt['binding']['path_sha256']!=sha(canonical)or receipt['binding']['spec_sha256']!=sha(specs['olive']):raise ValueError('Mixed CPU path/profile')
    for name,digest in receipt['images'].items():
     if sha(p.parent/name)!=digest:raise ValueError('CPU PNG changed')
    cpu[key]=dict(status='complete');continue
   previous=cpu.get(key,{});actual=identity(previous.get('pid',-1))if previous.get('pid')else None
   if actual and actual['state']!='Z'and all(actual[k]==previous.get(k)for k in ['start','boot']):continue
   if previous.get('status')=='running':cpu[key]=dict(status='pending',attempts=previous.get('attempts',1),last_worker_dead=True)
  running=sum(v.get('status')=='running'for v in cpu.values())
  for i in sorted(gathered):
   key=str(i)
   if cpu.get(key,{}).get('status')in ['running','complete']or running>=a.cpu_workers:continue
   previous=cpu.get(key,{});attempts=previous.get('attempts',0)+1
   if attempts>3:raise RuntimeError('Repeated CPU render failure '+key)
   p=out/'frames'/f'{i:04d}';p.mkdir(exist_ok=True);command=[hosts['olive']['python'],str(controls['olive']/'worker_params.py'),'cpu','--code',str(C/'code'),'--spec',str(specs['olive']),'--path',str(controls['olive']/'path.json'),'--index',str(i)]
   with(p/'CPU_stdout.log').open('ab')as log:child=subprocess.Popen(command,cwd=controls['olive'],env=cpuenv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   ident=identity(child.pid);cpu[key]=dict(status='running',attempts=attempts,command=command,**ident);running+=1
  write(O/'CPU_state.json',cpu);write(O/'status.json',dict(time=time.time(),geometry_gathered=len(gathered),CPU_complete=sum(v['status']=='complete'for v in cpu.values()),CPU_running=running))
  failures=[n for n,r in gpu['jobs'].items()if r['status']in ['failed','blocked','orphaned']]
  if failures:raise RuntimeError('Native geometry worker requires review '+','.join(failures))
  if len(cpu)==72 and all(v['status']=='complete'for v in cpu.values()):break
  time.sleep(5)
 # Frozen held-source2020 crop/shift rendering, then frozen encoding.
 local_spec=read(specs['olive']);local_spec['_spec_path']=str(specs['olive']);one=controls['olive']/'one.py'
 for i in range(72,120):
  with(O/'GT_CPU.log').open('ab')as log:subprocess.run([hosts['olive']['python'],str(one),'--cpu','--code',str(C/'code'),'--spec',str(specs['olive']),'--path',str(controls['olive']/'path.json'),'--index',str(i)],cwd=controls['olive'],env=cpuenv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
 renderer.encode(local_spec,canonical);artifacts={}
 for kind,label in [('lookcloser','LookCloser'),('hybrid','Hybrid')]:
  src=out/f'Marshawn1660_3s_24fps_{kind}_native_GT2s.mp4';dest=out/f'Marshawn1660_3sNeural_2sPaused2020_24fps_{label}_4K.mp4'
  if not dest.exists():os.link(src,dest)
  if sha(dest)!=sha(src):raise ValueError('Delivery alias changed')
  artifacts[str(dest)]=sha(dest)
 write(O/'delivery_complete.json',dict(artifacts_sha256=artifacts,path_sha256=sha(canonical),output_frames=120,fps=24,source_time_frozen_after_t3=True,GT_pan_noresize=True,raw_bulk_copy_required=False,CP_YAML_not_relocated=True,render_profile='qualified_sigma4_gain0p75_smoothfocal_nativeR0',manual_final_visual_review_required=True,finished=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fleet',type=Path,required=True);p.add_argument('--export-plan',type=Path,required=True);p.add_argument('--cpu-workers',type=int,default=3);a=p.parse_args();main(a)
