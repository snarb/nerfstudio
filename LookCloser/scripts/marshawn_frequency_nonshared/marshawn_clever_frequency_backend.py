"""Assigned-image-only Clever2D staging, SHA archive, then source-bound publication."""
from pathlib import Path
import fcntl,hashlib,importlib.util,json,os,shlex,subprocess,time
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(value,indent=2)+'\n'
 if p.exists():
  if p.read_text()!=data:raise ValueError('Immutable Clever transaction changed '+str(p))
  return
 tmp=p.with_suffix('.pending');tmp.write_text(data);tmp.replace(p)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def key(frame,camera):
 if type(frame)is not int or len(camera)!=2 or not camera.isalpha()or not camera.isupper():raise ValueError('Original frame and physical camera required')
 return '%06d_%s'%(frame,camera)

def host_call(host,args,timeout=45):
 command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host['address'],shlex.join([str(x)for x in args])]
 return subprocess.run(command,check=True,capture_output=True,text=True,timeout=timeout)

def rsync(args):return subprocess.run(['rsync','-aL','--partial','--timeout=30','--bwlimit=4096',*map(str,args)],check=True,capture_output=True,text=True,timeout=120)

def rpc(c,host_key,action,request):
 b=c['clever_backend'];host=c['hosts'][host_key];root=Path(b['private_root']if host_key=='clever'else b['archive_root']);request=dict(request,helper_sha256=b['helper_sha256']);token=digest(dict(action=action,request=request));local=Path(b['transactions'])/'rpc'/(token+'.json');write(local,request);remote=root/'requests'/(token+'.json');host_call(host,['mkdir','-p',str(remote.parent)]);rsync([local,host['address']+':'+str(remote)])
 helper=b['Clever_helper']if host_key=='clever'else b['SG_helper'];result=host_call(host,[host['python'],helper,action,'--request',str(remote)],timeout=90);return json.loads(result.stdout)

def storage_ok(c):
 b=c['clever_backend'];path=Path(b['transactions'])/'storage.json'
 if path.exists()and time.time()-read(path)['checked_at']<20:return read(path)
 try:
  result=rpc(c,'clever','storage',dict(root=b['private_root']));result.update(eligible=result['private_bytes']<b['max_private_bytes']and result['free_bytes']>b['min_free_bytes'],checked_at=time.time())
 except(subprocess.SubprocessError,OSError,ValueError)as error:result=dict(eligible=False,error=str(error),checked_at=time.time())
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path);return result

def source(c,frame,camera):
 V=Path(c['root']);marker=V/'sources'/str(frame)/'source_ready.json';ready=read(marker);row=next(x for x in ready['records']if x['camera']==camera)
 if sha(row['image'])!=row['image_sha256']or sha(row['receipt'])!=row['receipt_sha256']:raise ValueError('Actual own-frame RGB/source changed')
 with Image.open(row['image'])as image:
  if image.size!=(1920,1012)or image.mode!='RGB':raise ValueError('Expected frozen HD RGB, not native RAW')
 if Path(row['image']).stat().st_size>8*2**20:raise ValueError('Assigned-image staging exceeds HD bound')
 return row,dict(frame=frame,camera=camera,image_sha256=row['image_sha256'],source_ready_sha256=sha(marker),source_receipt_sha256=row['receipt_sha256'],worker_sha256=c['clever_backend']['worker_sha256'],preprocess_source_sha256=c['clever_backend']['preprocess_source_sha256'])

def stage(c,frame,camera):
 b=c['clever_backend'];k=key(frame,camera);local=Path(b['transactions'])/k;row,binding=source(c,frame,camera);remote=Path(b['private_root'])/'operations'/b.get('namespace','frequency13')/'maps'/k;spec=dict(frame=frame,camera=camera,image=str(remote/'source'/('train_'+camera+'.png')),image_sha256=row['image_sha256'],output=str(remote/'output'/('train_'+camera+'.pt')),source_code=str(Path(b['private_root'])/'code'),preprocess_source_sha256=b['preprocess_source_sha256'],allocator_GiB=4);specfile=local/'spec.json';write(specfile,spec)
 request=dict(binding=binding,environment=dict(bundle_manifest_sha256=b['bundle_manifest_sha256']),files_sha256={'train_'+camera+'.png':row['image_sha256'],'spec.json':sha(specfile)})
 stagefile=local/'stage.json';write(stagefile,request);incoming=remote/('.incoming_'+digest(binding)[:24]);host=c['hosts']['clever'];host_call(host,['mkdir','-p',str(incoming)])
 # LocalHDsource may be a symlink; rsync-L transfers only its verified RGBbytes.
 rsync([row['image'],host['address']+':'+str(incoming/('train_'+camera+'.png'))]);rsync([specfile,stagefile,host['address']+':'+str(incoming)+'/'])
 ready=rpc(c,'clever','install',dict(root=b['private_root'],directory=str(remote),incoming=str(incoming),namespace=b.get('namespace','frequency13')))
 if ready['binding']!=binding or not ready['source_ready']:raise ValueError('Remote source/stage proof differs')
 write(local/'stage_ready.json',ready);return ready['spec']

def map_job(c,item):
 h=c['hosts']['clever'];base=Path(c['clever_backend']['private_root']);env=dict(PYTHONPATH=os.pathsep.join(str(base/n)for n in ['overlay','code','code/LookCloser/scripts']),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_CACHE_PATH=str(base/'cache/cuda'),TORCH_EXTENSIONS_DIR=str(base/'cache/torch_extensions'),TRITON_CACHE_DIR=str(base/'cache/triton'),MPLCONFIGDIR=str(base/'cache/matplotlib'),XDG_CACHE_HOME=str(base/'cache'),TMPDIR=str(base/'tmp'))
 return dict(name=item['name'],host='clever',command=[h['python'],str(base/'code/LookCloser/scripts/marshawn_video_frequency.py'),'--spec',item['spec']],env=env)

def publish_local(c,frame,camera,package,archive):
 b=c['clever_backend'];V=Path(c['root']);root=Path(b.get('canonical_local_root',str(V)))
 if root!=V and not(b.get('qualification_only')and root.is_relative_to(V/'operations/frequency13/qualification')):raise ValueError('Nonproduction root only permitted in isolated qualification')
 row,binding=source(c,frame,camera);target=root/'frames'/('%06d'%frame)/'data/lookcloser_frequencies'/('train_'+camera+'.pt');package=Path(package);output=package/'output'/('train_'+camera+'.pt')
 # Reuse the exact CPU validator; paths differ only for transport verification.
 sys_source=V/'code/LookCloser/scripts/marshawn_video_frequency.py';specmod=importlib.util.spec_from_file_location('_clever_local_validator',sys_source);m=importlib.util.module_from_spec(specmod);specmod.loader.exec_module(m);spec=dict(frame=frame,camera=camera,image=row['image'],image_sha256=row['image_sha256'],output=str(output),source_code=str(V/'code'),preprocess_source_sha256=b['preprocess_source_sha256'],allocator_GiB=4);m.validate_spec(spec)
 if not m.validate_existing(spec):raise ValueError('Gathered current-image map invalid')
 from marshawn_frequency_remote_store import copy_new
 target.parent.mkdir(parents=True,exist_ok=True)
 with(target.with_suffix('.publish.lock')).open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for suffix in ['.pt','.json']:copy_new(output.with_suffix(suffix),target.with_suffix(suffix),sha(output.with_suffix(suffix)))
  spec['output']=str(target)
  if not m.validate_existing(spec):raise ValueError('Localcanonical map publication invalid')
 return dict(map_sha256=sha(target),metadata_sha256=sha(target.with_suffix('.json')),archive_receipt_sha256=archive['archive_receipt_sha256'])

def finish(c,item,job):
 b=c['clever_backend'];frame,camera=item['frame'],item['camera'];k=key(frame,camera);row,binding=source(c,frame,camera);remote=Path(b['private_root'])/'operations'/b.get('namespace','frequency13')/'maps'/k;local=Path(b['transactions'])/k;local.mkdir(parents=True,exist_ok=True)
 with(local/'publication.lock').open('a+')as held:
  fcntl.flock(held,fcntl.LOCK_EX)
  proof=rpc(c,'clever','verify-result',dict(root=b['private_root'],directory=str(remote),dispatch_root=c['hosts']['clever']['root'],job_name=item['name'],job_spec_sha256=job['spec_sha256'],GPU_uuid=b['GPU_uuid'],namespace=b.get('namespace','frequency13')))
  if proof['binding']!=binding:raise ValueError('Finished remote map belongs to another source')
  write(local/'remote_result.json',proof);listing=local/'files.list';listing.write_text('\n'.join(proof['files_sha256'])+'\n');package=local/'package';package.mkdir(exist_ok=True);rsync(['--files-from='+str(listing),c['hosts']['clever']['address']+':'+str(remote)+'/',str(package)+'/'])
  for name,value in proof['files_sha256'].items():
   if sha(package/name)!=value:raise ValueError('Gathered Clever artifact hash mismatch '+name)
  sgroot=Path(b['archive_root']);incoming=sgroot/'incoming'/(k+'_'+proof['manifest_sha256'][:16]);host_call(c['hosts']['silver'],['mkdir','-p',str(incoming)]);rsync([str(package)+'/',c['hosts']['silver']['address']+':'+str(incoming)+'/'])
  archive=rpc(c,'silver','archive',dict(root=str(sgroot),incoming=str(incoming),key=k,result_manifest_sha256=proof['manifest_sha256'],source_code=str(Path(c['hosts']['silver']['campaign'])/'code')));write(local/'archive_verified.json',archive)
  # Archive completion is mandatory before eithercanonical publication.
  sg_spec=Path(c['hosts']['silver']['campaign'])/'operations/future_rgb_sg'/('%06d'%frame)/'specs'/(camera+'.json')
  if b.get('qualification_only'):sg_spec=Path(b['qualification_SG_spec'])
  canonical_SG=rpc(c,'silver','publish',dict(archive_dir=archive['SG_archive'],spec_path=str(sg_spec),archive_receipt_sha256=archive['archive_receipt_sha256'],qualification=b.get('qualification_only',False)))
  canonical_LOCAL=publish_local(c,frame,camera,package,archive)
  result=dict(frame=frame,camera=camera,binding=binding,canonical_SG=canonical_SG,canonical_LOCAL=canonical_LOCAL,SG_archive=archive['SG_archive'],archive_receipt_sha256=archive['archive_receipt_sha256'],map_sha256=proof['files_sha256']['output/train_'+camera+'.pt'],metadata_sha256=proof['files_sha256']['output/train_'+camera+'.json'],checkpoint_eviction=dict(performed=False,eligible_files=[],reason='Frozen2Dfitter emits no large checkpoints; PNGs/maps/receipts retained'),no_original_or_3D_path_removed=True)
  write(local/'published.json',result)
  # Keep remote maps/receipts and the durable archivepointer; no input eviction.
  pointer=local/'archive_pointer.json';write(pointer,result);rsync([pointer,c['hosts']['clever']['address']+':'+str(remote/'archive_pointer.json')]);return result
