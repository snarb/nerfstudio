"""CPU-only immutable staging/archive/publication for the Clever2D backend."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,shutil,sys,uuid
os.environ['CUDA_VISIBLE_DEVICES']=''

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+str(uuid.uuid4())+'.pending');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def inside(path,root):
 path=Path(path);root=Path(root)
 if not path.is_absolute()or not root.is_absolute()or '..'in path.parts:raise ValueError('Absolute owned paths without traversal required')
 path.relative_to(root)
 for p in [path,*path.parents]:
  if p.is_symlink():raise ValueError('Owned path may not contain symlinks')
  if p==root:break
 return path

def check_files(root,files):
 for name,value in files.items():
  path=inside(Path(root)/name,root)
  if not path.is_file()or sha(path)!=value:raise ValueError('Source/archive hash mismatch '+name)

def copy_new(source,target,value):
 """Atomic create; never replace a pre-existing different artifact."""
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  if sha(target)!=value:raise ValueError('Refuse foreign existing artifact '+str(target))
  return
 tmp=target.with_name(target.name+'.'+str(uuid.uuid4())+'.pending');shutil.copyfile(source,tmp)
 if sha(tmp)!=value:raise ValueError('Copy hash mismatch')
 try:os.link(tmp,target)
 except FileExistsError:
  if sha(target)!=value:raise ValueError('Artifact appeared with another hash')
 finally:tmp.unlink()

def frequency(source):
 path=Path(source)/'LookCloser/scripts/marshawn_video_frequency.py';s=importlib.util.spec_from_file_location('_source_bound_frequency',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def storage(root):
 root=Path(root);size=sum(p.stat().st_size for p in root.rglob('*')if p.is_file()and not p.is_symlink());return dict(private_bytes=size,free_bytes=shutil.disk_usage(root).free)

def environment(root,expected):
 root=Path(root)
 if sha(root/'bundle_manifest.json')!=expected['bundle_manifest_sha256']:raise ValueError('Dependency manifest changed')
 manifest=read(root/'bundle_manifest.json');check_files(root,manifest['files_sha256'])
 sys.path[:0]=[str(root/'overlay'),str(root/'code'),str(root/'code/LookCloser/scripts')]
 import torch
 if torch.__version__!='2.7.1+cu128'or torch.version.cuda!='12.8'or not torch._C._GLIBCXX_USE_CXX11_ABI:raise ValueError('Qualified base Torch ABI changed')
 import tinycudann_bindings._120_C
 return dict(manifest_sha256=expected['bundle_manifest_sha256'],Torch=torch.__version__,CUDA=torch.version.cuda)

def install(root,directory,incoming,namespace='frequency13'):
 if namespace not in ['frequency13','frequency13_qualification']:raise ValueError('Unexpected private namespace')
 directory=inside(directory,Path(root)/'operations'/namespace/'maps');incoming=inside(incoming,directory);request=read(incoming/'stage.json');binding=request['binding'];source=directory/'source';directory.mkdir(parents=True,exist_ok=True)
 with(directory/'install.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  environment(root,request['environment'])
  if source.exists():
   old=read(source/'stage.json')
   if old!=request:raise ValueError('Existing current-image source belongs to another binding')
   check_files(source,request['files_sha256']);return dict(source_ready=True,binding=binding,spec=str(source/'spec.json'),stage_sha256=sha(source/'stage.json'))
  check_files(incoming,request['files_sha256']);spec=read(incoming/'spec.json')
  if spec['frame']!=binding['frame']or spec['camera']!=binding['camera']or spec['image_sha256']!=binding['image_sha256']:raise ValueError('Current-image spec binding mismatch')
  if Path(spec['source_code'])!=Path(root)/'code'or Path(spec['image'])!=source/('train_'+binding['camera']+'.png')or Path(spec['output'])!=directory/'output'/('train_'+binding['camera']+'.pt'):raise ValueError('Spec escaped private assigned-image namespace')
  if spec.get('allocator_GiB')!=4:raise ValueError('Canonical4GiB allocation required')
  (directory/'output').mkdir(exist_ok=True);incoming.rename(source)
  frequency(spec['source_code']).validate_spec(spec)
  return dict(source_ready=True,binding=binding,spec=str(source/'spec.json'),stage_sha256=sha(source/'stage.json'))

def verify_result(root,directory,request):
 namespace=request.get('namespace','frequency13')
 if namespace not in ['frequency13','frequency13_qualification']:raise ValueError('Unexpected private namespace')
 directory=inside(directory,Path(root)/'operations'/namespace/'maps');stage=read(directory/'source/stage.json');spec=read(directory/'source/spec.json');binding=stage['binding'];m=frequency(spec['source_code']);m.validate_spec(spec)
 if not m.validate_existing(spec):raise ValueError('Completed job has no valid current-image map')
 meta=read(Path(spec['output']).with_suffix('.json'))
 for key,value in dict(preprocess_source_sha256=binding['preprocess_source_sha256'],fit_script_sha256=binding['worker_sha256'],seed=42,torch_version='2.7.1+cu128',host='clever-shadow',GPU_uuid=request['GPU_uuid'],progressive_fit_from_scratch=True,previous_frame_state_loaded=False).items():
  if meta.get(key)!=value:raise ValueError('Map provenance differs '+key)
 if Path(spec['output']).stat().st_size>2*2**20:raise ValueError('Unexpected large map/checkpoint; never evict unqualified files')
 gpu=inside(Path(request['dispatch_root'])/'jobs'/request['job_name']/'receipt.json',request['dispatch_root']);receipt=read(gpu)
 if receipt.get('status')!='complete'or receipt.get('returncode')!=0 or receipt.get('spec_sha256')!=request['job_spec_sha256']or receipt.get('gpu_uuid')!=request['GPU_uuid']:raise ValueError('Actual own worker completion/provenance mismatch')
 proof=directory/'proofs';proof.mkdir(exist_ok=True);copy_new(gpu,proof/'GPU_receipt.json',sha(gpu));copy_new(gpu.with_name('spec.json'),proof/'GPU_spec.json',sha(gpu.with_name('spec.json')))
 names=['source/stage.json','source/spec.json','source/train_'+binding['camera']+'.png','output/train_'+binding['camera']+'.pt','output/train_'+binding['camera']+'.json','proofs/GPU_receipt.json','proofs/GPU_spec.json'];files={n:sha(directory/n)for n in names};result=dict(kind='COMPLETE_CLEVER_CURRENT_IMAGE_2D_RESULT_V1',binding=binding,files_sha256=files,job_name=request['job_name'],job_spec_sha256=request['job_spec_sha256'],GPU_uuid=request['GPU_uuid'],no_checkpoints_emitted=True)
 p=directory/'result_manifest.json'
 if p.exists()and read(p)!=result:raise ValueError('Immutable result manifest changed')
 if not p.exists():atomic(p,result)
 return dict(verified=True,directory=str(directory),manifest_sha256=sha(p),files_sha256=dict(files,**{'result_manifest.json':sha(p)}),binding=binding)

def archive(root,incoming,key,expected_manifest,source_code):
 root=Path(root);incoming=inside(incoming,root/'incoming');target=inside(root/'jobs'/key,root/'jobs');root.mkdir(parents=True,exist_ok=True);(root/'jobs').mkdir(exist_ok=True)
 with(root/'archive.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  directory=target if target.exists()else incoming
  if sha(directory/'result_manifest.json')!=expected_manifest:raise ValueError('Archive belongs to another result')
  result=read(directory/'result_manifest.json');check_files(directory,result['files_sha256']);binding=result['binding'];original=read(directory/'source/spec.json');local=dict(original,image=str(directory/'source'/('train_'+binding['camera']+'.png')),output=str(directory/'output'/('train_'+binding['camera']+'.pt')),source_code=str(source_code));m=frequency(source_code);m.validate_spec(local)
  if not m.validate_existing(local):raise ValueError('Archive map failed frozen CPU validation')
  receipt=dict(kind='SHA_VERIFIED_IMMUTABLE_CLEVER_2D_ARCHIVE_V1',binding=binding,result_manifest_sha256=expected_manifest,files_sha256=result['files_sha256'],SG_archive=str(target),no_originals_or_3D_models_moved=True)
  if(directory/'archive_receipt.json').exists()and read(directory/'archive_receipt.json')!=receipt:raise ValueError('Archive receipt changed')
  if not(directory/'archive_receipt.json').exists():atomic(directory/'archive_receipt.json',receipt)
  if not target.exists():directory.rename(target)
  return dict(receipt,archive_receipt_sha256=sha(target/'archive_receipt.json'))

def publish(archive_dir,spec_path,expected_archive_receipt,qualification=False):
 archive_dir=Path(archive_dir)
 if sha(archive_dir/'archive_receipt.json')!=expected_archive_receipt:raise ValueError('Archive proof changed')
 receipt=read(archive_dir/'archive_receipt.json');check_files(archive_dir,receipt['files_sha256']);spec=read(spec_path);binding=receipt['binding']
 if spec['frame']!=binding['frame']or spec['camera']!=binding['camera']or spec['image_sha256']!=binding['image_sha256']:raise ValueError('Canonical spec/source differs')
 m=frequency(spec['source_code']);m.validate_spec(spec);target=Path(spec['output'])
 expected=Path(spec['source_code']).parent/'frames'/('%06d'%binding['frame'])/'data/lookcloser_frequencies'/('train_'+binding['camera']+'.pt')
 if target!=expected and not(qualification and '/operations/clever_frequency_archive_v13_qualification/'in str(target)):raise ValueError('Publication escaped SOURCE cache/isolatedqualification')
 target.parent.mkdir(parents=True,exist_ok=True)
 with(target.with_suffix('.publish.lock')).open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for suffix in ['.pt','.json']:
   src=archive_dir/'output'/('train_'+binding['camera']+suffix);copy_new(src,target.with_suffix(suffix),sha(src))
  if not m.validate_existing(spec):raise ValueError('Canonical cache publication invalid')
 return dict(map_sha256=sha(target),metadata_sha256=sha(target.with_suffix('.json')),archive_receipt_sha256=expected_archive_receipt)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['storage','install','verify-result','archive','publish']);p.add_argument('--request',type=Path,required=True);a=p.parse_args();q=read(a.request)
 if q.get('helper_sha256')!=sha(__file__):raise ValueError('Remote helper source changed')
 result=storage(q['root'])if a.action=='storage'else install(q['root'],q['directory'],q['incoming'],q.get('namespace','frequency13'))if a.action=='install'else verify_result(q['root'],q['directory'],q)if a.action=='verify-result'else archive(q['root'],q['incoming'],q['key'],q['result_manifest_sha256'],q['source_code'])if a.action=='archive'else publish(q['archive_dir'],q['spec_path'],q['archive_receipt_sha256'],q.get('qualification',False))
 print(json.dumps(result))
