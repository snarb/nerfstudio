"""Model-native GPU geometry, then shared-FSX photograph transfer on CPU.

External render-only worker. The default reproduces the frozen R0/s8/g1
renderer. No model, checkpoint, YAML or scientific input is relocated/changed.
"""
from pathlib import Path
import argparse,fcntl,json,os,sys,time,hashlib
import numpy as np
if not hasattr(Path,'is_relative_to'):
 def is_relative_to(self,other):
  try:self.relative_to(other);return True
  except ValueError:return False
 Path.is_relative_to=is_relative_to

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.pending');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def setup(a):
 sys.path[:0]=[str(a.code),str(a.code/'LookCloser/scripts'),str(Path(__file__).parent)]
 import marshawn_video_render as R
 from one import source_eligibility
 spec=read(a.spec);spec['_spec_path']=str(a.spec);request=read(a.path);record=request['records'][a.index]
 if record['kind']!='neural':raise ValueError('This worker accepts neural records only')
 if spec.get('render_profile','frozen_R0_s8_g1')!='frozen_R0_s8_g1':raise ValueError('Candidate profile needs a separate qualified release')
 R.source_eligibility=source_eligibility
 if sha(spec['photometry_fit'])!=spec['photometry_fit_sha256']:raise ValueError('Photometry changed')
 if spec.get('native_vendor_pq'):
  import prepare_marshawn_color_variants
  prepare_marshawn_color_variants.PQ=Path(spec['native_vendor_pq'])
 if spec.get('native_rig'):
  import marshawn_actor_coordinates
  marshawn_actor_coordinates.RIG=Path(spec['native_rig'])
 return R,spec,request,record

def gpu(a):
 R,spec,request,record=setup(a)
 import torch
 from cap_loader import load_pipeline
 from render_gate_loader import tree_hash
 from marshawn_gpu_isolation import verify_cuda
 from PIL import Image
 if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'):raise ValueError('Physical UUID required')
 os.fstat(int(os.environ['LOOKCLOSER_GPU_LEASE_FD']));physical=verify_cuda(os.environ['CUDA_VISIBLE_DEVICES'],torch)
 root=Path(spec['frame_root_template'].format(frame_id=record['frame_id']));data=root/'data';out=Path(spec['geometry_output'])/'frames'/f'{a.index:04d}';out.mkdir(parents=True,exist_ok=True)
 lock=(out/'GPU.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 binding=dict(path_sha256=sha(a.path),record=record,model_frame_id=record['frame_id'],data_ready_sha256=sha(data/'ready.json'),transforms_sha256=sha(data/'transforms.json'),frame_completion_sha256=sha(root/'frame_completion.json'),worker_sha256=sha(__file__),code_seal_sha256=sha(root.parents[1]/'code_seal.json'))
 if(out/'geometry_complete.json').exists():
  receipt=read(out/'geometry_complete.json')
  if receipt['binding']!=binding:raise ValueError('Geometry request/model/source changed')
  for name,digest in receipt['files_sha256'].items():
   if sha(out/name)!=digest:raise ValueError('Bound geometry changed')
  return
 pipe,checkpoint=load_pipeline(root,checkpoint_kind=spec.get('checkpoint_kind','actor_lpips'));pipe.eval();pipe.requires_grad_(False);m=pipe.model;state=tree_hash(pipe.state_dict());old=R.legacy(spec)
 # Actual production-native parity, including the first genuine DEV3 model.
 selector=read(root/'frame_completion.json')['selectors'][spec.get('checkpoint_kind','actor_lpips')];archive=Path(selector['render_directory'])/'HC.png'
 k=next(i for i,p in enumerate(pipe.datamanager.eval_dataset.image_filenames)if Path(p).stem.split('_')[-1]=='HC');cam=pipe.datamanager.eval_dataset.cameras[k:k+1].to(m.device)
 with torch.inference_mode():native=m.get_outputs_for_camera(cam)['rgb'].clamp(0,1).cpu().numpy()
 archived=np.asarray(Image.open(archive).convert('RGB'));delta=np.abs(np.rint(native*255).astype(np.int16)-archived.astype(np.int16));parity=dict(archive_path=str(archive),archive_sha256=sha(archive),max_RGB_byte_difference=int(delta.max()),current_train_step=int(m.current_train_step),FR=float(m.field.feature_reweighting_strength))
 write(out/'native_HC_parity.json',parity)
 if parity['max_RGB_byte_difference']>1:raise ValueError('Native selected-checkpoint HC parity requires review')
 sources=R.Sources(data,record,spec,m,old,checkpoint['checkpoint_sha256'],out);names=old.candidates(sources,record['camera'],spec.get('source_count',16));R.geometry(m,record['camera'],out/'target.npz',old,checkpoint['checkpoint_sha256'],spec.get('ray_chunk',2048))
 source_rows={}
 for name in names:
  row=sources.rows[name];R.geometry(m,row,out/'sources'/(name+'.npz'),old,checkpoint['checkpoint_sha256'],spec.get('ray_chunk',2048));source_rows[name]=dict(row=row,semantic_sha256=sha(data/'semantics'/(name+'.npz')),valid_mask_sha256=sha(data/row['mask_path']))
 if tree_hash(pipe.state_dict())!=state:raise AssertionError('Geometry rendering changed the pipeline tensors')
 files={str(f.relative_to(out)):sha(f)for f in out.rglob('*')if f.is_file()and f.name!='geometry_complete.json'and f.suffix in ['.npz','.json']}
 write(out/'geometry_complete.json',dict(kind='NATIVE_MODEL_BOUND_FLOAT_GEOMETRY_V1',binding=binding,checkpoint_binding=checkpoint,source_rows=source_rows,source_names=names,files_sha256=files,native_HC_parity=parity,pipeline_tensor_SHA_before=state,pipeline_tensor_SHA_after=tree_hash(pipe.state_dict()),physical_GPU=physical,PID=os.getpid(),no_raw_TIFF_copied=True,no_model_or_YAML_moved=True,finished=time.time()))

def cpu(a):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Photograph transfer is CPU-only')
 R,spec,request,record=setup(a)
 profile=read(Path(__file__).parent/'hybrid_params.json');release=read(Path(__file__).parent/'root_release.json')
 if release.get('approved')is not True or release['profile_sha256']!=sha(Path(__file__).parent/'hybrid_params.json')or release['CPU_worker_sha256']!=sha(__file__):raise ValueError('ROOT parameter-profile release changed/missing')
 if profile['geometry']!={'kind':'native_saved_R0','weak_ray_guard':False,'runtime_grid_replacement':False}:raise ValueError('Only nativeR0 geometry is production qualified')
 from focal_scale import focal_scale
 if sha(Path(__file__).parent/'focal_scale.py')!=profile['pixel_scale']['focal_helper_sha256']:raise ValueError('Smooth angular pixel-scale helper changed')
 geometry=Path(spec['gathered_geometry'])/'frames'/f'{a.index:04d}';receipt=read(geometry/'geometry_complete.json')
 if receipt.get('kind')!='NATIVE_MODEL_BOUND_FLOAT_GEOMETRY_V1'or receipt['binding']['worker_sha256']!=release['native_R0_GPU_worker_sha256']:raise ValueError('Unqualified nativeR0 geometry producer')
 data=Path(spec['data_template'].format(frame_id=record['frame_id']));b=receipt['binding'];checkpoint=receipt['checkpoint_binding']
 if b['path_sha256']!=sha(a.path)or b['record']!=record or b['data_ready_sha256']!=sha(data/'ready.json')or b['transforms_sha256']!=sha(data/'transforms.json'):raise ValueError('Geometry and current same-frame source differ')
 if not checkpoint.get('active_hard140k_revision')or checkpoint['completed_iterations']>140000:raise ValueError('Wrong production checkpoint family')
 for name,digest in receipt['files_sha256'].items():
  if sha(geometry/name)!=digest:raise ValueError('Gathered float geometry changed')
 for name,source in receipt['source_rows'].items():
  if sha(data/'semantics'/(name+'.npz'))!=source['semantic_sha256']or sha(data/source['row']['mask_path'])!=source['valid_mask_sha256']:raise ValueError('Current donor semantic inputs differ from GPU package')
 def cache(model,row,destination,old,cp,chunk):
  metadata=read(destination.with_suffix('.json'))
  if metadata['binding']['row']!=row or metadata['binding']['checkpoint_sha256']!=cp or metadata['sha256']!=sha(destination):raise ValueError('Wrong cached donor/target rays')
  with np.load(destination,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
 R.geometry=cache
 def neural(spec,request,record,out):
  old=R.legacy(spec);cp=checkpoint['checkpoint_sha256'];calibration=profile['pixel_scale']['reference_calibration'];current={r['physical_camera']:r for r in read(data/'transforms.json')['frames']}
  for name,anchor in calibration.items():
   if any(abs(current[name][k]-value)>1e-10 for k,value in anchor.items()):raise ValueError('Fixed focal reference calibration differs from current source')
  pixel_scale,focal_proof=focal_scale(record['camera'],record['time_seconds'],calibration);target=cache(None,record['camera'],geometry/'target.npz',old,cp,2048);sources=R.Sources(data,record,spec,None,old,cp,geometry);old.native_sample=lambda raw,uv,row:R.photograph_samples(raw,uv,row,sources.fit);R.save(out/'lookcloser.png',target['rgb']);rgb,info,labels,weight=old.hybrid_image(out,sources,record['camera'],target,count=spec.get('source_count',16),mode='detail',sigma=profile['parameters']['detail_sigma'],gain=profile['parameters']['detail_gain'],pixel_scale=pixel_scale);R.save(out/'hybrid.png',rgb);np.savez_compressed(out/'transfer_audit.npz',source_labels=labels,transfer_weight=weight)
  return dict(kind='two_same_instant_variants_from_native_GPU_geometry',checkpoint_sha256=cp,checkpoint_binding=checkpoint,model_frame_id=record['frame_id'],geometry_receipt_sha256=sha(geometry/'geometry_complete.json'),hybrid=info,native_sources=sources.inventory,source_transforms_sha256=sha(data/'transforms.json'),geometry_model_native_host_preserved=True,CPU_native_RAW_from_shared_FSX=True,hybrid_params_profile_sha256=sha(Path(__file__).parent/'hybrid_params.json'),root_profile_release_sha256=sha(Path(__file__).parent/'root_release.json'),smooth_focal_proof=focal_proof,weak_grid_guard=False)
 R.render_neural=neural
 # Frozen render_one retains its output locks, binding checks and PNG hashes.
 R.render_one(spec,a.path,a.index)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['gpu','cpu']);p.add_argument('--code',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--path',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args();(gpu if a.mode=='gpu'else cpu)(a)
