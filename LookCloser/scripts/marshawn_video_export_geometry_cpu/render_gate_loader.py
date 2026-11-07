"""Technical SAME-frame checkpoint loader; never creates production completion."""
import hashlib,json,os
from pathlib import Path
import numpy as np


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
 return h.hexdigest()


def tree_hash(value):
 import torch
 h=hashlib.sha256()
 def visit(v):
  if torch.is_tensor(v):
   x=v.detach().cpu().contiguous();h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
  elif isinstance(v,(list,tuple)):
   for x in v:visit(x)
  elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
  else:h.update(repr(v).encode())
 visit(value);return h.hexdigest()


def load_pipeline(frame_root,checkpoint_kind='technical',device='cuda:0'):
 import torch
 from PIL import Image
 from nerfstudio.utils.eval_utils import eval_setup
 from marshawn_video_train import verify_code
 from marshawn_video_path import write
 import marshawn_video_render as renderer
 gate=json.loads(Path(os.environ['MARSHAWN_TECHNICAL_GATE']).read_text());root=Path(gate['campaign']);out=Path(gate['output'])
 if Path(frame_root).resolve()!=root/'frames/001660':raise ValueError('Technical gate is restricted to original1660')
 verify_code(root)
 for path,digest in gate['files_sha256'].items():
  if sha(path)!=digest:raise ValueError('Technical input/source changed: '+path)
 checkpoint=Path(gate['checkpoint']);payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if payload['step']!=gate['step']:raise ValueError('Wrong technical step')
 expected=tree_hash(payload['pipeline'])
 def selected(config):config.eval_checkpoint=checkpoint;return config
 config,pipe,actual,step=eval_setup(Path(gate['config']),test_mode='inference',update_config_callback=selected)
 if Path(actual).resolve()!=checkpoint.resolve() or step!=gate['step']:raise ValueError('Wrong checkpoint selected')
 pipe.eval();pipe.requires_grad_(False);model=pipe.model;model.current_train_step=step
 strength=1. if step<75941 else .3
 model.config.feature_reweighting_strength=strength;model.field.feature_reweighting_strength=strength
 if tree_hash(pipe.state_dict())!=expected:raise ValueError('Exact pipeline checkpoint restoration failed')
 out.mkdir(parents=True,exist_ok=True)
 dataset=pipe.datamanager.eval_dataset
 index=next(i for i,p in enumerate(dataset.image_filenames) if Path(p).stem.split('_')[-1]=='HC')
 camera=dataset.cameras[index:index+1].to(model.device)
 with torch.inference_mode():values=model.get_outputs_for_camera(camera)
 rgb=(values['rgb'].clamp(0,1).cpu().numpy()*255+.5).astype(np.uint8)
 archived=np.asarray(Image.open(gate['reference_HC']).convert('RGB'));delta=abs(rgb.astype(np.int16)-archived.astype(np.int16))
 Image.fromarray(rgb).save(out/'HC_restored_native_HD.png')
 native_parity=dict(max_code_difference=int(delta.max()),changed_fraction=float((delta!=0).mean()))
 write(out/'native_HD_parity.json',native_parity)
 if delta.max()>1:raise ValueError('Archived same-checkpoint HC monitor parity failed')
 rows=json.loads((root/'frames/001660/data/transforms.json').read_text())['frames'];row=next(r for r in rows if r['physical_camera']=='HC')
 spec=json.loads(Path(gate['render_spec']).read_text());old=renderer.legacy(spec)
 geometry=renderer.geometry(model,row,out/'HC_packed_geometry.npz',old,gate['checkpoint_sha256'],spec['ray_chunk'])
 packed=np.rint(np.clip(geometry['rgb'],0,1)*255).astype(np.uint8);diff=abs(packed.astype(np.int16)-rgb.astype(np.int16))
 packed_parity=dict(max_code_difference=int(diff.max()),changed_fraction=float((diff!=0).mean()))
 write(out/'packed_HD_parity.json',packed_parity)
 if diff.max()>1:raise ValueError('Packed camera rays differ from actual model HC rays')
 if tree_hash(pipe.state_dict())!=expected:raise ValueError('Read-only parity mutated pipeline')
 del payload
 pipe._technical_expected_tree=expected
 binding=dict(frame=1660,technical_only=True,step=step,checkpoint=str(checkpoint),checkpoint_sha256=gate['checkpoint_sha256'],
  config=gate['config'],code=str(root/'code'),feature_reweighting_strength=strength,current_train_step=step,
  pipeline_tree_hash=expected,native_HD_parity=native_parity,packed_HD_parity=packed_parity,
  production_frame_completion_created=False,quality_selection_claim=False)
 write(out/'technical_loader_complete.json',binding)
 return pipe,binding
