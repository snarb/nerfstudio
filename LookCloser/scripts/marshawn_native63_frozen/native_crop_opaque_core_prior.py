"""Optional deficit-only adaptation of green opacity recipe, not measured alpha GT.

A soft opaque-material assumption on32-native-pixel eroded actor interiors.
No profile term, field query, random draw, model parameter or registered buffer.
"""
from pathlib import Path
import hashlib,json
EROSION=32
COEFFICIENTS=(0,.01,.1)
KIND='NATIVE63_OPAQUE_INTERIOR_ASSUMPTION_V1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ramp(virtual_step):return min(1.,max(0.,(virtual_step-4096)/4096))
def eroded_core(black,grey,valid):
 import numpy as np
 from scipy.ndimage import distance_transform_edt
 if black.dtype!=bool or grey.dtype!=bool or valid.dtype!=bool or black.shape!=grey.shape or black.shape!=valid.shape or black.ndim!=2:raise ValueError('Aligned native Boolean masks required')
 union=(black|grey)&valid
 # Padding forces both sensor/crop outside tofalse. EDT>32 is exact Euclidean
 # binary erosion by all pixel offsets within a closed radius32 disk.
 distance=distance_transform_edt(np.pad(union,1,constant_values=False))[1:-1,1:-1]
 return distance>EROSION

def verify_manifest(data,manifest_path,expected_sha):
 data,manifest_path=Path(data),Path(manifest_path)
 if sha(manifest_path)!=expected_sha:raise ValueError('Opaque-core manifest changed')
 m=read(manifest_path)
 required=dict(kind=KIND,erosion_native_pixels=32,erosion_metric='Euclidean closed disk; EDT>32; outsidefalse',coefficient_is_soft_material_assumption=True,measured_alpha_GT=False,profile_term=False,transforms_sha256=sha(data/'transforms.json'),box_sha256=sha(data/'box.json'),helper_sha256=sha(__file__))
 if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Opaque interior provenance/definition changed')
 ready=read(data/'ready.json');tf=read(data/'transforms.json');frames={f['physical_camera']:f for f in tf['frames']};train_names={f['physical_camera']for f in tf['frames']if f['file_path']in tf['train_filenames']};rows={r['camera']:r for r in ready['records']if r['camera']in train_names}
 if ready['train']!=63 or len(rows)!=63 or len(train_names)!=63 or train_names&{'DC','HC','KA'}or set(m.get('records',{}))!=set(rows):raise ValueError('Strict63 TRAIN-only opaque source required')
 import numpy as np
 for c,r in rows.items():
  rec=m['records'][c];p=manifest_path.parent/rec['core_file']
  if Path(rec['core_file']).is_absolute()or'..'in Path(rec['core_file']).parts:raise ValueError('Owned relative core file required')
  if rec['source_semantic_sha256']!=r['semantic_sha256']or sha(data/'semantics'/(c+'.npz'))!=rec['source_semantic_sha256']or sha(p)!=rec['core_sha256']:raise ValueError('Opaque core/source mask changed: '+c)
  with np.load(p,allow_pickle=False)as z:core=z['core']
  f=frames[c]
  if core.dtype!=bool or core.shape!=(f['h'],f['w'])or int(core.sum())!=rec['core_pixels']:raise ValueError('Opaque core layout changed: '+c)
 return m

def core_loss(accumulation,core,coefficient,virtual_step):
 """Mean only over selected interior rays; exterior has exactly zero gradient."""
 import torch
 if isinstance(coefficient,bool)or coefficient not in COEFFICIENTS:raise ValueError('Coefficient must be0/.01/.1')
 if accumulation.numel()!=core.numel()or core.dtype!=torch.bool:raise ValueError('One Boolean core flag per existing ray required')
 values=accumulation.reshape(-1);scale=coefficient*ramp(virtual_step)
 # flags() is CPU: test empty before transfer, avoiding a CUDA scalar sync.
 # GPU core inputs retain their existing semantics as a fallback.
 if scale==0 or not bool(core.any()):return values.sum()*0
 selected=core.reshape(-1).to(values.device)
 return scale*(-torch.log(values[selected].clamp_min(1e-6))).mean()

class OpaqueCorePrior:
 def __init__(self,pipeline,data,coefficient,manifest_path,manifest_sha256):
  import numpy as np
  self.coefficient=coefficient;self.virtual_step=0;self.manifest=verify_manifest(data,manifest_path,manifest_sha256);self.rows=[];self.cameras=[]
  for p in pipeline.datamanager.train_dataset.image_filenames:
   c=Path(p).stem
   if c not in self.manifest['records']:raise ValueError('Unexpected dataset camera '+c)
   r=self.manifest['records'][c]
   with np.load(Path(manifest_path).parent/r['core_file'],allow_pickle=False)as z:self.rows.append(z['core'].copy())
   self.cameras.append(c)
  if len(self.rows)!=63 or len(set(self.cameras))!=63 or set(self.cameras)&{'DC','HC','KA'}:raise ValueError('Strict63 unique training cameras required')
  self._pack_rows()
 def _pack_rows(self):
  import numpy as np
  self.heights=np.asarray([r.shape[0]for r in self.rows],dtype=np.int64);self.widths=np.asarray([r.shape[1]for r in self.rows],dtype=np.int64);sizes=self.heights*self.widths;self.offsets=np.cumsum(sizes)-sizes
  self.packed=np.concatenate([r.reshape(-1)for r in self.rows])
  for a in [*self.rows,self.packed,self.heights,self.widths,self.offsets]:a.setflags(write=False)
 def set_step(self,virtual_step):self.virtual_step=int(virtual_step)
 def flags(self,indices):
  import numpy as np,torch
  ids=indices.detach().cpu().long().numpy()
  if ids.ndim!=2 or ids.shape[1]!=3:raise ValueError('Existing absolute(camera,y,x) indices required')
  if len(ids)and((ids[:,0]<0).any()or(ids[:,0]>=len(self.rows)).any()):raise ValueError('Unknown dataset camera')
  c,y,x=ids.T
  if (y<0).any()or(x<0).any()or(y>=self.heights[c]).any()or(x>=self.widths[c]).any():raise ValueError('Existing ray outside native crop')
  return torch.from_numpy(self.packed[self.offsets[c]+y*self.widths[c]+x])

def install_opaque_core_prior(pipeline,data,coefficient,manifest_path,manifest_sha256):
 if coefficient not in (.01,.1)or isinstance(coefficient,bool):raise ValueError('Only explicit nonzero prior installs')
 if hasattr(pipeline,'native_opaque_core_prior'):raise ValueError('Prior already installed; refuse double loss')
 if not hasattr(pipeline,'marshawn_sampling_adapter'):raise ValueError('Install after existing actor RGB loss adapter')
 prior=OpaqueCorePrior(pipeline,data,coefficient,manifest_path,manifest_sha256);model=pipeline.model;original=model.get_loss_dict
 def get_loss(outputs,batch,metrics_dict=None):
  result=original(outputs,batch,metrics_dict)
  if model.training:
   flags=prior.flags(batch['indices'])
   result['opaque_core_deficit_loss']=core_loss(outputs['accumulation'],flags,prior.coefficient,prior.virtual_step)
  return result
 model.get_loss_dict=get_loss;pipeline.native_opaque_core_prior=prior
 return prior
