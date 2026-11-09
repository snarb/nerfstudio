"""Self-contained frozen Marshawn ray-RGB background; never a depth surface."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch import nn


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

class RayBackgroundNetwork(nn.Module):
 def __init__(self,origin_levels=1,direction_levels=5,width=64):
  super().__init__();self.origin_levels=origin_levels;self.direction_levels=direction_levels
  self.layers=nn.Sequential(nn.Linear(6+6*(origin_levels+direction_levels),width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,3),nn.Sigmoid())
 def forward(self,origins,directions):
  features=[origins,directions]
  for x,levels in [(origins,self.origin_levels),(directions,self.direction_levels)]:
   for k in range(levels):
    angle=x*(torch.pi*(2**k));features.extend([angle.sin(),angle.cos()])
  return self.layers(torch.cat(features,-1))

def fingerprint(network):
 h=hashlib.sha256()
 for k,v in sorted(network.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

class FrozenMarshawnRayBackground:
 """Separate dependency, intentionally not registered in foreground state_dict."""
 def __init__(self,checkpoint,device='cuda',expected_sha256=None,data=None):
  checkpoint=Path(checkpoint);self.checkpoint_sha256=sha(checkpoint)
  if expected_sha256 and expected_sha256!=self.checkpoint_sha256:raise ValueError('Background weights checksum mismatch')
  self.request=json.loads((checkpoint.parent/'request.json').read_text());complete=json.loads((checkpoint.parent/'complete.json').read_text())
  if complete['checkpoint_sha256']!=self.checkpoint_sha256 or self.request['uses_validation_rgb'] is not False or self.request['uses_test_rgb'] is not False or self.request['train_cameras']!=63:raise ValueError('Background provenance mismatch')
  if data is not None and sha(Path(data)/'transforms.json') not in {self.request['data_sha256'],self.request['allowed_serving_dataset_sha256']}:raise ValueError('Serving cameras do not match background fit')
  state=torch.load(checkpoint,map_location='cpu',weights_only=False)
  with torch.random.fork_rng(devices=[torch.device(device).index or torch.cuda.current_device()] if str(device).startswith('cuda') else []):self.network=RayBackgroundNetwork(state['origin_levels'],state['direction_levels'],state['width']).to(device)
  self.network.load_state_dict(state['state_dict'],strict=True);self.network.requires_grad_(False);self.network.eval();self.center=torch.tensor(state['center'],device=device,dtype=torch.float32);self.radius=float(state['radius'])
  if not self.radius>0:raise ValueError('Positive background origin normalization radius required')
  self.before=fingerprint(self.network)
 @torch.no_grad()
 def predict(self,origins,directions):
  direction=torch.nn.functional.normalize(directions.float(),dim=-1)
  return self.network((origins.float()-self.center)/self.radius,direction)
 def verify(self):
  if fingerprint(self.network)!=self.before:raise ValueError('Frozen background weights changed')

def attach_inference(pipe,checkpoint,expected_sha256=None):
 """Native volume C_fg + T*B; all training/eval/novel rays use predicted B only."""
 model=pipe.model
 if model.config.background_color!='black':raise ValueError('Foreground must be premultiplied against black')
 if getattr(model,'marshawn_background_provider',None) is not None:raise ValueError('Background already attached')
 data=Path(pipe.datamanager.config.data);meta=json.loads((data/'transforms.json').read_text());by={Path(r['file_path']).name:r for r in meta['frames']};ds=pipe.datamanager.train_dataset
 for i,path in enumerate(ds.image_filenames):
  row=by[Path(path).name];live=ds.cameras.camera_to_worlds[i].detach().cpu().numpy()
  if not np.allclose(live,np.array(row['transform_matrix'])[:3],atol=1e-6,rtol=0):raise ValueError('Provider requires unchanged prepared_world camera poses')
 provider=FrozenMarshawnRayBackground(checkpoint,device=model.device,expected_sha256=expected_sha256,data=data);original=model.get_outputs
 def outputs(rays):
  result=original(rays);a=result['accumulation'];a=a[...,None] if a.ndim==result['rgb'].ndim-1 else a;background=provider.predict(rays.origins,rays.directions)
  result['foreground_rgb']=result['rgb'];result['ray_background_rgb']=background;result['rgb']=result['rgb']+(1-a).clamp(0,1)*background
  return result
 model.get_outputs=outputs;model.marshawn_background_provider=provider
 return provider
