"""Opt-in frozen ray background and conservative foreground empty-space training."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np,torch
from dec5_ray_background import RayBackgroundNetwork
from dec5_foreground_training import integrate_packed_thickness
from run_dec5_hair_campaign import sha,atomic_json


def network_fingerprint(network):
 h=hashlib.sha256()
 for k,v in sorted(network.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()


def integrate_fixed_thickness(density,length,num_rays):
 """The fixed warmup samples uniformly across each ray's physical interval."""
 density=torch.relu(density).reshape(num_rays,-1)
 return density.mean(-1)*length.reshape(num_rays)


class FrozenRayBackground:
 def __init__(self,pipe,checkpoint):
  data=pipe.datamanager.config.data;receipt=json.load(open(checkpoint.parent/'request.json'))
  if sha(data/'transforms.json') not in {receipt['allowed_serving_dataset_sha256'],receipt['data_sha256']} or receipt['uses_validation_rgb'] or receipt['uses_test_rgb']:raise ValueError('Background fit/data mismatch')
  state=torch.load(checkpoint,map_location='cpu',weights_only=False)
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   self.network=RayBackgroundNetwork(state['origin_levels'],state['direction_levels'],state['width']).to(pipe.device)
  self.network.load_state_dict(state['state_dict'],strict=True);self.network.requires_grad_(False);self.network.eval();self.before=network_fingerprint(self.network)
  payload=json.load(open(data/'transforms.json'));by={Path(f['file_path']).name:f for f in payload['frames']};frames=[by[Path(p).name] for p in pipe.datamanager.train_dataset.image_filenames];raw=np.array([f['transform_matrix'] for f in frames]);normal=pipe.datamanager.train_dataset.cameras.camera_to_worlds.cpu().numpy();R=normal[0,:3,:3]@raw[0,:3,:3].T;scale=np.linalg.norm(normal[1,:3,3]-normal[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);t=normal[0,:3,3]-scale*R@raw[0,:3,3]
  if np.max(np.linalg.norm(scale*raw[:,:3,3]@R.T+t-normal[:,:3,3],axis=-1))>1e-4:raise ValueError('Background camera alignment mismatch')
  self.rotation=torch.as_tensor(R,device=pipe.device,dtype=torch.float32);self.translation=torch.as_tensor(t,device=pipe.device,dtype=torch.float32);self.scale=float(scale);self.center=torch.as_tensor(state['center'],device=pipe.device,dtype=torch.float32);self.radius=float(state['radius']);self.checkpoint_sha256=sha(checkpoint);self.frames=frames

 @torch.no_grad()
 def predict(self,origins,directions):
  raw_origin=((origins-self.translation)@self.rotation)/self.scale;raw_direction=torch.nn.functional.normalize(directions@self.rotation,dim=-1)
  return self.network((raw_origin-self.center)/self.radius,raw_direction)

 def verify(self):
  if network_fingerprint(self.network)!=self.before:raise ValueError('Frozen background weights changed')


def attach_inference(pipe,checkpoint,volume_dependency=None):
 model=pipe.model
 if model.config.background_color!='black':raise ValueError('Separate background requires premultiplied black-background foreground')
 provider=FrozenRayBackground(pipe,checkpoint)
 if volume_dependency is not None:
  from dec5_neural_background_provider import HybridFrozenBackground
  provider=HybridFrozenBackground(provider,pipe,volume_dependency)
 original=model.get_outputs
 def outputs(rays):
  v=original(rays);a=v['accumulation'];a=a[...,None] if a.ndim==v['rgb'].ndim-1 else a
  background=provider.predict(rays.origins,rays.directions);v['ray_background_rgb']=background
  v['rgb']=v['rgb']+(1-a).clamp(0,1)*background
  return v
 model.get_outputs=outputs;model.dec5_background_provider=provider
 return provider


def install_training(pipe,checkpoint,semantics,output,weight=.05,known_background_rgb=False,volume_dependency=None):
 from nerfstudio.field_components.field_heads import FieldHeadNames
 provider=attach_inference(pipe,checkpoint,volume_dependency);model=pipe.model;known=[];width=[];height=[];records=[]
 for f in provider.frames:
  p=semantics/(f['physical_camera']+'.npz');v=np.load(p)['probability'].astype(np.float32)/65535;mask=cv2.erode((v[...,2]>.98).astype(np.uint8),np.ones((33,33),np.uint8))>0
  known.append(mask);width.append(f['w']);height.append(f['h']);records.append(dict(camera=f['physical_camera'],semantic_sha256=sha(p),semantic_raster=list(mask.shape),erode_radius_semantic_pixels=16,known_bg_pixels=int(mask.sum())))
 captured={};wrapped=model.get_outputs
 def outputs(rays):
  if model.training:
   captured.clear();captured['length']=(rays.fars-rays.nears).reshape(-1)
  return wrapped(rays)
 model.get_outputs=outputs
 original_query=model.field.query_points
 def query_points(*args,**kwargs):
  density,rgb=original_query(*args,**kwargs)
  if model.training:captured['fixed_density']=density
  return density,rgb
 model.field.query_points=query_points
 def field_hook(module,args,v):
  if model.training:captured['packed']=(v[FieldHeadNames.DENSITY],args[0].frustums.ends-args[0].frustums.starts)
 model.field.register_forward_hook(field_hook);original_loss=model.get_loss_dict;counts=dict(batches=0,known_bg_rays=0,total_rays=0)
 def loss(outputs,batch,metrics=None):
  indices=batch['indices'].detach().cpu().numpy();mask=np.zeros(len(indices),bool)
  for i in np.unique(indices[:,0]):
   pick=np.nonzero(indices[:,0]==i)[0];m=known[int(i)];x=np.clip(((indices[pick,2]+.5)*m.shape[1]/width[int(i)]).astype(int),0,m.shape[1]-1);y=np.clip(((indices[pick,1]+.5)*m.shape[0]/height[int(i)]).astype(int),0,m.shape[0]-1);mask[pick]=m[y,x]
  loss_outputs=outputs
  if known_background_rgb and model.training:
   loss_outputs=dict(outputs);a=outputs['accumulation'].reshape(-1,1);use=torch.as_tensor(mask,device=pipe.device)
   loss_outputs['rgb']=outputs['rgb']+torch.where(use[:,None],(1-a).clamp(0,1)*(batch['image'].to(outputs['rgb'])-outputs['ray_background_rgb']),torch.zeros_like(outputs['rgb']))
  losses=original_loss(loss_outputs,batch,metrics)
  if 'packed_ray_indices' in outputs:
   if len(outputs['packed_ray_indices']):
    density,deltas=captured.pop('packed');thickness=integrate_packed_thickness(density,deltas,outputs['packed_ray_indices'].long(),len(indices))
   else:thickness=outputs['accumulation'].reshape(-1)*0
  else:thickness=integrate_fixed_thickness(captured['fixed_density'],captured['length'],len(indices))
  use=torch.as_tensor(mask,device=pipe.device);losses['known_background_empty_space']=float(weight)*(thickness[use].mean() if mask.any() else thickness.sum()*0)
  counts['batches']+=1;counts['known_bg_rays']+=int(mask.sum());counts['total_rays']+=len(mask)
  if metrics is not None:metrics['known_background_fraction']=use.float().mean().detach()
  captured.clear();return losses
 model.get_loss_dict=loss
 atomic_json(output/'background_separation.json',dict(background_checkpoint_sha256=sha(checkpoint),volume_dependency=volume_dependency,records=records,empty_space_weight=weight,known_background_rgb=known_background_rgb,
  empty_space='full packed optical thickness on train BG probability >.98 eroded 16 semantic-raster pixels; sampled rays mapped to that raster; no positive hard alpha targets',
  composition='premultiplied foreground NeRF RGB + (1-foreground accumulation) * frozen ray background',
  additional_foreground_density_queries=0,foreground_scope='existing bounded scene; background is a separate ray-color function, not a depth surface',uses_validation_rgb=False,uses_test_rgb=False))
 return provider,counts


def resolve_background(request,data,training):
 entry=request.get('ray_background')
 if entry is None and 'fr_switch_after' in request and (training/'background_dependency.json').is_file():
  entry=json.loads((training/'background_dependency.json').read_text())
 if not entry:return None
 for p in [Path(entry['path']),data.parent/'ray_background/background_portable.pt',training.parent/'ray_background/background_portable.pt',data.parent/'ray_background/background.pt']:
  if p.exists() and sha(p)==entry['sha256']:return p
 raise ValueError('Missing hash-matched frozen ray-background checkpoint')
