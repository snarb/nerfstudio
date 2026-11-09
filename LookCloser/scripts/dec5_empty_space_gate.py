"""Frozen density gate from conservative train-camera background consensus."""
from pathlib import Path
import json
import numpy as np,torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion
from nerfstudio.field_components.field_heads import FieldHeadNames
from run_dec5_hair_campaign import sha


class EmptySpaceGate:
 def __init__(self,model,path,min_background_views,frame_id):
  path=Path(path);receipt=json.load(open(path.parent/'complete.json'))
  if str(frame_id)!=str(receipt['frame_id']) or sha(path)!=receipt['votes_sha256'] or receipt['uses_validation_rgb'] or receipt['uses_test_rgb']:raise ValueError('Unverified empty-space grid')
  audit=receipt['training_hair_witness_audit'].get(str(min_background_views))
  if audit is None or audit['reliable_train_hair_points']<100 or audit['raw_carved_fraction']>.005:raise ValueError('Train-hair witness safety gate failed')
  d=np.load(path);known=(d['background_votes']>=min_background_views)&(d['visible_votes']>=8);known=binary_erosion(known,structure=np.ones((3,3,3),bool),border_value=0);device=next(model.parameters()).device;self.volume=torch.as_tensor(known,device=device,dtype=torch.float32)[None,None];self.bounds=torch.as_tensor(d['bounds'],device=device,dtype=torch.float32);self.meta=dict(votes_sha256=sha(path),min_background_views=min_background_views,min_visible_views=8,erosion_voxels=1,density_gate='one minus trilinear known-empty volume; density unchanged outside grid; base occupancy remains frozen',known_empty_fraction=float(known.mean()),train_hair_witness_audit=audit);self.d=d
  self.handle=model.field.register_forward_hook(self.hook)
 def check_normalization(self,rotation,translation,scale):
  np.testing.assert_allclose(rotation,self.d['rotation'],rtol=0,atol=1e-5);np.testing.assert_allclose(translation,self.d['translation'],rtol=0,atol=1e-5);np.testing.assert_allclose(scale,self.d['scale'],rtol=1e-5,atol=1e-7)
 def gate(self,points):
  shape=points.shape[:-1];p=points.reshape(-1,3).float();values=[]
  for part in p.split(65536):
   norm=2*(part-self.bounds[0])/(self.bounds[1]-self.bounds[0])-1;inside=(norm.abs()<=1).all(-1);sample=F.grid_sample(self.volume,norm.reshape(1,-1,1,1,3),mode='bilinear',padding_mode='border',align_corners=False).reshape(-1);values.append(torch.where(inside,1-sample,torch.ones_like(sample)))
  return torch.cat(values).reshape(shape) if values else p.new_empty(shape)
 def hook(self,module,args,result):
  output=dict(result);density=output[FieldHeadNames.DENSITY];gate=self.gate(args[0].frustums.get_positions());output[FieldHeadNames.DENSITY]=density*gate.reshape_as(density);return output
