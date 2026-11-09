"""Frozen-region supplemental metrics. No rendering, training or source editing.

Use original float RGB/alpha in worker.evaluate, before saving quantized PNGs.
Call after the unchanged primary equal-actor/equal-view evaluator. This module
never supplies masks/ROIs to prediction or training.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_regions(path):
 d=json.loads(Path(path).read_text())
 if d.get('kind')!='FROZEN_SOURCE_ONLY_NATIVE_HELDOUT_REGIONS_V2' or not d.get('training_use_forbidden'):raise ValueError('Invalid metric region manifest')
 keys=[(r['camera'],r['region']) for r in d['rois']]
 if len(set(keys))!=len(keys):raise ValueError('Duplicate region')
 return d

def supplemental_metrics(prediction,target,alpha,semantic,camera,lpips_model,regions):
 """HWC floatRGB/HW[1] alpha; exact per-region actor&valid intersection."""
 import torch
 from scipy.ndimage import distance_transform_edt
 from marshawn_tuning_metrics import masked_actor_metrics
 if camera not in ('DC','HC','KA'):raise ValueError('Heldout cameras only')
 if prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[-1]!=3:raise ValueError('Native matching HWC required')
 h,w=prediction.shape[:2];a=alpha.detach().float().cpu().numpy().reshape(h,w)
 if not np.isfinite(a).all():raise ValueError('Nonfinite accumulation in opacity diagnostic')
 raw_min,raw_max=float(a.min()),float(a.max());tolerance=1e-4
 raw_range=dict(raw_min=raw_min,raw_max=raw_max,below_zero_pixels=int((a<0).sum()),above_one_pixels=int((a>1).sum()),allowed_absolute_roundoff=tolerance,clamp_scope='CPU opacity diagnostics only; RGB/primary metrics/model untouched')
 if raw_min < -tolerance or raw_max > 1+tolerance:
  raise ValueError('Accumulation outside diagnostic roundoff tolerance: '+json.dumps(raw_range,sort_keys=True))
 a=np.clip(a,0.,1.)  # New CPU array; never edits tensor/model/render buffers.
 union=semantic['black']|semantic['grey'];valid=semantic['valid'];rows=[]
 for r in regions['rois']:
  if r['camera']!=camera:continue
  x0,y0,x1,y1=r['xyxy']
  if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('Frozen region outside native raster')
  mask=((union if r['actor']=='union' else semantic[r['actor']])&valid)[y0:y1,x0:x1].copy()
  if mask.dtype!=np.bool_ or not mask.any():raise ValueError('Empty/invalid frozen region intersection')
  p=prediction[y0:y1,x0:x1].permute(2,0,1)[None];g=target[y0:y1,x0:x1].permute(2,0,1)[None]
  q=masked_actor_metrics(p,g,torch.from_numpy(mask).to(prediction.device)[None,None],lpips_model,crop_xyxy=[0,0,x1-x0,y1-y0])
  rows.append(dict(camera=camera,region=r['region'],actor=r['actor'],xyxy=r['xyxy'],actor_valid_pixels=int(mask.sum()),**{k:q[k] for k in ('psnr','ssim','lpips')},domain='native float RGB; fixed local actor&valid intersection; RGB.5 exterior; AlexNet without resize'))
 opacity=[]
 for actor in ('black','grey'):
  mask=semantic[actor]&valid
  for radius in (3,32):
   # Native Euclidean erosion, outside sensor/crop is strictly false.
   core=distance_transform_edt(np.pad(mask,1,constant_values=False))[1:-1,1:-1]>radius
   values=a[core]
   opacity.append(dict(camera=camera,actor=actor,erosion_native_pixels=radius,pixels=int(core.sum()),mean_alpha=float(values.mean()) if len(values) else None,below09_fraction=float((values<.9).mean()) if len(values) else None,below05_fraction=float((values<.5).mean()) if len(values) else None,alpha_is_not_measured_GT=True,raw_accumulation_range=raw_range))
 return dict(regions=rows,opacity_diagnostics=opacity,raw_accumulation_range=raw_range)

def regression_flags(candidate,control,relative_threshold=.05):
 """Review flags only; never selects/rejects a scientific result silently."""
 key=lambda r:(r['camera'],r['region'])
 c={key(r):r for r in candidate};b={key(r):r for r in control}
 if set(c)!=set(b):raise ValueError('Metric region set mismatch')
 out=[]
 for k in sorted(c):
  if c[k]['xyxy']!=b[k]['xyxy'] or c[k]['actor_valid_pixels']!=b[k]['actor_valid_pixels'] or c[k]['domain']!=b[k]['domain']:raise ValueError('Region domain changed')
  old,new=b[k]['lpips'],c[k]['lpips'];out.append(dict(camera=k[0],region=k[1],candidate_lpips=new,control_lpips=old,relative_change=(new/old-1) if old else None,visual_review_required=new>old*(1+relative_threshold),automatic_veto=False))
 return out
