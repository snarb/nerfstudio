"""Finite semantic RGB-loss confidence; NEVER alpha labels or RGB compositing."""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
P=math.log(.1)/math.log(.5)

def outside_profile(distance,width):
 d=np.asarray(distance,np.float64)
 if not math.isfinite(width)or width<0 or not np.isfinite(d).all()or(d<0).any():raise ValueError('Nonnegative finite distances/width required')
 if width==0:return np.zeros_like(d,dtype=np.float32)
 return np.maximum(1-d/width,0).clip(0,1).astype(np.float64).__pow__(P).astype(np.float32)

def native_confidence(foreground,source_ineligible,width_native,*,common_proposal=None):
 """Core1; continuous outsidefalloff .1athalfwidth,0atwidth; veto overridesALL.

`source_ineligible` means other-person/ambiguous-occluder/invalid source, NOT
hair merely missed by the nominal segmentation. Unknown hair belongs in the
finite soft band, without a forced alpha=0 or background RGB target.
 """
 fg=np.asarray(foreground);veto=np.asarray(source_ineligible)
 if fg.dtype!=np.bool_ or veto.dtype!=np.bool_ or fg.shape!=veto.shape or fg.ndim!=2:raise ValueError('Equal binary2D fields required')
 if not np.isfinite(width_native)or width_native<0:raise ValueError('Finitewidth>=0 required')
 out=np.zeros(fg.shape,np.float32)
 if fg.any():
  yy,xx=np.where(fg);pad=int(math.ceil(width_native))+2;y0=max(0,int(yy.min())-pad);y1=min(fg.shape[0],int(yy.max())+pad+1);x0=max(0,int(xx.min())-pad);x1=min(fg.shape[1],int(xx.max())+pad+1);part=fg[y0:y1,x0:x1]
  distance=distance_transform_edt(~part);w=outside_profile(distance,float(width_native));w[part]=1;out[y0:y1,x0:x1]=w
 out[veto]=0
 if common_proposal is None:proposal=out>0
 else:
  proposal=np.asarray(common_proposal)
  if proposal.dtype!=np.bool_ or proposal.shape!=fg.shape or np.any((out>0)&~proposal) or np.any(proposal&veto):raise ValueError('Commonproposal must containALLpositivepolicyweights')
 core=fg&~veto;band=(~fg)&(out>0);core_sum=float(out[core].sum(dtype=np.float64));outside_sum=float(out[band].sum(dtype=np.float64));count=int(proposal.sum());stats={'width_native_pixels':float(width_native),'exponent':P,'core_pixels':int(core.sum()),'outside_eligible_pixels':int(band.sum()),'source_ineligible_pixels':int(veto.sum()),'eligible_pixels':count,'positive_policy_weight_pixels':int((out>0).sum()),'sum_core_weights':core_sum,'sum_outside_weights':outside_sum,'outside_effective_weight_ratio':outside_sum/max(core_sum,1e-30),'outside_sampling_fraction_uniform_proposal':float((proposal&~fg).sum()/max(1,count)),'proposal_mean_weight':float(out.sum(dtype=np.float64)/max(1,count)),'sampling':'uniformCOMMONproposal(mask mayincludezero-weightpolicy pixels);confidenceweightedONCE/fixedglobalmean-normalization','NOT_alpha_GT':True,'RGB_target_unchanged':True}
 return out,stats

def weighted_rgb_mse(pred,target,confidence,*,proposal_mean_weight):
 """Unbiased under uniform eligible-pixel proposal (no confidence-based sampling).

Fixed mean confidence =sumeligiblew/Neligible for this train dataset/proposal.
Batch self-normalization sum(batchw) is intentionally avoided. No target edit.
 """
 import torch
 if pred.shape!=target.shape or pred.shape[-1]!=3:raise ValueError('Matched RGB3 required')
 w=confidence.to(pred.device,dtype=torch.float32).reshape(-1)
 if w.numel()!=pred.reshape(-1,3).shape[0]or not torch.isfinite(w).all()or(w<0).any()or(w>1).any():raise ValueError('Valid pixelconfidence required')
 if not math.isfinite(float(proposal_mean_weight))or not 0<float(proposal_mean_weight)<=1:raise ValueError('Fixedpositiveproposalmean required')
 with torch.autocast(device_type=pred.device.type,enabled=False):
  return ((pred.float().reshape(-1,3)-target.float().reshape(-1,3)).square().mean(-1)*w).mean()/float(proposal_mean_weight)
