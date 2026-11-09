#!/usr/bin/env python3
"""One bounded CPU-only shared-k1 residual test; no pose/board/neural fitting."""
from pathlib import Path
import json,hashlib,itertools
import cv2,numpy as np
from scipy.optimize import minimize_scalar
import marshawn_printed_target_refine_independent as m

DEST=m.OUT/'shared_radial_k1_HOLDOUT_20261106'
HELDOUT_CAMERAS={'DC','HC','KA'}
PRIOR_SIGMA=.002

def groups(obs,heldout_ids,heldout_cameras=False):
 out=[]
 for mid in sorted(set(mid for c,mid in obs)):
  if heldout_ids is not None and (mid%5==0)!=heldout_ids:continue
  rows=sorted((c,x)for(c,mm),x in obs.items()if mm==mid)
  for (a,xa),(b,xb)in itertools.combinations(rows,2):
   if heldout_cameras:
    if not({m.names[a],m.names[b]}&HELDOUT_CAMERAS):continue
   elif heldout_ids is False and ({m.names[a],m.names[b]}&HELDOUT_CAMERAS):continue
   rr=m.R0[b]@m.R0[a].T;tt=m.t0[b]-rr@m.t0[a];S=np.array([[0,-tt[2],tt[1]],[tt[2],0,-tt[0]],[-tt[1],tt[0],0]])
   out.append((a,b,mid,xa,xb,S@rr))
 return out

def errors(rows,D):
 e=[];ids=[]
 for a,b,mid,xa,xb,E in rows:
  ua=cv2.undistortPoints(xa[:,None,:],m.K[a],D[a]).reshape(4,2);ub=cv2.undistortPoints(xb[:,None,:],m.K[b],D[b]).reshape(4,2)
  x=np.c_[ua,np.ones(4)];y=np.c_[ub,np.ones(4)];l=x@E.T;ll=y@E;z=np.sum(y*l,axis=1)
  e.extend(z*np.sqrt(.5*(1/np.sum(l[:,:2]**2,axis=1)+1/np.sum(ll[:,:2]**2,axis=1)))*np.sqrt(m.K[a,0,0]*m.K[b,0,0]));ids.extend([mid]*4)
 return np.array(e),np.array(ids)

def main():
 DEST.mkdir(exist_ok=False)
 D0=m.D.copy();frames={f:m.read(f)for f in [408,448,488]};train=groups(frames[408],False)
 protocol=dict(kind='One fixed shared residual k1, fixed poses/K/other distortion; independent raw detections; no square/plane assumptions',training_frame=408,training_ids='ID%5!=0',training_excluded_cameras=sorted(HELDOUT_CAMERAS),heldout_ids='ID%5==0',independent_observation_frames=[448,488],k1_bounds=[-.01,.01],prior_sigma=PRIOR_SIGMA,robust_scale_px=.5,
 acceptance_frozen_before_fit='All3 heldout-ID crossview medians improve>=.05rawpx; no heldout-ID epipolar or crossview median/p90 worsens; heldout-camera epi median/p90 no worse; majority>=60percent of heldoutID medians improve; optimum not at bound',
 semantics='Same detection/support/splits baseline/candidate; geometry for crossview scoring excludes scored camera; repeated staticframes are extra observations not independent3Dtruth; no source rig/apply')
 (DEST/'protocol_frozen.json').write_text(json.dumps(protocol,indent=2))
 def matrix(delta):
  d=D0.copy();d[:,0]+=delta;return d
 def objective(delta):
  e,_=errors(train,matrix(delta));z=(e/.5)**2
  return float(np.sum(2*(np.sqrt(1+z)-1))+(delta/PRIOR_SIGMA)**2)
 fit=minimize_scalar(objective,bounds=(-.01,.01),method='bounded',options={'xatol':1e-10});delta=float(fit.x);D1=matrix(delta)
 result=dict(protocol=protocol,success=bool(fit.success),delta_k1=delta,train_objective_before=objective(0),train_objective_after=float(fit.fun),training_corners=len(errors(train,D0)[0]),metrics={},camera_heldout_metrics={},per_heldout_id={},source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted((m.OUT/'detections_v2').glob('*.json'))},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 for f,obs in frames.items():
  test=groups(obs,True);camtest=groups(obs,None,True);be,mid=errors(test,D0);ae,_=errors(test,D1);ce,_=errors(camtest,D0);de,_=errors(camtest,D1)
  result['metrics'][f]=dict(baseline_epipolar_px=m.stat(np.abs(be)),candidate_epipolar_px=m.stat(np.abs(ae)))
  result['camera_heldout_metrics'][f]=dict(baseline=m.stat(np.abs(ce)),candidate=m.stat(np.abs(de)))
  result['per_heldout_id'][f]={int(i):dict(baseline=m.stat(np.abs(be[mid==i])),candidate=m.stat(np.abs(ae[mid==i])))for i in sorted(set(mid))}
  try:
   for key,d in [('baseline_crossview_px',D0),('candidate_crossview_px',D1)]:
    m.D=d;result['metrics'][f][key]=m.cross_reproject(obs,m.R0,m.t0)
  finally:m.D=D0
 reasons=[]
 for f,row in result['metrics'].items():
  if row['baseline_crossview_px']['median']-row['candidate_crossview_px']['median']<.05:reasons.append(f'{f}: crossview median gain below frozen .05px')
  for kind in ['epipolar_px','crossview_px']:
   for q in ['median','p90']:
    if row['candidate_'+kind][q]>row['baseline_'+kind][q]:reasons.append(f'{f}: heldoutID {kind} {q} worsens')
  for q in ['median','p90']:
   if result['camera_heldout_metrics'][f]['candidate'][q]>result['camera_heldout_metrics'][f]['baseline'][q]:reasons.append(f'{f}: heldout-camera {q} worsens')
  per=result['per_heldout_id'][f];fraction=np.mean([z['candidate']['median']<z['baseline']['median']for z in per.values()]);result['metrics'][f]['heldout_ID_fraction_improved']=float(fraction)
  if fraction<.6:reasons.append(f'{f}: fewer than60percent heldout IDs improve')
 if abs(delta)>.0099:reasons.append('Parameter reached bound')
 result['accepted']=not reasons;result['rejection_reasons']=reasons
 np.savez(DEST/'candidate_SHARED_K1_DO_NOT_APPLY.npz',names=m.names,K=m.K,D=D1,R=m.R0,t=m.t0)
 (DEST/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k]for k in ['success','delta_k1','training_corners','metrics','camera_heldout_metrics','accepted','rejection_reasons']},indent=2))
if __name__=='__main__':main()
