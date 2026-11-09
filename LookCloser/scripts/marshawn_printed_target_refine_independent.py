#!/usr/bin/env python3
"""Isolated, regularized camera/individual-square refinement with held-out cards."""
import json,sys,itertools
from pathlib import Path
import numpy as np,cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from marshawn_printed_target_independent import rig,OUT
C=rig();names=sorted(C);ni={n:i for i,n in enumerate(names)}
R0=np.array([C[n]['R'] for n in names]);t0=np.array([C[n]['t'] for n in names]);K=np.array([C[n]['K'] for n in names]);D=np.array([C[n]['d'] for n in names]);N=len(names)
def read(frame):
 obs={}
 for p in (OUT/'detections_v2').glob(f'*_{frame}.json'):
  a=json.loads(p.read_text());cam=a['camera']
  for m in a['markers']:obs[(ni[cam],m['id'])]=np.array(m['corners'])
 return obs
def project(X,ci,R,t):
 Y=np.einsum('nij,nj->ni',R[ci],X)+t[ci];x=Y[:,0]/Y[:,2];y=Y[:,1]/Y[:,2];r=x*x+y*y;k=D[ci];rad=1+k[:,0]*r+k[:,1]*r*r
 u=x*rad+2*k[:,2]*x*y+k[:,3]*(r+2*x*x);v=y*rad+k[:,2]*(r+2*y*y)+2*k[:,3]*x*y
 return np.stack([K[ci,0,0]*u+K[ci,0,2],K[ci,1,1]*v+K[ci,1,2]],1)
def triang(obs,R,t):
 A=[]
 for ci,xy in obs:
  uv=cv2.undistortPoints(np.float64(xy).reshape(1,1,2),K[ci],D[ci]).ravel();P=np.c_[R[ci],t[ci]];A.extend([uv[0]*P[2]-P[0],uv[1]*P[2]-P[1]])
 _,_,v=np.linalg.svd(A);return v[-1,:3]/v[-1,3]
def stat(e):
 e=np.array(e);return dict(n=len(e),median=float(np.median(e)),mean=float(np.mean(e)),p90=float(np.percentile(e,90)),rms=float(np.sqrt(np.mean(e*e)))) if len(e) else {}
def epipolar(obs,R,t,heldout=True):
 errors=[]
 for mid in sorted(set(m for c,m in obs)):
  if heldout and mid%5!=0:continue
  rows=[(c,x) for (c,m),x in obs.items() if m==mid]
  for (a,xa),(b,xb) in itertools.combinations(rows,2):
   ua=cv2.undistortPoints(xa[:,None,:],K[a],D[a]).reshape(4,2);ub=cv2.undistortPoints(xb[:,None,:],K[b],D[b]).reshape(4,2)
   rr=R[b]@R[a].T;tt=t[b]-rr@t[a];S=np.array([[0,-tt[2],tt[1]],[tt[2],0,-tt[0]],[-tt[1],tt[0],0]]);E=S@rr
   x=np.c_[ua,np.ones(4)];y=np.c_[ub,np.ones(4)];l=x@E.T;ll=y@E;z=np.sum(y*l,axis=1);e=np.abs(z)*np.sqrt(.5*(1/np.sum(l[:,:2]**2,axis=1)+1/np.sum(ll[:,:2]**2,axis=1)))*np.sqrt(K[a,0,0]*K[b,0,0]);errors.extend(e)
 return stat(errors)
def cross_reproject(obs,R,t,details=False):
 errors=[];records=[]
 for mid in sorted(set(m for c,m in obs)):
  if mid%5!=0:continue
  rows=sorted((c,x) for (c,m),x in obs.items() if m==mid)
  for parity in [0,1]:
   tr=rows[parity::2];te=rows[1-parity::2]
   if len(tr)<3:continue
   for k in range(4):
    X=triang([(c,x[k]) for c,x in tr],R,t);ci=np.array([c for c,x in tr]);xy=np.array([x[k] for c,x in tr]);res=least_squares(lambda v:(project(np.tile(v,(len(ci),1)),ci,R,t)-xy).ravel(),X,loss='soft_l1',f_scale=2,max_nfev=30)
    for c,x in te:
     pred=project(res.x[None],np.array([c]),R,t)[0];e=np.linalg.norm(pred-x[k]);errors.append(e);records.append(dict(camera=names[c],marker=mid,corner=k,observed=x[k].tolist(),predicted=pred.tolist(),error=float(e),fit_cameras=[names[z] for z,_ in tr]))
 return records if details else stat(errors)
def main():
 obs=read(408);mids=sorted(m for m in set(m for c,m in obs) if m%5!=0 and sum(mm==m for c,mm in obs)>=4);mi={m:i for i,m in enumerate(mids)}
 base=np.array([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]])
 bparams=[];good=[]
 for m in mids:
  rows=[(c,x) for (c,mm),x in obs.items() if mm==m];X=np.array([triang([(c,x[k]) for c,x in rows],R0,t0) for k in range(4)]);cen=X.mean(0);U,s,V=np.linalg.svd(base.T@(X-cen));r=V.T@U.T
  if np.linalg.det(r)<0:V[-1]*=-1;r=V.T@U.T
  size=np.sum((base@r.T)*(X-cen))/np.sum(base**2);err=np.linalg.norm(project(np.tile(X,(len(rows),1)),np.repeat([c for c,x in rows],4),R0,t0)-np.concatenate([x for c,x in rows]),axis=1)
  if np.median(err)>5 or size<=0:continue
  good.append(m);bparams.append(np.r_[Rotation.from_matrix(r).as_rotvec(),cen,np.log(size)])
 mids=good;mi={m:i for i,m in enumerate(mids)};ci=[];bi=[];ki=[];xy=[]
 for (c,m),x in obs.items():
  if m not in mi:continue
  for k in range(3):ci.append(c);bi.append(mi[m]);ki.append(k);xy.append(x[k])
 ci=np.array(ci);bi=np.array(bi);ki=np.array(ki);xy=np.array(xy);B=len(mids);x0=np.r_[np.zeros(N*6),np.array(bparams).ravel()]
 def unpack(x):
  cp=x[:N*6].reshape(N,6);bp=x[N*6:].reshape(B,7);R=Rotation.from_rotvec(cp[:,:3]).as_matrix()@R0;t=t0+cp[:,3:];br=Rotation.from_rotvec(bp[:,:3]).as_matrix();X=np.einsum('nij,nj->ni',br[bi],base[ki]*np.exp(bp[bi,6,None]))+bp[bi,3:6];return R,t,X
 def fun(x):
  R,t,X=unpack(x);cp=x[:N*6].reshape(N,6);pri=cp/np.array([.003]*3+[.03]*3);pri[ni['HC']]*=100
  return np.r_[(project(X,ci,R,t)-xy).ravel(),pri.ravel()]
 S=lil_matrix((len(ci)*2+N*6,len(x0)),dtype=int)
 for i,(c,b) in enumerate(zip(ci,bi)):S[2*i:2*i+2,c*6:c*6+6]=1;S[2*i:2*i+2,N*6+b*7:N*6+b*7+7]=1
 S[len(ci)*2:,:N*6]=np.eye(N*6,dtype=int)
 print('start',N,B,len(ci),'initial',stat(np.linalg.norm(fun(x0)[:len(ci)*2].reshape(-1,2),axis=1)),flush=True)
 fit=least_squares(fun,x0,jac_sparsity=S.tocsr(),loss='soft_l1',f_scale=2,x_scale='jac',max_nfev=150,verbose=1,ftol=1e-7)
 R,t,X=unpack(fit.x);summary=dict(cameras=N,training_markers=mids,training_corners=len(ci),gauge='HC strongly anchored; all camera rotation/translation increments regularized sigma .003 rad/.03 original rig units; intrinsics/distortion fixed; individual marker sizes unknown and independently optimized; no metric dimensions asserted',optimization=dict(success=bool(fit.success),message=fit.message,nfev=fit.nfev),metrics={})
 for frame in [408,448,488]:
  oo=read(frame);summary['metrics'][frame]=dict(observations=len(oo),baseline_heldout_card_epipolar_px=epipolar(oo,R0,t0),refined_heldout_card_epipolar_px=epipolar(oo,R,t),baseline_heldout_card_crossview_reprojection_px=cross_reproject(oo,R0,t0),refined_heldout_card_crossview_reprojection_px=cross_reproject(oo,R,t))
 for tag,rr,tt in [('baseline',R0,t0),('refined',R,t)]:
  records=cross_reproject(obs,rr,tt,details=True);(OUT/f'{tag}_heldout_predictions.json').write_text(json.dumps(records,indent=2));summary[tag+'_heldout_per_camera']={n:stat([z['error'] for z in records if z['camera']==n]) for n in names};summary['detection_coverage']={n:dict(markers=sum(c==ni[n] for c,m in obs),heldout_markers=sum(c==ni[n] and m%5==0 for c,m in obs),raw_marker_bbox=None if not any(c==ni[n] for c,m in obs) else np.concatenate([x for (c,m),x in obs.items() if c==ni[n]]).min(0).tolist()+np.concatenate([x for (c,m),x in obs.items() if c==ni[n]]).max(0).tolist()) for n in names}
 # Fourth corner is excluded from fitting every training marker.
 for tag,rr,tt in [('baseline',R0,t0),('refined',R,t)]:
  err=[]
  # Geometry must be estimated separately under each camera model using only train corners.
  for b,m in enumerate(mids):
   rows=[(c,x) for (c,mm),x in obs.items() if mm==m];cc=np.array([c for c,x in rows]);xx=np.array([x for c,x in rows]);p0=np.array(bparams[b]);
   def pp(p):return base@Rotation.from_rotvec(p[:3]).as_matrix().T*np.exp(p[6])+p[3:6]
   def ff(p):return (project(np.tile(pp(p)[:3],(len(cc),1)),np.repeat(cc,3),rr,tt)-xx[:,:3].reshape(-1,2)).ravel()
   f=least_squares(ff,p0,loss='soft_l1',f_scale=2,max_nfev=80);pred=project(np.tile(pp(f.x)[3],(len(cc),1)),cc,rr,tt);err.extend(np.linalg.norm(pred-xx[:,3],axis=1))
  summary[tag+'_withheld_fourth_corner_px']=stat(err)
 cp=fit.x[:N*6].reshape(N,6);summary['camera_updates']={n:dict(rotation_deg=float(np.linalg.norm(cp[i,:3])*180/np.pi),translation_rig_units=float(np.linalg.norm(cp[i,3:])),detected_markers=sum(c==i for c,m in obs)) for i,n in enumerate(names)}
 (OUT/'refinement_summary.json').write_text(json.dumps(summary,indent=2));np.savez(OUT/'candidate_cameras_DO_NOT_APPLY.npz',names=names,R=R,t=t,K=K,D=D,parameters=fit.x,marker_ids=mids)
 print(json.dumps(summary['metrics'],indent=2),flush=True)
if __name__=='__main__':main()
