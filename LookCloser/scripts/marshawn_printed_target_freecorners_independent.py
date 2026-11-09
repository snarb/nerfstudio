#!/usr/bin/env python3
"""Sensitivity alternative: no square/planar print assumption, tighter pose priors."""
from marshawn_printed_target_refine_independent import *
def main():
 obs=read(408);mids=sorted(m for m in set(m for c,m in obs) if m%5!=0 and sum(mm==m for c,mm in obs)>=4);mi={m:i for i,m in enumerate(mids)};ci=[];pi=[];xy=[];points=[]
 for m in mids:
  rows=[(c,x) for (c,mm),x in obs.items() if mm==m]
  for k in range(4):
   ids=np.array([c for c,x in rows]);yy=np.array([x[k] for c,x in rows]);X=triang([(c,x[k]) for c,x in rows],R0,t0);X=least_squares(lambda v:(project(np.tile(v,(len(ids),1)),ids,R0,t0)-yy).ravel(),X,loss='soft_l1',f_scale=1,max_nfev=50).x;points.append(X)
   for c,x in rows:ci.append(c);pi.append(len(points)-1);xy.append(x[k])
 ci=np.array(ci);pi=np.array(pi);xy=np.array(xy);x0=np.r_[np.zeros(N*6),np.array(points).ravel()];support=np.array([len(set(m for (c,m) in obs if c==i and m in mi)) for i in range(N)])
 def unpack(x):
  p=x[:N*6].reshape(N,6);return Rotation.from_rotvec(p[:,:3]).as_matrix()@R0,t0+p[:,3:],x[N*6:].reshape(-1,3)
 def fun(x):
  r,t,P=unpack(x);prior=x[:N*6].reshape(N,6)/np.array([.0003]*3+[.003]*3);prior[support<5]*=1e5;prior[ni['HC']]*=100
  return np.r_[(project(P[pi],ci,r,t)-xy).ravel(),prior.ravel()]
 S=lil_matrix((len(ci)*2+N*6,len(x0)),dtype=int)
 for i,(c,p) in enumerate(zip(ci,pi)):S[2*i:2*i+2,c*6:c*6+6]=1;S[2*i:2*i+2,N*6+p*3:N*6+p*3+3]=1
 S[len(ci)*2:,:N*6]=np.eye(N*6,dtype=int)
 def mixed_loss(z):
  out=np.empty((3,len(z)));a=1+z;out[0]=2*(np.sqrt(a)-1);out[1]=1/np.sqrt(a);out[2]=-.5/a**1.5;out[0,len(ci)*2:]=z[len(ci)*2:];out[1,len(ci)*2:]=1;out[2,len(ci)*2:]=0;return out
 bounds=np.full_like(x0,np.inf);bounds[:N*6]=np.tile([.0009]*3+[.009]*3,N);bb=bounds[:N*6].reshape(N,6);bb[support<5]=1e-12;bb[ni['HC']]=1e-9
 fit=least_squares(fun,x0,jac_sparsity=S.tocsr(),loss=mixed_loss,bounds=(-bounds,bounds),f_scale=1,x_scale='jac',max_nfev=1000,ftol=1e-5);r,t,P=unpack(fit.x)
 summary=dict(success=bool(fit.success),message=fit.message,nfev=fit.nfev,training_marker_ids=mids,prior='rotation .0003 rad, translation .003 original rig units; HC anchor x100; <5 train markers frozen; hard 3sigma pose-component bounds; HC fixed; Gaussian priors remain quadratic under robust image loss; fixed intrinsics/distortion; free 3D corners',metrics={})
 for f in [408,448,488]:
  o=read(f);summary['metrics'][f]={}
  for tag,rr,tt in [('baseline',R0,t0),('refined',r,t)]:summary['metrics'][f][tag]=dict(epipolar=epipolar(o,rr,tt),crossview=cross_reproject(o,rr,tt))
 for tag,rr,tt in [('baseline',R0,t0),('refined',r,t)]:
  rec=cross_reproject(obs,rr,tt,details=True);(OUT/f'freecorners_{tag}_heldout_predictions.json').write_text(json.dumps(rec,indent=2));summary[tag+'_per_camera']={n:stat([z['error'] for z in rec if z['camera']==n]) for n in names}
 cp=fit.x[:N*6].reshape(N,6);summary['camera_updates']={n:dict(rotation_deg=float(np.linalg.norm(cp[i,:3])*180/np.pi),translation_rig_units=float(np.linalg.norm(cp[i,3:])),train_markers=int(support[i])) for i,n in enumerate(names)}
 (OUT/'freecorners_summary.json').write_text(json.dumps(summary,indent=2));np.savez(OUT/'freecorners_candidate_DO_NOT_APPLY.npz',names=names,R=r,t=t,K=K,D=D,points=P,marker_ids=mids);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
