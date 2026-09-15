"""Optional exact displacement-ball solver; no main-environment dependency.

Solve min ||M x-r||² with A x>=b and ||offset_i+x_i||<=radius.
The cone form and independent primal/dual/complementarity checks use x=.001*y.
"""
from pathlib import Path
import sys
import numpy as np
from scipy import sparse

PRIVATE=Path('/home/brans/lookcloser_temp/clarabel_correction_20260915')
sys.path.insert(0,str(PRIVATE))
import clarabel
assert clarabel.__version__=='0.11.1'
SETTINGS=dict(verbose=False,max_iter=150,tol_gap_abs=1e-10,tol_gap_rel=1e-10,
              tol_feas=1e-10,max_threads=2)
TOLERANCES=dict(primal=1e-8,dual=1e-8,stationarity=1e-7,complementarity=1e-7)


def certificate(h,q,a,b,y,z,linear_count):
    """Check actual affine slack, including each self-dual Lorentz cone."""
    y=np.asarray(y,float);z=np.asarray(z,float);s=np.asarray(b-a@y).ravel()
    if not all(np.isfinite(v).all() for v in [y,z,s]):raise ValueError('Nonfinite cone certificate')
    k=linear_count;ss=s[k:].reshape(-1,4);zz=z[k:].reshape(-1,4)
    primal=min(s[:k].min(initial=0), (ss[:,0]-np.linalg.norm(ss[:,1:],axis=1)).min(initial=0))
    dual=min(z[:k].min(initial=0), (zz[:,0]-np.linalg.norm(zz[:,1:],axis=1)).min(initial=0))
    stationarity=float(np.abs(h@y+q+a.T@z).max(initial=0))
    comp=float(max(np.abs(s[:k]*z[:k]).max(initial=0),np.abs(np.sum(ss*zz,axis=1)).max(initial=0)))
    detail=dict(primal_violation=float(max(0,-primal)),dual_violation=float(max(0,-dual)),
                stationarity=stationarity,complementarity=comp)
    for key,value in [('primal',-primal),('dual',-dual),('stationarity',stationarity),('complementarity',comp)]:
        if value>TOLERANCES[key]:raise ValueError(f'Cone {key} check failed: {detail}')
    return detail


def solve(system,rhs,constraint,lower,offset,radius):
    m=sparse.csc_matrix(system);a=sparse.csc_matrix(constraint)
    rhs=np.asarray(rhs,float);b=np.asarray(lower,float);offset=np.asarray(offset,float)
    n=m.shape[1];count=n//3;unit=.001
    if n%3 or offset.shape!=(count,3) or a.shape!=(len(b),n) or len(rhs)!=m.shape[0]:
        raise ValueError('Shape mismatch')
    if not radius>0 or not all(np.isfinite(v).all() for v in [m.data,a.data,rhs,b,offset,np.array(radius)]):
        raise ValueError('Invalid cone input')
    h=(m.T@m).tocsc();g=np.asarray(m.T@rhs).ravel();scale=float(h.diagonal().max())
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid Hessian scale')
    h=h/scale;q=-g/(scale*unit)
    rows=(4*np.arange(count)[:,None]+np.arange(1,4)).ravel()
    soc=sparse.coo_matrix((-np.ones(n),(rows,np.arange(n))),shape=(4*count,n)).tocsc()
    aa=sparse.vstack([-a,soc],format='csc')
    bb=np.r_[-b/unit,np.c_[np.full(count,radius/unit),offset/unit].ravel()]
    cones=([clarabel.NonnegativeConeT(len(b))] if len(b) else [])+[clarabel.SecondOrderConeT(4) for _ in range(count)]
    settings=clarabel.DefaultSettings()
    for key,value in SETTINGS.items():setattr(settings,key,value)
    result=clarabel.DefaultSolver(sparse.triu(h,format='csc'),q,aa,bb,cones,settings).solve()
    if str(result.status)!='Solved':raise ValueError(f'Clarabel did not solve: {result.status}')
    y=np.asarray(result.x);z=np.asarray(result.z)
    checks=certificate(h,q,aa,bb,y,z,len(b));x=unit*y
    slack=np.asarray(a@x).ravel()-b;displacement=np.linalg.norm(offset+x.reshape(-1,3),axis=1)
    if slack.min(initial=0)<-1e-11 or displacement.max(initial=0)>radius+1e-11:
        raise ValueError('Original-unit feasibility check failed')
    active=np.flatnonzero(z[:len(b)]>1e-10);residual=m@x-rhs
    return x,dict(iterations=int(result.iterations),status=str(result.status),backend='clarabel_0.11.1_exact_balls',
        active=active.tolist(),multipliers=(z[active]*scale*unit).tolist(),history=[],
        squared_residual=float(residual@residual),minimum_slack=float(slack.min(initial=0)),
        maximum_displacement=float(displacement.max(initial=0)),certificate=checks,
        variable_unit=unit,objective_hessian_scale=scale,ball_cones=count)


def provenance():
    from study_multiview_face_prior import sha
    files=[p for p in (PRIVATE/'clarabel').rglob('*') if p.is_file() and p.suffix in ['.py','.so']]
    files.append(Path(__file__).resolve())
    return dict(version=clarabel.__version__,private_path=str(PRIVATE),settings=SETTINGS,
        certificate_tolerances=TOLERANCES,helper_hashes={str(p):sha(p) for p in files},
        source='https://clarabel.org/stable/python/getting_started_py/')
