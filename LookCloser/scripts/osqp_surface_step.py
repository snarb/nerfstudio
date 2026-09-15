"""Pinned private OSQP backend for the same sparse lower-bound least squares."""
from pathlib import Path
import sys
import numpy as np
from scipy import sparse

PRIVATE=Path('/home/brans/lookcloser_temp/osqp_correction_20260915')
sys.path.insert(0,str(PRIVATE))
import osqp
assert osqp.__version__=='1.0.4'
SETTINGS=dict(eps_abs=1e-10,eps_rel=1e-10,max_iter=100000,polishing=True,
              adaptive_rho_interval=25,check_termination=25,verbose=False)


def lower_bound_lsq(system,rhs,constraint,lower,**unused):
    m=sparse.csc_matrix(system);a=sparse.csc_matrix(constraint)
    rhs=np.asarray(rhs,float);b=np.asarray(lower,float)
    if m.shape[0]!=len(rhs) or a.shape[1]!=m.shape[1] or a.shape[0]!=len(b):raise ValueError('Shape mismatch')
    if not all(np.isfinite(v).all() for v in [m.data,a.data,rhs,b]):raise ValueError('Nonfinite input')
    h=(m.T@m).tocsc();g=np.asarray(m.T@rhs).ravel();scale=float(h.diagonal().max());unit=.001
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid Hessian scale')
    p=sparse.triu(h/scale,format='csc');q=-g/(scale*unit)
    solver=osqp.OSQP();solver.setup(P=p,q=q,A=a,l=b/unit,u=np.full(len(b),np.inf),**SETTINGS)
    result=solver.solve(raise_error=True)
    if result.info.status_val!=1:raise ValueError('OSQP did not solve')
    y=result.x;x=y*unit;dual=result.y
    slack=np.asarray(a@x).ravel()-b
    stationarity=np.asarray((h/scale)@y).ravel()+q+np.asarray(a.T@dual).ravel()
    if not np.isfinite(x).all() or slack.min(initial=0)<-1e-11:raise ValueError('QP primal check failed')
    if np.max(np.abs(stationarity),initial=0)>1e-7:raise ValueError('QP stationarity check failed')
    if dual.max(initial=0)>1e-8:raise ValueError('QP dual sign check failed')
    active=np.flatnonzero(dual < -1e-10);residual=m@x-rhs
    return x,dict(iterations=int(result.info.iter),active=active.tolist(),
        multipliers=(-dual[active]*scale*unit).tolist(),minimum_slack=float(slack.min(initial=0)),
        squared_residual=float(residual@residual),history=[],backend='osqp_1.0.4',
        status=result.info.status,polish_status=int(result.info.status_polish),
        scaled_stationarity_inf=float(np.abs(stationarity).max(initial=0)),
        scaled_complementarity_inf=float(np.abs(dual*(slack/unit)).max(initial=0)),
        variable_unit=unit,objective_hessian_scale=scale)


def provenance():
    from study_multiview_face_prior import sha
    files=[p for p in (PRIVATE/'osqp').rglob('*') if p.is_file() and p.suffix in ['.py','.so']]
    files.append(Path(__file__).resolve())
    return dict(version=osqp.__version__,private_path=str(PRIVATE),settings=SETTINGS,
        checked_primal_tolerance=1e-11,checked_scaled_stationarity_tolerance=1e-7,
        helper_hashes={str(p):sha(p) for p in files},source='https://osqp.org/docs/interfaces/python.html')
