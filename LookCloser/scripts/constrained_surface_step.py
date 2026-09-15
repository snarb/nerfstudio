"""Sparse least squares with linear lower bounds, using a bounded active set.

The exact nonlinear geometry check remains the caller's responsibility.
No camera/image-specific exceptions are part of this solver.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import linprog


def lower_bound_lsq(system,rhs,constraint,lower,max_iterations=256,tolerance=1e-11):
    m=sparse.csc_matrix(system);a=sparse.csr_matrix(constraint);rhs=np.asarray(rhs,float);b=np.asarray(lower,float)
    if m.shape[0]!=len(rhs) or a.shape[1]!=m.shape[1] or a.shape[0]!=len(b):raise ValueError('Shape mismatch')
    if not all(np.isfinite(v).all() for v in [m.data,a.data,rhs,b]):raise ValueError('Nonfinite input')
    factor=splu((m.T@m).tocsc());x0=factor.solve(np.asarray(m.T@rhs).ravel())
    active=[];cache={};history=[];x=np.zeros_like(x0)
    if (b>0).any():
        feasible=linprog(np.zeros_like(x),A_ub=-a,b_ub=-b,bounds=(None,None),method='highs')
        if not feasible.success:raise ValueError('No feasible initial point')
        x=feasible.x
    for iteration in range(max_iterations):
        p0=x0-x
        if active:
            for i in active:
                if i not in cache:cache[i]=factor.solve(a.getrow(i).toarray().ravel())
            ha=np.column_stack([cache[i] for i in active]);aa=a[active]
            schur=np.asarray(aa@ha);target=-aa@p0
            lam=np.linalg.lstsq(schur,target,rcond=1e-12)[0]
            if np.max(np.abs(schur@lam-target),initial=0)>tolerance*10:
                raise ValueError('Degenerate or inconsistent active constraints')
            direction=p0+ha@lam
        else:lam=np.zeros(0);direction=p0
        slack=np.asarray(a@x).ravel()-b
        if slack.min(initial=0)<-tolerance*10:raise ValueError('Lost primal feasibility')
        if np.linalg.norm(direction,np.inf)<=tolerance:
            if len(lam) and lam.min() < -tolerance:
                removed=active.pop(int(np.argmin(lam)));history.append(dict(remove=removed));continue
            residual=m@x-rhs
            return x,dict(iterations=iteration+1,active=active,multipliers=lam.tolist(),
                minimum_slack=float(slack.min(initial=0)),squared_residual=float(residual@residual),history=history)
        change=np.asarray(a@direction).ravel();eligible=change < -tolerance
        eligible[active]=False;ratios=np.full(len(b),np.inf)
        ratios[eligible]=np.maximum(slack[eligible],0)/-change[eligible]
        blocking=int(np.argmin(ratios)) if len(ratios) else None
        alpha=min(1.,ratios[blocking]) if blocking is not None else 1.
        x=x+alpha*direction
        if alpha<1.:
            active.append(blocking);history.append(dict(add=blocking,step_factor=float(alpha)))
    raise ValueError('Active-set iteration cap; no approximate result accepted')


def oriented_area_constraints(vertices,triangles,reference_cross,reference_good,ids,floor_area):
    """Linearization of n_ref dot cross(e1,e2), rows normalized to unit norm."""
    v=np.asarray(vertices);t=np.asarray(triangles);ids=np.asarray(ids)
    lookup=np.full(len(v),-1,int);lookup[ids]=np.arange(len(ids))
    selected=np.flatnonzero(reference_good & (lookup[t]>=0).any(1));faces=t[selected]
    n=reference_cross[selected];n=n/np.linalg.norm(n,axis=1,keepdims=True)
    e1=v[faces[:,1]]-v[faces[:,0]];e2=v[faces[:,2]]-v[faces[:,0]]
    signed=np.sum(np.cross(e1,e2)*n,axis=1)
    gb=np.cross(e2,n);gc=np.cross(n,e1);grad=np.stack((-gb-gc,gb,gc),axis=1)
    columns=lookup[faces];valid=columns>=0
    ri=np.broadcast_to(np.arange(len(faces))[:,None,None],grad.shape)
    ci=columns[:,:,None]*3+np.arange(3)[None,None,:]
    use=np.broadcast_to(valid[:,:,None],grad.shape)
    a=sparse.coo_matrix((grad[use],(ri[use],ci[use])),shape=(len(faces),len(ids)*3)).tocsr()
    norm=np.sqrt(np.asarray(a.multiply(a).sum(1)).ravel())
    if (norm<=0).any():raise ValueError('Degenerate oriented-area constraint')
    a=sparse.diags(1/norm)@a;b=(np.asarray(floor_area)[selected]-signed)/norm
    return a.tocsr(),b,selected
