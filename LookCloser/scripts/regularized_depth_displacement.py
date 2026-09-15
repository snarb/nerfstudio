"""Bounded smooth 3D displacement from calibrated camera-depth equations."""
import inspect
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg


def solve_displacement(hessian,rhs,edges,*,prior_weight=.05,smoothness=.2,maximum_step=.002):
    h=np.asarray(hessian,float);g=np.asarray(rhs,float);n=len(h)
    if h.shape!=(n,3,3) or g.shape!=(n,3) or not np.isfinite(h).all() or not np.isfinite(g).all():
        raise ValueError('Invalid depth equations')
    if prior_weight<=0 or smoothness<0 or maximum_step<=0:raise ValueError('Invalid regularization/bound')
    e=np.asarray(edges,dtype=int).reshape(-1,2)
    if len(e) and (e.min()<0 or e.max()>=n):raise ValueError('Invalid graph index')
    e=np.unique(np.sort(e,axis=1),axis=0);e=e[e[:,0]!=e[:,1]]
    degree=np.bincount(e.ravel(),minlength=n)
    weights=1/np.sqrt(np.maximum(degree[e[:,0]],1)*np.maximum(degree[e[:,1]],1))
    rr=np.concatenate([e[:,0],e[:,1],e[:,0],e[:,1]])
    cc=np.concatenate([e[:,0],e[:,1],e[:,1],e[:,0]])
    lap=sparse.coo_matrix((np.concatenate([weights,weights,-weights,-weights]),(rr,cc)),shape=(n,n)).tocsr()
    indices=np.arange(n*3).reshape(n,3)
    r=np.broadcast_to(indices[:,:,None],h.shape).ravel();c=np.broadcast_to(indices[:,None,:],h.shape).ravel()
    matrix=sparse.coo_matrix((h.ravel(),(r,c)),shape=(3*n,3*n)).tocsr()
    matrix+=prior_weight*sparse.eye(3*n,format='csr')+smoothness*sparse.kron(lap,sparse.eye(3),format='csr')
    preconditioner=sparse.diags(1/matrix.diagonal())
    kw={'rtol':1e-9,'atol':0} if 'rtol' in inspect.signature(cg).parameters else {'tol':1e-9,'atol':0}
    delta,info=cg(matrix,g.ravel(),M=preconditioner,maxiter=500,**kw)
    if info!=0 or not np.isfinite(delta).all():raise ValueError('Displacement solve did not converge')
    residual=float(np.linalg.norm(matrix@delta-g.ravel()))
    displacement=delta.reshape(n,3);norm=np.linalg.norm(displacement,axis=1)
    bounded=displacement*np.minimum(1,maximum_step/np.maximum(norm,1e-20))[:,None]
    return bounded,dict(linear_residual=residual,unclipped_maximum=float(norm.max(initial=0)),
        clipped_vertices=int((norm>maximum_step).sum()),energy_before=0.,
        unconstrained_energy_after=float(.5*delta@(matrix@delta)-delta@g.ravel()))
