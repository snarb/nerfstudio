"""Coordinate descent on a grid with per-node discrete feasible residuals.

Checkerboard updates exactly minimize each conditional quadratic. Disconnected
feasible sets stay disconnected; this does not claim a global optimum.
"""
import numpy as np
from scipy import sparse


def solve(domain,options,valid,pins,pin_residual,regularization=.05,max_iterations=1000):
    y,x=np.nonzero(domain);n=len(x)
    if not n or regularization<=0 or max_iterations<1:raise ValueError('Invalid solve domain or parameters')
    options=np.asarray(options,float);valid=np.asarray(valid,bool);pins=np.asarray(pins,bool);pin_residual=np.asarray(pin_residual,float)
    if options.shape!=valid.shape or options.ndim!=2 or options.shape[0]!=n or pins.shape!=(n,) or pin_residual.shape!=(n,):raise ValueError('Shape mismatch')
    if not valid.any(1).all() or not np.isfinite(options[valid]).all() or not np.isfinite(pin_residual[pins]).all():raise ValueError('Invalid feasible set')
    index=np.full(domain.shape,-1,int);index[y,x]=np.arange(n);aa=[];bb=[]
    for dy,dx in [(0,1),(0,-1),(1,0),(-1,0)]:
        yy,xx=y+dy,x+dx;inside=(yy>=0)&(yy<domain.shape[0])&(xx>=0)&(xx<domain.shape[1])
        a=np.flatnonzero(inside);b=index[yy[a],xx[a]];keep=b>=0;aa.extend(a[keep]);bb.extend(b[keep])
    adjacency=sparse.csr_matrix((np.ones(len(aa)),(aa,bb)),shape=(n,n));degree=np.asarray(adjacency.sum(1)).ravel()
    residual=options[np.arange(n),np.where(valid,np.abs(options),np.inf).argmin(1)];residual[pins]=pin_residual[pins]
    def energy():return float(.5*(np.dot(degree+regularization,residual**2)-residual@(adjacency@residual)))
    energies=[energy()];converged=False
    for iteration in range(max_iterations):
        changed=0
        for parity in [0,1]:
            ids=np.flatnonzero(((x+y)%2==parity)&~pins)
            target=(adjacency@residual)[ids]/(degree[ids]+regularization)
            costs=np.where(valid[ids],(options[ids]-target[:,None])**2,np.inf)
            candidate=options[ids,costs.argmin(1)]
            # Keep the current feasible value on a numerical tie.
            improve=(candidate-target)**2<(residual[ids]-target)**2-1e-20
            changed+=int(improve.sum());residual[ids[improve]]=candidate[improve]
        energies.append(energy())
        if energies[-1]>energies[-2]+1e-12:raise ValueError('Coordinate update increased objective')
        if not changed:converged=True;break
    if not converged:raise ValueError('Discrete surface solve did not converge')
    if not np.array_equal(residual[pins],pin_residual[pins]):raise ValueError('Measured pins moved')
    free=~pins
    if not ((np.abs(options[free]-residual[free,None])<1e-14)&valid[free]).any(1).all():raise ValueError('Left discrete feasible set')
    out=np.zeros(domain.shape);out[y,x]=residual
    return out,dict(nodes=n,measured_pins=int(pins.sum()),iterations=iteration+1,energy_initial=energies[0],energy_final=energies[-1],
        energy_history=energies,coordinatewise_converged=True,global_optimum_claimed=False,pins_exact=True,regularization=regularization)
