"""Screen-grid quadric residual solve with measured, not arbitrary, boundary pins."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def solve_depth(domain,model,observed,trusted,regularization=.05):
    if regularization<=0:raise ValueError('Positive model regularization required')
    if not (domain.shape==model.shape==observed.shape==trusted.shape):raise ValueError('Grid shape mismatch')
    y,x=np.nonzero(domain);n=len(x)
    if not n:raise ValueError('Empty domain')
    if not np.isfinite(model[domain]).all() or (model[domain]<=0).any():raise ValueError('Invalid model depth')
    pins=trusted[domain]&(observed[domain]>0)&np.isfinite(observed[domain])
    if pins.sum()<3:raise ValueError('Insufficient measured boundary pins')
    index=np.full(domain.shape,-1,int);index[y,x]=np.arange(n)
    rr=[];cc=[];vv=[];degree=np.zeros(n)
    for dy,dx in [(0,1),(0,-1),(1,0),(-1,0)]:
        yy,xx=y+dy,x+dx;inside=(yy>=0)&(yy<domain.shape[0])&(xx>=0)&(xx<domain.shape[1])
        a=np.flatnonzero(inside);b=index[yy[a],xx[a]];a,b=a[b>=0],b[b>=0]
        rr.extend(a);cc.extend(b);vv.extend(-np.ones(len(a)));degree[a]+=1
    rr.extend(range(n));cc.extend(range(n));vv.extend(degree+regularization)
    matrix=sparse.csr_matrix((vv,(rr,cc)),shape=(n,n));residual=np.zeros(n)
    residual[pins]=observed[domain][pins]-model[domain][pins];unknown=~pins
    residual[unknown]=spsolve(matrix[unknown][:,unknown],-matrix[unknown][:,pins]@residual[pins])
    result=np.zeros(domain.shape,np.float64);result[domain]=model[domain]+residual
    result[trusted&domain]=observed[trusted&domain]
    if not np.isfinite(result[domain]).all() or (result[domain]<=0).any():raise ValueError('Invalid solved depth')
    return result,dict(nodes=n,measured_pins=int(pins.sum()),regularization=regularization,
        residual_quantiles=np.quantile(residual,[0,.5,1]).tolist(),pins_exact=True)

def grid_faces(domain,active,index):
    a,b,c,d=index[:-1,:-1],index[:-1,1:],index[1:,:-1],index[1:,1:]
    faces=[]
    for aa,bb,cc,touch in [(a,b,c,active[:-1,:-1]|active[:-1,1:]|active[1:,:-1]),
                            (b,d,c,active[:-1,1:]|active[1:,1:]|active[1:,:-1])]:
        valid=(aa>=0)&(bb>=0)&(cc>=0)&touch
        faces.append(np.column_stack([aa[valid],bb[valid],cc[valid]]))
    return np.concatenate(faces)
