"""Geometric interpolation certificate, not a depth observation at the query."""
import numpy as np
from scipy.spatial import ConvexHull, QhullError


def certify(query, normal, seeds, *, tolerance=.0005):
    seeds=np.asarray(seeds,float);q=np.asarray(query,float);n=np.asarray(normal,float)
    if len(seeds)<8 or not np.isfinite(seeds).all() or not np.isfinite(q).all() or not np.isfinite(n).all() or np.linalg.norm(n)<1e-9:
        return False,dict(reason='insufficient_or_invalid_seeds')
    n=n/np.linalg.norm(n);axis=np.eye(3)[np.argmin(np.abs(n))];u=np.cross(n,axis);u/=np.linalg.norm(u);w=np.cross(n,u)
    local=(seeds-q)@np.column_stack([u,w,n]);radius=max(float(np.linalg.norm(local[:,:2],axis=1).max()),1e-12)
    xy=local[:,:2]/radius
    try:hull=ConvexHull(xy)
    except QhullError:return False,dict(reason='degenerate_hull')
    if (hull.equations[:,-1]>1e-9).any():return False,dict(reason='outside_seed_hull')
    x,y=xy.T;a=np.column_stack([np.ones(len(x)),x,y,x*x,x*y,y*y]);sv=np.linalg.svd(a,compute_uv=False)
    if sv[-1]<sv[0]*1e-4:return False,dict(reason='ill_conditioned_fit')
    pinv=np.linalg.pinv(a);coef=pinv@local[:,2];leverage=np.einsum('ij,ji->i',a,pinv)
    if (leverage>.99).any():return False,dict(reason='unstable_cross_validation')
    loo=np.abs((a@coef-local[:,2])/(1-leverage));p90=float(np.quantile(loo,.9));offset=float(abs(coef[0]))
    return p90<=tolerance and offset<=tolerance,dict(reason='fit_checked',loo_p90=p90,predicted_offset=offset,seeds=len(seeds))
