"""Fit a smooth bounded depth residual to observed local surface anchors.

One median per camera/node followed by a median across cameras prevents repeated
samples from one source from dominating. The input surface remains the prior;
zero-displacement boundary pins and all production vertices stay fixed.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


def anchor_targets(uv,depth,anchor_uv,anchor_z,sources,max_distance=.75):
    distance,node=cKDTree(uv).query(anchor_uv)
    valid=np.isfinite(anchor_z)&(anchor_z>0)&(distance<=max_distance)
    groups={}
    for i in np.flatnonzero(valid):
        groups.setdefault((int(node[i]),int(sources[i])),[]).append(float(anchor_z[i]))
    bynode={}
    for (i,camera),values in groups.items():bynode.setdefault(i,[]).append(float(np.median(values)))
    target=np.zeros(len(depth));weight=np.zeros(len(depth));count=np.zeros(len(depth),int)
    for i,values in bynode.items():
        target[i]=np.median(values)-depth[i];weight[i]=1;count[i]=len(values)
    return target,weight,count,dict(nearby_samples=int(valid.sum()),observed_nodes=int(weight.sum()),
        camera_node_pairs=len(groups),max_assignment_distance_px=max_distance)


def solve_residual(node_count,edges,target,weight,pinned,*,smoothness=.2,prior_weight=.001,max_displacement=.006):
    if smoothness<=0 or prior_weight<=0 or max_displacement<=0:raise ValueError('Positive regularization/bound required')
    if any(len(a)!=node_count for a in [target,weight,pinned]):raise ValueError('Node shape mismatch')
    if not np.isfinite(target).all() or not np.isfinite(weight).all() or (weight<0).any():raise ValueError('Invalid anchor data')
    if not (weight[~pinned]>0).any():raise ValueError('No movable observed anchor')
    edges=np.unique(np.sort(np.asarray(edges,int).reshape(-1,2),axis=1),axis=0)
    if len(edges) and ((edges<0).any() or (edges>=node_count).any()):raise ValueError('Invalid graph index')
    i,j=edges.T;degree=np.bincount(edges.ravel(),minlength=node_count)
    matrix=sparse.diags(weight+prior_weight+smoothness*degree).tocsr()
    matrix+=sparse.csr_matrix((-smoothness*np.ones(2*len(edges)),(np.r_[i,j],np.r_[j,i])),shape=(node_count,node_count))
    free=~np.asarray(pinned,bool);result=np.zeros(node_count)
    result[free]=spsolve(matrix[free][:,free],(weight*target)[free])
    if not np.isfinite(result).all():raise ValueError('Nonfinite residual')
    clipped=np.abs(result)>max_displacement;result=np.clip(result,-max_displacement,max_displacement)
    return result,dict(observed_nodes=int((weight>0).sum()),pinned_nodes=int(pinned.sum()),
        smoothness=smoothness,prior_weight=prior_weight,max_displacement=max_displacement,
        clipped_nodes=int(clipped.sum()),residual_quantiles=np.quantile(result,[0,.5,1]).tolist())
