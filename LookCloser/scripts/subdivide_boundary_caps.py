"""Boundary-preserving cap subdivision with an optional bounded local quadric.

Original mesh vertices/triangles and all proposal perimeter edges remain exact.
The quadric is an inferred continuation of adjacent geometry, not measured depth.
"""
import numpy as np


def design(xy):
    x,y=np.asarray(xy).T
    return np.column_stack([np.ones(len(x)),x,y,x*x,x*y,y*y])


def subdivide(vertices, triangles, proposals, notes, *, curved=False, max_shift=.0005, rings=2):
    v=np.asarray(vertices,float);t=np.asarray(triangles,int);p=np.asarray(proposals,int)
    if not np.isfinite(v).all() or max_shift<0:raise ValueError('Invalid geometry/bound')
    adjacency=[set() for _ in v]
    for a,b,c in t:
        adjacency[a].update([int(b),int(c)]);adjacency[b].update([int(a),int(c)]);adjacency[c].update([int(a),int(b)])
    centers=v[p].mean(1);fitted=centers.copy();records=[];offset=0
    for note in notes:
        count=note['triangles'];arc=np.array(note['vertices'],int)
        support=set(arc.tolist());frontier=set(support)
        for _ in range(rings):
            frontier=set().union(*(adjacency[i] for i in frontier))-support if frontier else set()
            support.update(frontier)
        ids=np.array(sorted(support),int);origin=v[arc].mean(0)
        _,_,basis=np.linalg.svd(v[arc]-origin,full_matrices=False)
        local=(v[ids]-origin)@basis.T
        radius=max(float(np.linalg.norm(local[:,:2],axis=1).max()),1e-8)
        a=design(local[:,:2]/radius);weight=np.where(np.isin(ids,arc),4.,1.)
        coef,_,rank,_=np.linalg.lstsq(a*np.sqrt(weight[:,None]),local[:,2]*np.sqrt(weight),rcond=None)
        rmse=float(np.sqrt(np.average((a@coef-local[:,2])**2,weights=weight)))
        q=(centers[offset:offset+count]-origin)@basis.T
        displacement=design(q[:,:2]/radius)@coef-q[:,2]
        accepted=rank==6 and rmse<=max_shift and np.isfinite(displacement).all() and np.max(np.abs(displacement),initial=0)<=max_shift
        if curved and accepted:fitted[offset:offset+count]+=displacement[:,None]*basis[2]
        records.append(dict(triangles=count,support_vertices=len(ids),rank=int(rank),fit_rmse=rmse,
                            bound_pass=bool(accepted),applied=bool(curved and accepted),
                            proposed_max_shift=float(np.max(np.abs(displacement),initial=0))))
        offset+=count
    if offset!=len(p):raise ValueError('Proposal notes do not cover triangles')
    added=np.arange(len(v),len(v)+len(p))
    split=np.stack([np.column_stack([p[:,0],p[:,1],added]),
                    np.column_stack([p[:,1],p[:,2],added]),
                    np.column_stack([p[:,2],p[:,0],added])],axis=1).reshape(-1,3)
    if np.max(np.linalg.norm(fitted-centers,axis=1),initial=0)>max_shift+1e-12:raise ValueError('Displacement exceeds bound')
    return np.concatenate([v,fitted]),split,records
