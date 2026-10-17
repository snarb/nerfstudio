"""Metric alpha expansion for hard single-photo patches with appearance seams.

Each edge's label metric is the mean L1 difference of both incident patch
feature vectors, plus a small Potts floor. Features are single-source detail
corrections at unchanged calibrated coordinates. They never become blended RGB.
"""
import numpy as np
import maxflow


def grid_edges(shape):
    ids=np.arange(np.prod(shape),dtype=np.int64).reshape(shape)
    return np.concatenate([ids[:,:-1].ravel(),ids[:-1,:].ravel()]),np.concatenate([ids[:,1:].ravel(),ids[1:,:].ravel()])


def metric(features,p,q,first,second,weight,scale=.02,floor=.02):
    left=np.mean(np.abs(features[first,p]-features[second,p]),axis=-1)
    right=np.mean(np.abs(features[first,q]-features[second,q]),axis=-1)
    return weight*(.5*(left+right)/scale+floor*(np.asarray(first)!=np.asarray(second)))


def energy(unary,features,p,q,labels,weight):
    return float(unary[np.arange(len(labels)),labels].sum()+metric(features,p,q,labels[p],labels[q],weight).sum())


def expansion(unary,features,p,q,labels,alpha,weight):
    """Exact binary alpha move for an edge-dependent metric (submodular)."""
    count=len(labels);a=metric(features,p,q,labels[p],labels[q],weight)
    b=metric(features,p,q,labels[p],alpha,weight)
    c=metric(features,p,q,alpha,labels[q],weight)
    edge=(b+c-a)*.5
    assert float(edge.min(initial=0))>=-1e-6
    edge=np.maximum(edge,0)
    keep=unary[np.arange(count),labels].astype(np.float64).copy()
    take=unary[:,alpha].astype(np.float64).copy()
    np.add.at(take,p,c-a-edge);np.add.at(take,q,b-a-edge)
    shift=np.minimum(keep,take);keep-=shift;take-=shift
    graph=maxflow.Graph[float](count,len(p));nodes=graph.add_nodes(count)
    graph.add_edges(p,q,edge,edge)
    graph.add_grid_tedges(nodes,take,keep)
    graph.maxflow();switch=graph.get_grid_segments(nodes)
    result=labels.copy();result[switch]=alpha
    return result


def optimize(costs,features,weight,max_cycles=3):
    """Return discrete labels and a monotonically nonincreasing energy trace."""
    assert costs.ndim==3 and np.isfinite(costs).all() and weight>=0
    shape=costs.shape[:2];k=costs.shape[-1];n=int(np.prod(shape))
    assert features.shape[:2]==(k,n) and np.isfinite(features).all()
    unary=costs.reshape(n,k).astype(np.float64);p,q=grid_edges(shape)
    labels=unary.argmin(axis=-1).astype(np.int16)
    previous=energy(unary,features,p,q,labels,weight);trace=[previous]
    if weight:
        for cycle in range(max_cycles):
            changed=False
            for alpha in range(k):
                proposed=expansion(unary,features,p,q,labels,alpha,weight)
                current=energy(unary,features,p,q,proposed,weight)
                assert current<=previous+1e-5*max(1,abs(previous)),(previous,current)
                if current<previous-1e-8:
                    changed=True;labels=proposed;previous=current;trace.append(current)
            if not changed:break
    return labels.reshape(shape),trace


def patch_vectors(rgb,step):
    """Keep every native pixel in a patch; no cancellation by RGB averaging."""
    h,w,c=rgb.shape;hh=(h+step-1)//step;ww=(w+step-1)//step
    padded=np.pad(rgb,((0,hh*step-h),(0,ww*step-w),(0,0)),mode='edge')
    return padded.reshape(hh,step,ww,step,c).transpose(0,2,1,3,4).reshape(hh*ww,step*step*c)
