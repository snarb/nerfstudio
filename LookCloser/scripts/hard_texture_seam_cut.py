"""Visibility-constrained hard RGB source labelling by metric alpha expansion.

No colours are averaged or synthesized: optimization only changes the source
index. Pair costs penalize visible colour differences along a source seam.
"""
from __future__ import annotations
import numpy as np


def visibility_rank_confidence(valid,depth,radius):
    """Reduce the primary-rank prior near large occlusions on the same depth layer."""
    from scipy import ndimage
    if radius<=0:return np.ones(valid.shape[1:],np.float32)
    holes=valid.any(0)&~valid[0]
    labels,_=ndimage.label(holes);sizes=np.bincount(labels.ravel())
    large=(labels>0)&(sizes[labels]>=100)
    if not large.any():return np.ones(holes.shape,np.float32)
    distance,nearest=ndimage.distance_transform_edt(~large,return_indices=True)
    near_depth=depth[tuple(nearest)]
    same_layer=(depth>0)&(near_depth>0)&(np.abs(np.log(depth.clip(1e-7)/near_depth.clip(1e-7)))<.0075)
    return np.where(same_layer,np.minimum(distance/radius,1)**2,1).astype(np.float32)


def optimize_source_labels(rgb: np.ndarray, valid: np.ndarray, *, rank_penalty: float = .0001,
                           smoothness: float = 1., iterations: int = 2,
                           rank_confidence: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Input SHWC display RGB and SHW visibility; output HW indices (-1 for misses)."""
    import maxflow
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or valid.shape != rgb.shape[:-1]:
        raise ValueError("Expected SHWC RGB and SHW visibility")
    if not np.isfinite(rgb).all() or rank_penalty < 0 or smoothness < 0 or iterations < 1:
        raise ValueError("Invalid finite RGB or optimization parameters")
    count,height,width,_ = rgb.shape
    if rank_confidence is None:rank_confidence=np.ones((height,width),np.float32)
    if rank_confidence.shape!=(height,width) or not np.isfinite(rank_confidence).all() or (rank_confidence<0).any():
        raise ValueError('Invalid spatial source-rank confidence')
    # Cropping the graph to the union of valid observations affects no surface pixels.
    support = valid.any(0)
    if not support.any():
        return np.full((height,width),-1,np.int32),{"energy":[],"changed_pixels":0}
    yy,xx=np.where(support)
    y0,y1,x0,x1=int(yy.min()),int(yy.max()+1),int(xx.min()),int(xx.max()+1)
    colors=np.ascontiguousarray(rgb[:,y0:y1,x0:x1],dtype=np.float32)
    visible=np.ascontiguousarray(valid[:,y0:y1,x0:x1],dtype=bool)
    confidence=rank_confidence[y0:y1,x0:x1]
    active=visible.any(0)
    h,w=active.shape
    labels=np.argmax(visible,axis=0).astype(np.int32)
    original=labels.copy()
    iy,ix=np.indices((h,w))
    flat=np.arange(h*w).reshape(h,w)
    edges=[(flat[:,:-1].ravel(),flat[:,1:].ravel()),(flat[:-1].ravel(),flat[1:].ravel())]
    flat_colors=colors.reshape(count,-1,3)
    active_flat=active.ravel()
    edges=[(a[active_flat[a]&active_flat[b]],b[active_flat[a]&active_flat[b]]) for a,b in edges]
    def cost(a,b,la,lb):
        # Sum of L1 distances at both edge endpoints is a metric over labels.
        value=.5*(np.abs(flat_colors[la,a]-flat_colors[lb,a]).mean(-1)
                   +np.abs(flat_colors[la,b]-flat_colors[lb,b]).mean(-1))
        return smoothness*(value+.01*(la!=lb))
    def energy(state):
        value=float((state[active]*rank_penalty*confidence[active]).sum())
        sf=state.ravel()
        for a,b in edges:value+=float(cost(a,b,sf[a],sf[b]).sum())
        return value
    history=[energy(labels)]
    for sweep in range(iterations):
        changed=0
        for alpha in range(count):
            if not visible[alpha].any():continue
            d0=np.where(active,labels*rank_penalty*confidence,0.).astype(np.float64).ravel()
            d1=np.where(visible[alpha],alpha*rank_penalty*confidence,1e6).astype(np.float64).ravel()
            state=labels.ravel()
            graph=maxflow.Graph[float](h*w,2*h*w)
            nodes=graph.add_grid_nodes((h,w))
            for a,b in edges:
                e00=cost(a,b,state[a],state[b])
                e01=cost(a,b,state[a],alpha)
                e10=cost(a,b,alpha,state[b])
                weight=np.maximum(.5*(e01+e10-e00),0)
                np.add.at(d1,a,e10-e00-weight)
                np.add.at(d1,b,e01-e00-weight)
                graph.add_edges(a,b,weight,weight)
            offset=np.minimum(d0,d1)
            d0-=offset;d1-=offset
            graph.add_grid_tedges(nodes,d1.reshape(h,w),d0.reshape(h,w))
            graph.maxflow()
            proposal=np.where(graph.get_grid_segments(nodes)&visible[alpha],alpha,labels).astype(np.int32)
            proposal_energy=energy(proposal)
            if proposal_energy < history[-1]-1e-7:
                changed+=int(np.count_nonzero(proposal!=labels))
                labels=proposal;history.append(proposal_energy)
        if not changed:break
    if not visible[labels,iy,ix][active].all():
        raise RuntimeError("Graph cut assigned a geometrically invisible source")
    result=np.full((height,width),-1,np.int32)
    result[y0:y1,x0:x1]=np.where(active,labels,-1)
    return result,{"energy":history,"changed_pixels":int(np.count_nonzero((labels!=original)&active)),
                   "rank_penalty":rank_penalty,"smoothness":smoothness,"iterations":iterations,
                   "uses_eval_rgb":False,"averages_sources":False}
