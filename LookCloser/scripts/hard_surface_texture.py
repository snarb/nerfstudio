"""Mesh-adjacency source selection; RGB is never averaged across cameras.

Each face gets one source label. A metric alpha-expansion objective discourages
seams on adjacent faces, including across UV chart boundaries. Per-texel fallback
is permitted only when the selected camera fails a geometry visibility check.
"""
from __future__ import annotations
import numpy as np


def face_adjacency(triangles):
    triangles=np.asarray(triangles)
    if triangles.ndim!=2 or triangles.shape[1]!=3:raise ValueError('Expected triangular faces')
    if not len(triangles):return np.empty((0,2),np.int32)
    edges=np.sort(triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    faces=np.repeat(np.arange(len(triangles)),3)
    order=np.lexsort(edges.T[::-1]);edges=edges[order];faces=faces[order]
    starts=np.r_[0,np.flatnonzero(np.any(edges[1:]!=edges[:-1],axis=1))+1,len(edges)]
    good=np.diff(starts)==2;indices=starts[:-1][good]
    return np.column_stack((faces[indices],faces[indices+1])).astype(np.int32)


def select_surface_sources(rgb,quality,triangles,*,smoothness=.08,iterations=2,color_weight=0.):
    rgb=np.asarray(rgb,np.float32);quality=np.asarray(quality,np.float32)
    if quality.ndim!=2 or not all(quality.shape) or rgb.shape!=(*quality.shape,3) or quality.shape[1]!=len(triangles):
        raise ValueError('Expected camera-by-face RGB/quality')
    if not np.isfinite(rgb).all() or not np.isfinite(quality).all() or (quality<0).any():
        raise ValueError('Invalid observations')
    if not np.isfinite([smoothness,color_weight]).all() or min(smoothness,color_weight)<0 or iterations<1:raise ValueError('Invalid graph options')
    import maxflow
    count,n=quality.shape;visible=quality>0;active=visible.any(0)
    best=quality.max(0);unary=-.025*np.log(np.maximum(quality/np.maximum(best,1e-12),1e-12))
    if color_weight:
        # A train-only low-frequency consensus rejects inconsistent shading or
        # highlights during LABEL selection. Never write this statistic to RGB.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            consensus=np.nanmedian(np.where(visible[...,None],rgb,np.nan),axis=0)
        consensus=np.nan_to_num(consensus)
        unary+=color_weight*np.abs(rgb-consensus[None]).mean(-1)
    unary[~visible]=1e5;unary[:,~active]=0
    labels=quality.argmax(0).astype(np.int32);initial=labels.copy()
    edges=face_adjacency(triangles);edges=edges[active[edges].all(1)]
    left,right=edges.T if len(edges) else (np.empty(0,int),np.empty(0,int))
    def pair(la,lb):
        # Sum of two L1 distances and Potts is a metric over camera labels.
        difference=(np.abs(rgb[la,left]-rgb[lb,left]).mean(-1)+
                    np.abs(rgb[la,right]-rgb[lb,right]).mean(-1))*.5
        return smoothness*((la!=lb)+difference*5)
    def energy(state):
        return float(unary[state,np.arange(n)].sum(dtype=np.float64)+pair(state[left],state[right]).sum(dtype=np.float64))
    history=[energy(labels)]
    for sweep in range(iterations):
        changed=0
        for alpha in range(count):
            if not visible[alpha].any():continue
            d0=unary[labels,np.arange(n)].astype(np.float64);d1=unary[alpha].astype(np.float64)
            e00=pair(labels[left],labels[right]);e01=pair(labels[left],alpha);e10=pair(alpha,labels[right])
            weight=np.maximum((e01+e10-e00)*.5,0)
            np.add.at(d1,left,e10-e00-weight);np.add.at(d1,right,e01-e00-weight)
            offset=np.minimum(d0,d1);d0-=offset;d1-=offset
            graph=maxflow.Graph[float](n,len(edges));nodes=graph.add_grid_nodes(n)
            graph.add_edges(left,right,weight,weight);graph.add_grid_tedges(nodes,d1,d0)
            graph.maxflow();proposal=np.where(graph.get_grid_segments(nodes)&visible[alpha],alpha,labels)
            value=energy(proposal)
            if value<history[-1]-1e-7:
                changed+=int(np.count_nonzero(labels!=proposal));labels=proposal;history.append(value)
        print(f'surface_labels sweep={sweep+1} changed={changed} energy={history[-1]:.4f}',flush=True)
        if not changed:break
    if not visible[labels,np.arange(n)][active].all():raise ValueError('Invisible source selected')
    labels=np.where(active,labels,-1).astype(np.int16)
    return labels,{'energy':history,'adjacency_edges':len(edges),'changed_faces':int(np.count_nonzero(labels[active]!=initial[active])),
                   'unsupported_faces':int((~active).sum()),'smoothness':smoothness,'iterations':iterations,'color_weight':color_weight,
                   'source_counts':{str(i):int((labels==i).sum()) for i in range(count)},'averages_rgb':False}


def gather_hard_rgb(colors,weights,preferred):
    """C×3×N -> 3×N, with a single valid source (or black) for every sample."""
    import torch
    if weights.ndim!=2 or colors.shape!=(weights.shape[0],3,weights.shape[1]) or preferred.shape!=(weights.shape[1],):
        raise ValueError('Invalid hard source tensor shapes')
    n=weights.shape[1];index=torch.arange(n,device=weights.device)
    preferred=preferred.to(device=weights.device,dtype=torch.long)
    safe_preferred=preferred.clamp(0,weights.shape[0]-1)
    good=(preferred>=0)&(preferred<weights.shape[0])&(weights[safe_preferred,index]>0)
    chosen=torch.where(good,preferred,weights.argmax(0))
    supported=weights.max(0).values>0
    result=colors[chosen,:,index].T
    result=torch.where(supported[None],result,0)
    return result,torch.where(supported,chosen,-1),supported&~good
