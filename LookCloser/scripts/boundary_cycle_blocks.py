"""Small cyclic blocks inside branched mesh boundaries; opt-in proposal only."""
import numpy as np
import networkx as nx


def cyclic_boundary_blocks(triangles):
    """Return simple biconnected cycles, excluding bridges and complex blocks.

    Unlike connected-component degree tests, an articulation vertex does not
    disqualify a small closed contour. No geometry is changed by this function.
    """
    triangles=np.asarray(triangles)
    directed=triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    edges,counts=np.unique(np.sort(directed,axis=1),axis=0,return_counts=True)
    graph=nx.Graph();graph.add_edges_from(edges[counts==1].tolist())
    orientation=set(map(tuple,directed));loops=[];complex_blocks=0
    for block in nx.biconnected_components(graph):
        if len(block)<3: continue
        local=graph.subgraph(block)
        if any(degree!=2 for _,degree in local.degree()):
            complex_blocks+=1;continue
        start=min(block);loop=[start];previous=None;current=start
        while True:
            nxt=min(n for n in local[current] if n!=previous)
            if nxt==start:break
            loop.append(nxt);previous,current=current,nxt
            if len(loop)>len(block):raise ValueError('Non-simple cycle')
        if len(loop)!=len(block):raise ValueError('Incomplete block')
        if (loop[0],loop[1]) not in orientation:loop=loop[::-1]
        loops.append(np.asarray(loop,np.int32))
    loops.sort(key=lambda x:(int(x.min()),len(x)))
    return loops,dict(boundary_edges=int((counts==1).sum()),complex_blocks=complex_blocks,
        bridges=nx.number_of_edges(graph)-sum(len(c) for c in loops) if not complex_blocks else None)
