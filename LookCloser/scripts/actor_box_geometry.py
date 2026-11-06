"""Geometric actor anchors and component qualification, never free-space carving."""
from itertools import combinations

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from audit_marshawn_geometry import projection, fundamental, sampson, triangulate


def fit_floor_frame(points):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<3 or not np.isfinite(points).all():
        raise ValueError('At least three finite floor points required')
    center=points.mean(0);_,singular,v=np.linalg.svd(points-center)
    if singular[1]<1e-10:raise ValueError('Floor points must not be collinear')
    normal=v[-1]*np.sign(v[-1,2])
    if normal[2]<=0:raise ValueError('Floor normal must have an upward component')
    offset=-normal@center
    x=np.array([1.,0,0]);x-=normal*(x@normal);x/=np.linalg.norm(x)
    y=np.cross(normal,x)
    T=np.eye(4);T[:3,:3]=np.stack([x,y,normal]);T[2,3]=offset
    return T,points@normal+offset


def triangulate_actor_tracks(rows,images,foreground):
    features=[]
    for image,mask in zip(images,foreground):
        gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
        key,des=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.015).detectAndCompute(gray,mask.astype('uint8')*255)
        features.append((np.array([k.pt for k in key],np.float32)+.5,des))
    matcher=cv2.BFMatcher();parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for i,j in combinations(range(len(rows)),2):
        ua,da=features[i];ub,db=features[j]
        if da is None or db is None or len(db)<2:continue
        pairs=matcher.knnMatch(da,db,k=2)
        good=[m for m,n in pairs if m.distance<.68*n.distance]
        if not good:continue
        a=ua[[m.queryIdx for m in good]];b=ub[[m.trainIdx for m in good]]
        xyz,valid,err,angle=triangulate(rows[i],rows[j],a,b)
        valid &= sampson(fundamental(rows[i],rows[j]),a,b)<1.
        for k in np.flatnonzero(valid):
            x,y=find((i,good[k].queryIdx)),find((j,good[k].trainIdx))
            if x!=y:parent[x]=y
    tracks={}
    for x in parent:tracks.setdefault(find(x),[]).append(x)
    points=[];observations=[]
    for track in tracks.values():
        if len(track)<3 or len(set(i for i,k in track))!=len(track):continue
        equations=[]
        for i,k in track:
            p,*_=projection(rows[i]);u,v=features[i][0][k]
            equations.extend([u*p[2]-p[0],v*p[2]-p[1]])
        _,_,vt=np.linalg.svd(equations,full_matrices=False);hom=vt[-1]
        if abs(hom[3])<1e-12:continue
        xyz=hom[:3]/hom[3];obs=[];good=True
        for i,k in track:
            p,*_=projection(rows[i]);q=p@np.r_[xyz,1];uv=features[i][0][k]
            error=float(np.linalg.norm(q[:2]/q[2]-uv))
            if q[2]<=0 or error>1:good=False;break
            obs.append(dict(camera=rows[i]['physical_camera'],uv=uv.tolist(),reprojection_error=error))
        if good:points.append(xyz);observations.append(obs)
    return np.asarray(points).reshape(-1,3),observations


def anchor_components(lower,upper,anchors,nearby_margin=.03):
    """Retain anchor-bearing components and nearby pieces such as detached hair.

    Remote components are unresolved silhouette aliases, not certified empty.
    Anchors must be qualified independently in synchronized source photographs.
    """
    lower,upper,anchors=map(np.asarray,[lower,upper,anchors])
    if len(anchors)<1:raise ValueError('At least one actor anchor triangulated from three cameras is required')
    center=(lower+upper)/2;step=(upper-lower)[0]
    if not np.allclose(upper-lower,step):raise ValueError('Uniform leaf extents required')
    pairs=cKDTree(center/step).query_pairs(1.001,p=np.inf,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(center),len(center)))
    count,labels=connected_components(graph,directed=False)
    distance,index=cKDTree(center).query(anchors)
    close=distance<=2*np.linalg.norm(step)
    support=np.bincount(labels[index[close]],minlength=count)
    accepted=np.flatnonzero(support>0)
    if not len(accepted):raise ValueError('No hull component agrees with triangulated actor geometry')
    primary=np.isin(labels,accepted)
    proximity=cKDTree(center[primary]).query(center)[0]
    nearby=np.unique(labels[proximity<=nearby_margin])
    keep=np.isin(labels,nearby)
    rows=[]
    for k in range(count):
        use=labels==k
        rows.append(dict(component=k,cells=int(use.sum()),anchors=int(support[k]),
            retained=bool(k in nearby),bounds=[lower[use].min(0).tolist(),upper[use].max(0).tolist()]))
    return keep,dict(components=rows,nearby_margin=nearby_margin,
        anchors_within_two_leaf_diagonals=int(close.sum()),anchors_total=len(anchors),
        discarded_semantics='unresolved remote silhouette components, NOT known free space')
