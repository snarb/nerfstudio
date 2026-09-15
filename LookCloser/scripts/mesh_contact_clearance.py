"""Positive separating-plane clearance, including coplanar in-plane axes.

Clearance is a numerical geometric margin in normalized scene units. No
camera, mask, target ray or residual coordinate is read by this helper.
"""
import numpy as np
from scipy import sparse

CLEARANCE=1e-8


def contact_constraints(vertices,triangles,pairs,ids):
    v=np.asarray(vertices,float);t=np.asarray(triangles,int);ids=np.asarray(ids,int)
    if not np.isfinite(v).all():raise ValueError('Nonfinite contact geometry')
    lookup=np.full(len(v),-1,int);lookup[ids]=np.arange(len(ids))
    ri=[];ci=[];data=[];lower=[];records=[]
    for pa,pb in sorted(pairs):
        ta,tb=t[pa],t[pb];a,b=v[ta],v[tb]
        ea=np.roll(a,-1,axis=0)-a;eb=np.roll(b,-1,axis=0)-b
        na=np.cross(ea[0],ea[1]);nb=np.cross(eb[0],eb[1])
        if min(np.linalg.norm(na),np.linalg.norm(nb))<=1e-20:raise ValueError('Degenerate contact triangle')
        na/=np.linalg.norm(na);nb/=np.linalg.norm(nb)
        axes=np.concatenate(([na,nb],np.cross(ea[:,None,:],eb[None,:,:]).reshape(-1,3),
            np.cross(na,ea),np.cross(nb,eb),np.cross(na,eb),np.cross(nb,ea)))
        length=np.linalg.norm(axes,axis=1);axes=axes[length>1e-15]/length[length>1e-15,None]
        axes=np.concatenate((axes,-axes))
        gap=(b@axes.T).min(0)-(a@axes.T).max(0)
        usable=np.ones(len(axes),bool)
        fixed_a=a[lookup[ta]<0];fixed_b=b[lookup[tb]<0]
        if len(fixed_a) and len(fixed_b):
            usable=(fixed_b@axes.T).min(0)-(fixed_a@axes.T).max(0)>=CLEARANCE
        if not usable.any():raise ValueError('Fixed vertices cannot satisfy positive contact clearance')
        pick=int(np.argmax(np.where(usable,gap,-np.inf)));axis=axes[pick]
        if gap[pick]<-1e-10:raise ValueError(f'Already penetrating contact pair {pa},{pb}: {gap[pick]}')
        start=len(lower)
        for va in ta:
            for vb in tb:
                coeff={}
                for vertex,sign in [(va,-1),(vb,1)]:
                    if lookup[vertex]<0:continue
                    for j in range(3):
                        col=3*lookup[vertex]+j;coeff[col]=coeff.get(col,0.)+sign*axis[j]
                norm=np.sqrt(sum(c*c for c in coeff.values()))
                if norm<1e-15:continue  # Fixed/fixed feasibility was checked above.
                row=len(lower);lower.append((CLEARANCE-float(axis@(v[vb]-v[va])))/norm)
                for col,c in coeff.items():ri.append(row);ci.append(col);data.append(c/norm)
        records.append(dict(pair=[int(pa),int(pb)],axis=axis.tolist(),separation=float(gap[pick]),
            clearance=CLEARANCE,first_constraint=start,constraint_count=len(lower)-start,
            coplanar_in_plane_axes_included=True))
    return sparse.coo_matrix((data,(ri,ci)),shape=(len(lower),len(ids)*3)).tocsr(),np.asarray(lower),records
