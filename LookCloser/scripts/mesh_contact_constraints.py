"""Local separating-axis inequalities for disjoint triangle pairs.

Fixed separating axes guarantee separation for the proposed endpoint, not a
continuous rotation certificate. Existing penetrating pairs are not repaired.
"""
import numpy as np
from scipy import sparse


def contact_constraints(vertices,triangles,pairs,ids):
    v=np.asarray(vertices);t=np.asarray(triangles);ids=np.asarray(ids)
    lookup=np.full(len(v),-1,int);lookup[ids]=np.arange(len(ids))
    ri=[];ci=[];data=[];lower=[];records=[]
    for pa,pb in sorted(pairs):
        ta,tb=t[pa],t[pb];a,b=v[ta],v[tb]
        ea=np.roll(a,-1,axis=0)-a;eb=np.roll(b,-1,axis=0)-b
        axes=np.concatenate((np.cross(ea[:1],ea[1:2]),np.cross(eb[:1],eb[1:2]),
                             np.cross(ea[:,None,:],eb[None,:,:]).reshape(-1,3)))
        norm=np.linalg.norm(axes,axis=1);axes=axes[norm>1e-15]/norm[norm>1e-15,None]
        if not len(axes):raise ValueError('No nondegenerate contact axis')
        axes=np.concatenate((axes,-axes));gap=(b@axes.T).min(0)-(a@axes.T).max(0)
        pick=int(np.argmax(gap));axis=axes[pick]
        if gap[pick]<-1e-10:raise ValueError(f'Already penetrating contact pair {pa},{pb}: {gap[pick]}')
        start=len(lower)
        for va in ta:
            for vb in tb:
                coeff={}
                for vertex,sign in [(va,-1),(vb,1)]:
                    if lookup[vertex]<0:continue
                    for j in range(3):
                        col=3*lookup[vertex]+j;coeff[col]=coeff.get(col,0.)+sign*axis[j]
                length=np.sqrt(sum(c*c for c in coeff.values()))
                if length<1e-15:continue
                row=len(lower)
                # Only roundoff-sized overlap is admitted, never a new physical gap tolerance.
                lower.append(-max(float(axis@(v[vb]-v[va])),0.)/length)
                for col,c in coeff.items():ri.append(row);ci.append(col);data.append(c/length)
        records.append(dict(pair=[int(pa),int(pb)],axis=axis.tolist(),separation=float(gap[pick]),
                            first_constraint=start,constraint_count=len(lower)-start))
    matrix=sparse.coo_matrix((data,(ri,ci)),shape=(len(lower),len(ids)*3)).tocsr()
    return matrix,np.asarray(lower),records
