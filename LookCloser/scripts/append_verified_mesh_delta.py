"""Transfer only an append-only geometric delta onto a separately repaired mesh."""
import numpy as np


def append_delta(target_v,target_t,base_v,base_t,proposal_v,proposal_t):
    if not np.array_equal(proposal_v[:len(base_v)],base_v) or not np.array_equal(proposal_t[:len(base_t)],base_t):
        raise ValueError('Proposal is not an exact append-only delta')
    if not all(np.isfinite(v).all() for v in [target_v,base_v,proposal_v]):raise ValueError('Nonfinite vertices')
    added=proposal_t[len(base_t):]
    lookup={}
    for i,p in enumerate(target_v):lookup.setdefault(tuple(p),i)
    new=[];mapping={};reused=0
    for i in np.unique(added):
        key=tuple(proposal_v[i])
        if key in lookup:mapping[int(i)]=lookup[key];reused+=1
        else:
            index=len(target_v)+len(new);lookup[key]=index;mapping[int(i)]=index;new.append(proposal_v[i])
    faces=np.array([[mapping[int(i)] for i in tri] for tri in added],dtype=np.int32).reshape(-1,3)
    if len(faces) and np.any(np.diff(np.sort(faces,axis=1),axis=1)==0):raise ValueError('Degenerate mapped triangle')
    existing=set(map(tuple,np.sort(target_t,axis=1)))
    keep=np.array([tuple(t) not in existing for t in np.sort(faces,axis=1)],bool)
    v=np.concatenate([target_v,np.array(new).reshape(-1,3)]);t=np.concatenate([target_t,faces[keep]])
    return v,t,dict(source_added_triangles=len(added),transferred_triangles=int(keep.sum()),
        already_present_triangles=int((~keep).sum()),new_vertices=len(new),reused_target_vertices=reused,
        original_target_vertices_and_triangles_preserved=True,original_source_faces_not_restored=True)
