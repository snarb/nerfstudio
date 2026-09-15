"""Replay refinement and independent native depth decisions on removed subfaces."""
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from study_subface_free_space import ROOT,FRAME,PARENT,DEPTH_ROOT
from subdivide_conflicted_surface import subdivide,verify_coverage
from study_confidence_depth_prior import load_real,support,unproject


def main():
    root=ROOT/FRAME; q=read(root/'request.json'); r=read(root/'result.json')
    assert r['request_sha256']==sha(root/'request.json')
    bindings=dict(q['scripts'])
    bindings[q['mesh']]=q['mesh_sha256']
    bindings[str(PARENT/FRAME/'request.json')]=q['parent_request_sha256']
    bindings[str(PARENT/FRAME/'evidence.npz')]=q['parent_evidence_sha256']
    bindings.update({str(root/n):h for n,h in r['hashes'].items()})
    for path,h in bindings.items():assert sha(path)==h,path
    old=o3d.io.read_triangle_mesh(q['mesh']);ov,ot=np.asarray(old.vertices),np.asarray(old.triangles)
    previous=np.load(PARENT/FRAME/'evidence.npz'); old_s=previous['sample_indices']
    selected=(previous['near_counts'][old_s]>0).any(1)&(previous['stable_far_counts'][old_s]>=6).any(1)
    a=np.load(root/'subdivision.npz'); e=np.load(root/'evidence.npz')
    np.testing.assert_array_equal(selected,a['selected_parents'])
    v,t,p=ov,ot,np.arange(len(ot))
    for _ in range(2):v,t,p=subdivide(v,t,selected[p],p)
    for x,y in [(v,a['vertices']),(t,a['triangles']),(p,a['parents'])]:np.testing.assert_array_equal(x,y)
    coverage=verify_coverage(ov,ot,v,t,p)
    removed=e['removed_triangle_ids'];keep=np.ones(len(t),bool);keep[removed]=False
    for control,expected in [('refined',t),('pruned',t[keep])]:
        dest=ROOT/control/FRAME; cr=read(dest/'result.json')
        assert cr['request_sha256']==sha(dest/'request.json')
        assert sha(dest/'mesh.ply')==cr['hashes']['mesh.ply']
        actual=o3d.io.read_triangle_mesh(str(dest/'mesh.ply'))
        np.testing.assert_array_equal(np.asarray(actual.vertices),v)
        np.testing.assert_array_equal(np.asarray(actual.triangles),expected)
        for name in ['request.json','result.json','mesh.ply']:bindings[str(dest/name)]=sha(dest/name)
    points=np.concatenate([v[t[removed]],v[t[removed]].mean(1)[:,None]],axis=1).reshape(-1,3)
    np.testing.assert_array_equal(points,e['points'][e['sample_indices'][removed]].reshape(-1,3))
    rows,depths,receipt=load_real(DEPTH_ROOT,FRAME);assert receipt==q['depth_receipt']
    near=np.zeros(len(points),int);far=np.zeros_like(near);trusted=np.zeros_like(near)
    for camera,depth in zip(rows,depths):
        pose=np.asarray(camera['transform_matrix']);pc=(points-pose[:3,3])@pose[:3,:3];z=-pc[:,2]
        uv=np.column_stack([camera['fl_x']*pc[:,0]/z+camera['cx'],
                            -camera['fl_y']*pc[:,1]/z+camera['cy']]).astype(np.float32)
        xy=np.rint(uv).astype(int);x,y=xy[:,0],xy[:,1]
        valid=np.isfinite(uv).all(1)&np.isfinite(z)&(z>0)
        inside=valid&(x>=2)&(x<1918)&(y>=2)&(y<1078)
        taps=[]
        for dy in range(-2,3):
            for dx in range(-2,3):
                ok=valid&(x+dx>=0)&(x+dx<1920)&(y+dy>=0)&(y+dy<1080)
                value=np.zeros(len(points));value[ok]=depth[y[ok]+dy,x[ok]+dx];taps.append(value)
        taps=np.asarray(taps);positive=np.isfinite(taps)&(taps>0)
        near+=(positive&(abs(taps-z)<=.0015))[12]
        ordered=np.sort(np.where(positive,taps,np.inf),axis=0);gap=np.maximum(.005,.01*z)
        with np.errstate(invalid='ignore'):
            stable=np.isfinite(ordered[19])&(ordered[4]>0)&((ordered[19]-ordered[4])<=.005*ordered[4])
        far+=inside&stable&((positive&(taps>z+gap)).sum(0)>=20)
        obs=taps[12]
        take=np.flatnonzero(valid&np.isfinite(obs)&(obs>0)&(obs>z+gap))
        counts,_=support(unproject(camera,x[take],y[take],obs[take]),camera,rows,depths)
        trusted[take]+=counts>=3
    assert (near==0).all() and (far>=6).all() and (trusted>=6).all()
    np.testing.assert_array_equal(near,e['near_counts'][e['sample_indices'][removed]].ravel())
    np.testing.assert_array_equal(far,e['stable_far_counts'][e['sample_indices'][removed]].ravel())
    index=np.searchsorted(e['candidates'],removed)
    np.testing.assert_array_equal(e['candidates'][index],removed)
    np.testing.assert_array_equal(trusted,e['trusted_far_by_camera'].sum(0)[index].ravel())
    # Combinatorial conforming check: new boundary edges must subdivide an old
    # boundary, not an internal edge. Counts include pre-existing nonmanifolds.
    edges=np.sort(t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    oldedges=np.sort(ot[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    oe,oc=np.unique(oldedges,axis=0,return_counts=True)
    boundary=ov[oe[oc==1]]; checks=0
    for edge in unique[counts==1]:
        ends=v[edge];middle=ends.mean(0);direction=boundary[:,1]-boundary[:,0]
        alpha=np.einsum('ij,ij->i',middle-boundary[:,0],direction)/np.einsum('ij,ij->i',direction,direction)
        residual=np.linalg.norm(middle-boundary[:,0]-alpha[:,None]*direction,axis=1)
        assert ((alpha>=-1e-10)&(alpha<=1+1e-10)&(residual<1e-12)).any(),'New refinement boundary'
        checks+=1
    atomic_json(root/'independent_audit.json',dict(status='passed',result_sha256=sha(root/'result.json'),
        bindings=bindings,coverage=coverage,refinement_replay_exact=True,refinement_helper_reused=True,
        native_footprints_independent=True,far_corroboration_helper_reused=True,
        removed_subfaces=len(removed),removed_samples=len(points),camera_sample_checks=len(rows)*len(points),
        near_max=int(near.max()),stable_far_min=int(far.min()),trusted_far_min=int(trusted.min()),
        refined_boundary_edges_checked=checks,script_sha256=sha(__file__),quality_approval=False))
    print('independent subface audit passed',len(removed),'removed faces',flush=True)


if __name__=='__main__':main()
