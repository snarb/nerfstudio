"""Independent native footprint replay for every actually removed triangle."""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from prune_measured_free_surface import ROOT
from review_full_block_transfer import ROOT as DEPTH_ROOT
from study_confidence_depth_prior import load_real,support,unproject


def audit(frame,root_base=ROOT):
    root=root_base/frame;q=read(root/'request.json');r=read(root/'result.json')
    radius=q['parameters']['near_native_radius'];assert radius in (0,2)
    assert sha(root/'request.json')==r['request_sha256']
    for name,digest in r['hashes'].items():assert sha(root/name)==digest
    for p,digest in q['scripts'].items():assert sha(p)==digest
    assert sha(q['mesh'])==q['mesh_sha256']
    old=o3d.io.read_triangle_mesh(q['mesh']);new=o3d.io.read_triangle_mesh(str(root/'mesh.ply'))
    v,t=np.asarray(old.vertices),np.asarray(old.triangles)
    evidence=np.load(root/'evidence.npz');removed=evidence['removed_triangle_ids']
    keep=np.ones(len(t),bool);keep[removed]=False
    assert len(np.unique(removed))==len(removed)==r['removed_triangles']
    np.testing.assert_array_equal(np.asarray(new.vertices),v)
    np.testing.assert_array_equal(np.asarray(new.triangles),t[keep])
    points=np.concatenate([v[t[removed]],v[t[removed]].mean(1)[:,None]],axis=1).reshape(-1,3)
    rows,depths,receipt=load_real(DEPTH_ROOT,frame);assert receipt==q['depth_receipt']
    near=np.zeros(len(points),int);far=np.zeros_like(near);trusted=np.zeros_like(near)
    per_camera=[]
    for camera,d in zip(rows,depths):
        pose=np.asarray(camera['transform_matrix']);pc=(points-pose[:3,3])@pose[:3,:3]
        z=-pc[:,2]
        # The producer stores native UV as float32. Reproduce only this explicit
        # precision convention; footprint and decision code are independent.
        uv=np.column_stack([camera['fl_x']*pc[:,0]/z+camera['cx'],
                            -camera['fl_y']*pc[:,1]/z+camera['cy']]).astype(np.float32)
        xy=np.rint(uv).astype(int);x,y=xy[:,0],xy[:,1]
        values=[]
        valid=np.isfinite(uv).all(1)&np.isfinite(z)&(z>0)
        inside=valid&(x>=2)&(x<1918)&(y>=2)&(y<1078)
        for dy in range(-2,3):
            for dx in range(-2,3):
                ok=valid&(x+dx>=0)&(x+dx<1920)&(y+dy>=0)&(y+dy<1080)
                value=np.zeros(len(points));value[ok]=d[y[ok]+dy,x[ok]+dx];values.append(value)
        taps=np.array(values);positive=np.isfinite(taps)&(taps>0)
        near_taps=(positive&(np.abs(taps-z)<=.0015))
        near_here=near_taps.any(0) if radius==2 else near_taps[12]
        gap=np.maximum(.005,.01*z)
        ordered=np.sort(np.where(positive,taps,np.inf),axis=0)
        with np.errstate(invalid='ignore'):
            stable=np.isfinite(ordered[19])&(ordered[4]>0)&((ordered[19]-ordered[4])<=.005*ordered[4])
        far_here=inside&stable&((positive&(taps>z+gap)).sum(0)>=20)
        near+=near_here;far+=far_here
        ok=valid&(x>=0)&(x<1920)&(y>=0)&(y<1080)
        obs=np.zeros(len(points));obs[ok]=d[y[ok],x[ok]]
        j=np.flatnonzero(ok&np.isfinite(obs)&(obs>0)&(obs>z+gap))
        count,_=support(unproject(camera,x[j],y[j],obs[j]),camera,rows,depths)
        trusted[j]+=count>=3
        per_camera.append(dict(camera=camera['physical_camera'],near=int(near_here.sum()),
            stable_far=int(far_here.sum()),trusted_far=int((count>=3).sum())))
    assert (near==0).all() and (far>=6).all() and (trusted>=6).all()
    np.testing.assert_array_equal(near,evidence['near_counts'][evidence['sample_indices'][removed]].ravel())
    np.testing.assert_array_equal(far,evidence['stable_far_counts'][evidence['sample_indices'][removed]].ravel())
    ix={int(ti):i for i,ti in enumerate(evidence['candidates'])}
    np.testing.assert_array_equal(trusted,evidence['trusted_far_by_camera'].sum(0)[[ix[int(ti)] for ti in removed]].ravel())
    atomic_json(root/'independent_audit.json',dict(frame=frame,removed_triangles=len(removed),
        all_removed_samples_replayed=True,independent_native_footprints=True,near_native_radius=radius,
        far_corroboration_helper_reused=True,vertices_and_triangle_subset_verified=True,
        near_max=int(near.max(initial=0)),stable_far_min=int(far.min(initial=62)),
        trusted_far_min=int(trusted.min(initial=62)),per_camera=per_camera,
        request_sha256=sha(root/'request.json'),result_sha256=sha(root/'result.json'),
        script_sha256=sha(__file__),quality_approval=False))
    print(frame,'independent replay passed',len(removed),'removed triangles',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',default='000995',choices=['000995'])
    p.add_argument('--center',action='store_true');a=p.parse_args()
    audit(a.frame,Path('/mnt/data/dec5_measured_free_center_pruning') if a.center else ROOT)
