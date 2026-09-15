"""Single controlled ablation: native-center near evidence vs 5x5 protection.

Everything else, including six stable far footprints and independently supported
farther observations at all four triangle samples, stays frozen. Not production.
"""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from prune_measured_free_surface import ROOT as PARENT,near_tap_evidence,removable
from review_full_block_transfer import ROOT as DEPTH_ROOT
from study_confidence_depth_prior import load_real,project_integer,support,unproject
from diagnose_jaw_measured_depth import observed_at

ROOT=Path('/mnt/data/dec5_measured_free_center_pruning')


def geometry(frame):
    parent=PARENT/frame;root=ROOT/frame;root.mkdir(parents=True,exist_ok=False)
    q=read(parent/'request.json');r=read(parent/'result.json')
    assert r['request_sha256']==sha(parent/'request.json')
    assert sha(parent/'evidence.npz')==r['hashes']['evidence.npz']
    assert sha(q['mesh'])==q['mesh_sha256']
    for p,digest in q['scripts'].items():assert sha(p)==digest
    a=np.load(parent/'evidence.npz');points=a['points'];samples=a['sample_indices']
    stable=a['stable_far_counts'];rows,depths,receipt=load_real(DEPTH_ROOT,frame)
    assert receipt==q['depth_receipt']
    q['parent_request_sha256']=sha(parent/'request.json')
    q['parent_evidence_sha256']=sha(parent/'evidence.npz')
    q['parameters']['near_native_radius']=0
    q['ablation']='only near footprint radius 2 -> 0; other gates unchanged'
    q['scripts'][str(Path(__file__).resolve())]=sha(__file__)
    atomic_json(root/'request.json',q)
    near=np.zeros(len(points),np.uint8)
    for row,d in zip(rows,depths):
        uv,z=project_integer(row,points);near+=near_tap_evidence(d,uv,z,radius=0)
    candidates=np.flatnonzero((near[samples]==0).all(1)&(stable[samples]>=6).all(1))
    p=points[samples[candidates]].reshape(-1,3);trusted=np.zeros((62,len(candidates),4),bool)
    for ci,(row,d) in enumerate(zip(rows,depths)):
        xy,z,obs,ok=observed_at(p,row,d);j=np.flatnonzero(ok&(obs>z+np.maximum(.005,.01*z)))
        if len(j):
            count,_=support(unproject(row,xy[j,0],xy[j,1],obs[j]),row,rows,depths)
            trusted[ci].reshape(-1)[j]=count>=3
    remove=candidates[removable(near[samples[candidates]],stable[samples[candidates]],trusted.sum(0))]
    mesh=o3d.io.read_triangle_mesh(q['mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    keep=np.ones(len(t),bool);keep[remove]=False;assert keep.any()
    out=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[keep]))
    out.compute_vertex_normals();assert o3d.io.write_triangle_mesh(str(root/'mesh.ply'),out)
    saved=o3d.io.read_triangle_mesh(str(root/'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(saved.vertices),v)
    np.testing.assert_array_equal(np.asarray(saved.triangles),t[keep])
    np.savez_compressed(root/'evidence.npz',points=points,sample_indices=samples,
        near_counts=near,stable_far_counts=stable,candidates=candidates,
        trusted_far_by_camera=trusted,removed_triangle_ids=remove)
    _,components,_=out.cluster_connected_triangles()
    atomic_json(root/'result.json',dict(request_sha256=sha(root/'request.json'),before_triangles=len(t),
        after_triangles=int(keep.sum()),removed_triangles=len(remove),candidate_triangles=len(candidates),
        components=sorted(map(int,components),reverse=True),vertices_unchanged=True,triangle_subset_exact=True,
        no_component_cleanup=True,production_changed=False,visual_status='pending',
        hashes={n:sha(root/n) for n in ['mesh.ply','evidence.npz']}))
    print(frame,'center-footprint candidates',len(candidates),'removed',len(remove),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['geometry','prepare','render','review'])
    parser.add_argument('--view');a=parser.parse_args();frame='000995'
    if a.action=='geometry':geometry(frame)
    else:
        import review_measured_free_surface as workflow
        workflow.ROOT=ROOT
        if a.action=='render':
            if a.view not in workflow.VIEWS:parser.error('Expected one reviewed view')
            workflow.render(frame,a.view)
        else:getattr(workflow,a.action)(frame)
