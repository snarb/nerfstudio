"""Replay proposal provenance and fresh 124-ray safety checks for each inset arm."""
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json, cameras
from probe_inset_head_completion import ROOT, RAW, SOURCE, MASKS, FRAMES, inset_vertices
from study_confidence_depth_prior import load_real
from study_jaw_repair_transfer import mask_votes
from guard_jaw_measured_depth import measured_pixel_veto
from diffusion_mesh_repair import scene_for


def run():
    records=[]
    for frame in FRAMES:
        folder=ROOT/frame; req=read(folder/'request.json'); arm=folder/'inset_001000'; final=folder/'guarded'
        b=read(SOURCE/frame/'request.json'); g=read(final/'result.json'); q=read(final/'request.json')
        assert g['native_free_space_guard_passed'] and sha(final/'request.json')==g['request_sha256']
        for data in [req,q]:
            for p,h in data['scripts'].items():assert sha(p)==h,p
        assert sha(arm/'mesh.ply')==q['source_mesh_sha256']
        assert sha(final/'mesh.ply')==g['hashes']['mesh.ply']
        assert sha(RAW/frame/'poisson_raw.ply')==req['raw_mesh_sha256']
        assert sha(MASKS/frame/'masks.npz')==req['refined_masks_sha256']
        assert sha(b['source_mesh'])==req['source_mesh_sha256']
        original=o3d.io.read_triangle_mesh(b['source_mesh']);v=np.asarray(original.vertices);t=np.asarray(original.triangles)
        raw=o3d.io.read_triangle_mesh(str(RAW/frame/'poisson_raw.ply')); rv=np.asarray(raw.vertices);rt=np.asarray(raw.triangles)
        center=np.median(v[v[:,0]>req['min_head_x']],axis=0)
        np.testing.assert_array_equal(center,req['center'])
        candidate=o3d.io.read_triangle_mesh(str(arm/'mesh.ply'));cv=np.asarray(candidate.vertices);ct=np.asarray(candidate.triangles)
        np.testing.assert_array_equal(cv,np.concatenate([v,inset_vertices(rv,center,.001)]))
        a=np.load(arm/'evidence.npz'); rows,_,_=cameras(frame)
        masks=np.load(MASKS/frame/'masks.npz')['masks'];names=read(MASKS/frame/'cameras.json')
        s,o=mask_votes(cv[len(v):],rt[a['proposal_ids']],rows,masks,names)
        np.testing.assert_array_equal(s,a['mask_support']);np.testing.assert_array_equal(o,a['mask_outside'])
        retained=a['proposal_ids'][(s>=2)&(o==0)]
        np.testing.assert_array_equal(retained,a['retained_raw_triangle_ids'])
        np.testing.assert_array_equal(ct,np.concatenate([t,rt[retained]+len(v)]))
        mesh=o3d.io.read_triangle_mesh(str(final/'mesh.ply'));mv=np.asarray(mesh.vertices);mt=np.asarray(mesh.triangles)
        f=np.load(final/'evidence.npz')['retained_candidate_triangle_ids']
        np.testing.assert_array_equal(mv,cv);np.testing.assert_array_equal(mt,np.concatenate([t,ct[len(t):][f]]))
        rows,depths,receipt=load_real(Path(b['depth_root']),frame);assert receipt==q['depth_receipt']
        scene=scene_for(mv,mt);checks=[]
        for ci,(row,d) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                ids,count,raw_count=measured_pixel_veto(scene,row,d,rows,depths,len(t),len(mt),offset)
                assert not len(ids) and count==0
                checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
            if (ci+1)%16==0:print(frame,'fresh guard cameras',ci+1,flush=True)
        components,_,_=mesh.cluster_connected_triangles()
        records.append(dict(frame=frame,added_triangles=len(f),original_prefix_exact=True,
            shifted_geometry_replayed=True,semantic_admission_replayed=True,native_ray_checks=checks,
            mesh_components=len(np.unique(components)),nonmanifold_edges=len(mesh.get_non_manifold_edges(allow_boundary_edges=True)),
            mesh_sha256=sha(final/'mesh.ply'),inferred_not_measured=True,observed_neighborhood_certificates=False))
    atomic_json(ROOT/'geometry_audit.json',dict(records=records,production_updated=False,
        script_sha256=sha(__file__),quality_approval=False))
    print('Geometry audits complete',flush=True)


if __name__=='__main__':run()
