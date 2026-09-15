"""Replay cap geometry and run fresh 62-camera, two-grid measured-depth guards."""
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from subdivide_boundary_caps import subdivide
from study_subdivided_jaw_caps import OUT,SOURCE
from study_jaw_boundary_notches import propose
from study_confidence_depth_prior import load_real
from guard_jaw_measured_depth import measured_pixel_veto,initial_admission
from diffusion_mesh_repair import scene_for


def run():
    frame='001193';base=SOURCE/frame;bq=read(base/'request.json');br=read(base/'result.json')
    rows,depths,receipt=load_real(Path(bq['depth_root']),frame)
    assert receipt==bq['depth_receipt']
    mesh=o3d.io.read_triangle_mesh(bq['source_mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    audits=[]
    for root in [OUT/'planar',OUT/'curved',Path('/mnt/data/dec5_large_curved_jaw_caps/curved')]:
        folder=root/frame;req=read(folder/'request.json');result=read(folder/'result.json');a=np.load(folder/'evidence.npz')
        assert sha(base/'request.json')==req['source_request_sha256'] and sha(base/'result.json')==req['source_result_sha256']
        assert sha(req['source_mesh'])==req['source_mesh_sha256'] and sha(folder/'request.json')==result['request_sha256']
        for path,digest in result['hashes'].items():assert sha(folder/path)==digest
        for name,digest in req['scripts'].items():
            current=Path(__file__).with_name(name);archived=OUT/'config/initial'/name
            if sha(current)!=digest:
                assert archived.exists() and sha(archived)==digest
        if req.get('large',False):
            raw,notes=propose(v,t,req['cap_settings']);raw=raw[len(t):]
            np.testing.assert_array_equal(raw,a['raw_proposals'])
        else:raw=np.load(base/'evidence.npz')['proposals'];notes=br['proposal_notes']
        vv,pp,fit=subdivide(v,t,raw,notes,curved=req['arm']=='curved')
        np.testing.assert_array_equal(pp,a['proposals'])
        for measured,replayed in zip(result['proposal_fit'],fit):
            assert measured.keys()==replayed.keys()
            for key in measured:np.testing.assert_allclose(measured[key],replayed[key],rtol=0,atol=1e-12)
        ids=a['retained_proposal_ids'];keep=initial_admission(a['votes'],a['free'],a['mask_support'],a['mask_outside'])
        assert keep[ids].all()
        saved=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'));sv=np.asarray(saved.vertices);st=np.asarray(saved.triangles)
        np.testing.assert_array_equal(sv,vv);np.testing.assert_array_equal(st,np.concatenate([t,pp[ids]]))
        scene=scene_for(sv,st);checks=[]
        for camera,depth in zip(rows,depths):
            for offset in [0,.5]:
                implicated,count,raw_count=measured_pixel_veto(scene,camera,depth,rows,depths,len(t),len(st),offset)
                if len(implicated) or count:raise ValueError('Failed independent measured-depth guard')
                checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
        clusters,counts,areas=saved.cluster_connected_triangles()
        audit=dict(mesh_sha256=sha(folder/'mesh.ply'),geometry_replay=True,original_prefix_exact=True,
            fresh_native_ray_checks=checks,nonmanifold_edges=len(saved.get_non_manifold_edges(allow_boundary_edges=True)),
            components=len(counts),largest_component_triangles=int(max(counts)),
            initial_evidence_recomputed=False,observed_depth_receipt=receipt,script_sha256=sha(__file__))
        atomic_json(folder/'audit.json',audit);audits.append(str(folder/'audit.json'));print(root,124,'fresh checks passed',flush=True)
    atomic_json(OUT/'audit_summary.json',dict(audits={p:sha(p) for p in audits},total_fresh_ray_checks=372))


if __name__=='__main__':run()
