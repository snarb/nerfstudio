"""Same raw surface/certificates; reinstate only the original centroid-gap gate."""
import argparse,time
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from study_close_boundary_completion import ROOT,SOURCE
from study_confidence_depth_prior import load_real
from guard_jaw_measured_depth import measured_pixel_veto
from diffusion_mesh_repair import scene_for


def run(frame):
    root=ROOT/frame;out=root/'matched_gap';out.mkdir(exist_ok=False)
    original=read(SOURCE/frame/'request.json');result=read(root/'interpolated'/frame/'result.json')
    for name,h in result['hashes'].items():
        if sha(root/'interpolated'/frame/name)!=h:raise ValueError('Changed completion')
    local=o3d.io.read_triangle_mesh(str(root/'local_raw.ply'));v=np.asarray(local.vertices);nt=result['original_triangles']
    original_mesh=o3d.io.read_triangle_mesh(original['source_mesh']);ot=np.asarray(original_mesh.triangles)
    np.testing.assert_array_equal(np.asarray(local.triangles)[:nt],ot)
    evidence=np.load(root/'interpolated'/frame/'evidence.npz');admission=np.load(root/'admission/samples.npz')
    proposals=np.load(root/'proposal_evidence.npz')['proposals']
    scene=scene_for(np.asarray(original_mesh.vertices),ot)
    centers=v[proposals].mean(1);closest=scene.compute_closest_points(o3d.core.Tensor(centers.astype(np.float32)))['points'].numpy()
    gap=np.linalg.norm(centers-closest,axis=1);eligible=admission['semantic_ids'][admission['strict']|evidence['prior']]
    ids=eligible[gap[eligible]>=.00002];t=np.concatenate([ot,proposals[ids]])
    rows,depths,receipt=load_real(Path(original['depth_root']),frame)
    if receipt!=original['depth_receipt']:raise ValueError('Changed measured depths')
    atomic_json(out/'request.json',dict(frame=frame,shared_local_mesh_sha256=sha(root/'local_raw.ply'),
        shared_admission_sha256=sha(root/'admission/samples.npz'),shared_certificates_sha256=sha(root/'interpolated'/frame/'evidence.npz'),
        minimum_centroid_gap=.00002,comparison='same raw surface, semantic samples and observed-neighborhood certificates',
        source_mesh_sha256=original['source_mesh_sha256'],script_sha256=sha(__file__),heldout_used=False))
    rounds=[]
    for iteration in range(8):
        scene=scene_for(v,t);remove=set();checks=[]
        for ci,(row,depth) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                bad,count,raw_count=measured_pixel_veto(scene,row,depth,rows,depths,nt,len(t),offset)
                remove.update(bad.tolist());checks.append(dict(camera=row['physical_camera'],offset=offset,qualified_free=count))
            if (ci+1)%10==0:atomic_json(out/'progress.json',dict(stage='native_guard',camera_count=ci+1,iteration=iteration,unix_time=time.time()))
        rounds.append(dict(removed=len(remove),checks=checks))
        if not remove:break
        keep=np.ones(len(t),bool);keep[list(remove)]=False
        if not keep[:nt].all():raise ValueError('Original geometry removal')
        ids=ids[keep[nt:]];t=t[keep]
    if rounds[-1]['removed']:raise ValueError('Guard did not converge')
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out/'mesh.ply'),mesh)
    np.savez_compressed(out/'evidence.npz',centroid_gap=gap,retained_proposal_ids=ids,eligible=eligible)
    atomic_json(out/'result.json',dict(request_sha256=sha(out/'request.json'),added=len(ids),rounds=rounds,
        observed_guard_passed=True,shared_raw_surface=True,original_prefix_exact=True,
        hashes={n:sha(out/n) for n in ['mesh.ply','evidence.npz']}))
    print('matched gap control',frame,len(ids),'faces',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001193','001195'],required=True);run(p.parse_args().frame)
