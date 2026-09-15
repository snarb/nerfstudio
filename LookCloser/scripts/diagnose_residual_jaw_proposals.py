"""Trace remaining moving-view jaw misses to individual proposal rejection gates."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth

ROOT=Path('/mnt/data/dec5_residual_jaw_proposals')
BASE=Path('/mnt/data/dec5_poisson_jaw_completion')
MOVIE=Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')


def run():
    ROOT.mkdir(exist_ok=False)
    request=read(MOVIE/'request.json');entry=next(r for r in request['inventory'] if r['frame_id']=='001193')
    target=entry['camera'];proposal=read(BASE/'result.json')
    for name,h in proposal['hashes'].items():
        if sha(BASE/name)!=h:raise ValueError('Changed proposal')
    raw=o3d.io.read_triangle_mesh(str(BASE/'local_raw.ply'))
    v=np.asarray(raw.vertices);t=np.asarray(raw.triangles);nt=proposal['original_triangles']
    depth,faces,_=camera_depth(scene_for(v,t),target)
    depth=np.rot90(depth);faces=np.rot90(faces)
    before=np.rot90(np.load(MOVIE/'frames/001193/target_depth.npz')['depth'])
    candidate=BASE/'interpolated/001193'
    mesh=o3d.io.read_triangle_mesh(str(candidate/'mesh.ply'))
    after=np.rot90(camera_depth(scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles)),target)[0])
    evidence=np.load(candidate/'evidence.npz');samples=np.load(BASE/'admission/samples.npz')
    certificates=read(candidate/'certificates.json')['notes']
    semantic={int(p):i for i,p in enumerate(samples['semantic_ids'])}
    lookup={int(q):i for i,q in enumerate(evidence['query_ids'])}
    retained=set(evidence['retained_proposal_ids'].tolist())
    # Fixed diagnostic rectangle enclosing the isolated fleck, not a quality ROI.
    box=(590,1170,610,1185);x0,y0,x1,y1=box;records=[]
    for yy,xx in np.argwhere(before[y0:y1,x0:x1]==0):
        y,x=int(yy+y0),int(xx+x0);fid=int(faces[y,x]);pid=fid-nt
        row=dict(x=x,y=y,before_missing=True,after_missing=not bool(np.isfinite(after[y,x])),
            raw_missing=not bool(np.isfinite(depth[y,x])),raw_face=fid,proposal_id=pid)
        if np.isfinite(depth[y,x]) and fid>=nt:
            si=semantic.get(pid)
            row.update(retained=pid in retained,semantic_admitted=si is not None,
                mask_support=int(samples['mask_support'][pid]),mask_outside=int(samples['mask_outside'][pid]))
            if si is not None:
                row.update(strict=bool(samples['strict'][si]),prior=bool(evidence['prior'][si]),
                    free_veto=bool(samples['free'][:,si].any()),depth_votes=samples['votes'][si].tolist(),
                    vertex_certificates=[certificates[lookup[int(q)]] for q in t[fid]],
                    certificate_flags=[bool(evidence['certificate'][lookup[int(q)]]) for q in t[fid]])
        records.append(row)
    np.savez_compressed(ROOT/'evidence.npz',before=before,after=after,raw=depth,raw_faces=faces)
    atomic_json(ROOT/'result.json',dict(frame='001193',diagnostic_box=box,records=records,
        script_sha256=sha(__file__),movie_request_sha256=sha(MOVIE/'request.json'),
        candidate_sha256=sha(candidate/'mesh.ply'),arrays_sha256=sha(ROOT/'evidence.npz'),
        geometry_changed=False,quality_metrics_computed=False))
    print(records,flush=True)


def full():
    from scipy.spatial import cKDTree
    entry=next(r for r in read(MOVIE/'request.json')['inventory'] if r['frame_id']=='001193')
    old=o3d.io.read_triangle_mesh(entry['mesh']);old.compute_triangle_normals()
    v=np.asarray(old.vertices);t=np.asarray(old.triangles);scene=scene_for(v,t)
    raw=o3d.io.read_triangle_mesh(str(BASE/'poisson_raw.ply'));raw.compute_vertex_normals()
    rv=np.asarray(raw.vertices);rt=np.asarray(raw.triangles)
    depth,faces,_=camera_depth(scene_for(rv,rt),entry['camera'])
    depth=np.rot90(depth);faces=np.rot90(faces)
    edges,counts=np.unique(np.sort(t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    boundary_vertices=np.unique(edges[counts==1]);tree=cKDTree(v[boundary_vertices])
    records=[]
    for row in read(ROOT/'result.json')['records']:
        if not row['after_missing']:continue
        x,y=row['x'],row['y'];fid=int(faces[y,x])
        record=dict(x=x,y=y,raw_poisson_hit=bool(np.isfinite(depth[y,x])))
        if np.isfinite(depth[y,x]):
            points=rv[rt[fid]];closest=scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
            distance=np.linalg.norm(points-closest['points'].numpy(),axis=1)
            normals=np.asarray(raw.vertex_normals)[rt[fid]]
            dots=(normals*np.asarray(old.triangle_normals)[closest['primitive_ids'].numpy()]).sum(1)
            center=points.mean(0,keepdims=True);cc=scene.compute_closest_points(o3d.core.Tensor(center.astype(np.float32)))
            uv=cc['primitive_uvs'].numpy()[0];bary=np.r_[1-uv.sum(),uv]
            record.update(raw_triangle=fid,original_distance=distance.tolist(),boundary_distance=tree.query(points)[0].tolist(),
                minimum_head_x=float(points[:,0].min()),normal_dot=dots.tolist(),
                centroid_distance=float(np.linalg.norm(center-cc['points'].numpy())),barycentric=bary.tolist(),
                near_boundary_face=bool(np.isin(t[cc['primitive_ids'].numpy()[0]],boundary_vertices).any()),
                max_edge=float(np.linalg.norm(points-points[[1,2,0]],axis=1).max()))
        records.append(record)
    atomic_json(ROOT/'raw_selection.json',dict(records=records,script_sha256=sha(__file__)))
    print(records,flush=True)


if __name__=='__main__':
    import sys
    full() if '--full' in sys.argv else run()
