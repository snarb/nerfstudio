"""Local continuous-surface prior; never replace observed production geometry."""
from pathlib import Path
import argparse,time
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from joint_temporal_texture import read,sha,atomic_json,cameras
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth

OUT=Path('/mnt/data/dec5_poisson_jaw_completion')
SOURCE=Path('/mnt/data/dec5_jaw_measured_mask_control/001193')
FRAME='001193'


def prepare(output):
    output.mkdir(parents=True,exist_ok=False)
    source=read(SOURCE/'request.json')
    if sha(source['source_mesh'])!=source['source_mesh_sha256']:raise ValueError('Changed source mesh')
    mesh=o3d.io.read_triangle_mesh(source['source_mesh']);mesh.compute_vertex_normals();mesh.compute_triangle_normals()
    v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    request=dict(frame=FRAME,source_request_sha256=sha(SOURCE/'request.json'),source_mesh=source['source_mesh'],source_mesh_sha256=source['source_mesh_sha256'],
        sample_count=150000,seed=17,min_head_x=-.03,poisson_depth=10,poisson_scale=1.05,poisson_linear_fit=True,n_threads=4,
        maximum_original_surface_distance=.003,maximum_boundary_distance=.003,maximum_triangle_edge=.0015,
        minimum_centroid_distance=.00002,nearest_boundary_barycentric_threshold=.03,minimum_normal_dot=.25,
        heldout_used=False,geometry_uses_target=False,inferred_not_measured_geometry=True,
        open3d_version=o3d.__version__,script_sha256=sha(__file__))
    atomic_json(output/'request.json',request)
    selected=(v[t,:,][...,0]>request['min_head_x']).all(1)
    head=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[selected]));head.remove_unreferenced_vertices();head.compute_triangle_normals()
    o3d.utility.random.seed(request['seed'])
    cloud=head.sample_points_uniformly(number_of_points=request['sample_count'],use_triangle_normal=True)
    o3d.io.write_point_cloud(str(output/'oriented_samples.ply'),cloud)
    atomic_json(output/'progress.json',dict(stage='poisson_solve',unix_time=time.time()))
    raw,density=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud,depth=10,scale=1.05,linear_fit=True,n_threads=4)
    raw.compute_triangle_normals();rv=np.asarray(raw.vertices);rt=np.asarray(raw.triangles)
    if not np.isfinite(rv).all():raise ValueError('Nonfinite Poisson surface')
    o3d.io.write_triangle_mesh(str(output/'poisson_raw.ply'),raw)
    scene=scene_for(v,t)
    closest=scene.compute_closest_points(o3d.core.Tensor(rv.astype(np.float32)))
    closest_points=closest['points'].numpy();closest_ids=closest['primitive_ids'].numpy();uv=closest['primitive_uvs'].numpy()
    distance=np.linalg.norm(rv-closest_points,axis=1)
    edges,counts=np.unique(np.sort(t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    boundary=edges[counts==1];boundary_vertices=np.unique(boundary)
    boundary_distance=cKDTree(v[boundary_vertices]).query(rv)[0]
    normal_dot=np.sum(np.asarray(raw.vertex_normals)*np.asarray(mesh.triangle_normals)[closest_ids],axis=1) if raw.has_vertex_normals() else None
    raw.compute_vertex_normals();normal_dot=np.sum(np.asarray(raw.vertex_normals)*np.asarray(mesh.triangle_normals)[closest_ids],axis=1)
    vertex_good=(distance<=.003)&(boundary_distance<=.003)&(rv[:,0]>-.03)&(normal_dot>=.25)
    centers=rv[rt].mean(1);cc=scene.compute_closest_points(o3d.core.Tensor(centers.astype(np.float32)))
    center_distance=np.linalg.norm(centers-cc['points'].numpy(),axis=1)
    bary=np.column_stack([1-cc['primitive_uvs'].numpy().sum(1),cc['primitive_uvs'].numpy()])
    near_boundary_face=np.isin(t[cc['primitive_ids'].numpy()],boundary_vertices).any(1)
    lengths=np.linalg.norm(rv[rt[:,[0,1,2]]]-rv[rt[:,[1,2,0]]],axis=2).max(1)
    keep=vertex_good[rt].all(1)&(center_distance>=.00002)&(bary.min(1)<=.03)&near_boundary_face&(lengths<=.0015)
    proposal=rt[keep];used=np.unique(proposal);remap=np.full(len(rv),-1,int);remap[used]=np.arange(len(used))
    vv=np.concatenate([v,rv[used]]);pp=remap[proposal]+len(v);tt=np.concatenate([t,pp])
    candidate=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt));candidate.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output/'local_raw.ply'),candidate)
    np.savez_compressed(output/'proposal_evidence.npz',raw_vertex_ids=used,raw_triangle_ids=np.flatnonzero(keep),proposals=pp,
        closest_points=closest_points[used],closest_triangle_ids=closest_ids[used],distance=distance[used],
        boundary_distance=boundary_distance[used],normal_dot=normal_dot[used],density=np.asarray(density)[used])
    rows,_,_=cameras(FRAME)
    moving=next(r for r in read('/mnt/data/dec5_phase30_early_texture_dynamic_150/request.json')['inventory'] if r['frame_id']==FRAME)['camera']
    controls=[('moving',moving)]+[(n,next(r for r in rows if r['physical_camera']==n)) for n in ['F004_E005_1210FP','D004_D005_1210LZ']]
    new=scene_for(vv,tt);normals=np.asarray(candidate.triangle_normals);stats=[]
    for name,camera in controls:
        d,ids,_=camera_depth(new,camera);old,_,_=camera_depth(scene,camera);valid=np.isfinite(d)
        rgb=np.zeros((*d.shape,3),np.uint8);light=np.abs(normals@np.array([.3,.4,.866]))
        rgb[valid]=(60+170*light[ids[valid],None]).astype(np.uint8);rgb[valid&(ids>=len(t))]=[255,60,40]
        Image.fromarray(np.rot90(rgb)).save(output/(name+'_raw_added.png'))
        stats.append(dict(camera=name,newly_visible=int((valid&~np.isfinite(old)).sum()),added_visible=int((valid&(ids>=len(t))).sum())))
    atomic_json(output/'result.json',dict(request_sha256=sha(output/'request.json'),raw_vertices=len(rv),raw_triangles=len(rt),
        original_vertices=len(v),original_triangles=len(t),local_added_vertices=len(used),local_proposals=len(pp),controls=stats,
        hashes={n:sha(output/n) for n in ['oriented_samples.ply','poisson_raw.ply','local_raw.ply','proposal_evidence.npz']},
        observed_depth_guard_passed=False,production_accepted=False,visual_status='pending'))
    print('Poisson local proposals',len(pp),'controls',stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);prepare(p.parse_args().output)
