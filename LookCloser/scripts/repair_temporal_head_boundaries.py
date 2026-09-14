"""Opt-in local boundary completion; a shape prior, not recovered TSDF evidence.

Split zero-area topological joins without moving existing triangles, then close
small head boundary loops. Preserve the open torso and original surface detail.
"""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
import pymeshlab as ml
from joint_temporal_texture import read,sha,atomic_json
from local_mesh_repair import boundary_loops

SETTINGS=dict(min_head_x=-.03,max_loop_extent=.03,max_loop_edges=1000,edge_length=.0005)

def close_loops(v,t,loops):
    """Triangulate each isolated boundary, never select neighboring mesh faces."""
    import mapbox_earcut
    from shapely.geometry import Polygon
    def edges(faces):
        u,c=np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
        return set(map(tuple,u[c==1])),int((c>2).sum())
    existing_edges=set(map(tuple,np.sort(t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)))
    existing_faces=set(map(tuple,np.sort(t,axis=1)))
    added=[];allowed=set();skipped=[];new_vertices=[];membranes=[]
    for index,loop in enumerate(loops):
        points=v[loop];centered=points-points.mean(0);basis=np.linalg.svd(centered,full_matrices=False)[2]
        polygon=centered@basis[:2].T
        wanted={tuple(sorted((int(a),int(b)))) for a,b in zip(loop,np.roll(loop,-1))}
        faces=np.empty((0,3),int)
        if Polygon(polygon).is_valid:
            faces=loop[mapbox_earcut.triangulate_float64(np.ascontiguousarray(polygon),np.array([len(loop)],np.uint32)).reshape(-1,3)]
        boundary,nonmanifold=edges(faces)
        candidate_edges=set(map(tuple,np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)))
        if any(tuple(face) in existing_faces for face in np.sort(faces,axis=1)):
            skipped.append(index);continue
        if boundary!=wanted or nonmanifold or ((candidate_edges-wanted)&existing_edges):
            # A local membrane is an explicit shape prior for curved boundaries
            # without a simple planar projection. Never expand its convex hull.
            center_id=len(v)+len(new_vertices);new_vertices.append(points.mean(0));membranes.append(index)
            faces=np.column_stack((np.roll(loop,-1),loop,np.full(len(loop),center_id)))
            candidate_edges=set(map(tuple,np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)))
        directed=set(map(tuple,faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)))
        if (loop[0],loop[1]) in directed:faces=faces[:,::-1]
        added.append(faces);allowed|=wanted;existing_edges|=candidate_edges
    tt=np.concatenate([t,*added]);vv=np.concatenate([v,np.array(new_vertices).reshape(-1,3)]);before,nb=edges(t);after,na=edges(tt)
    if after-before or (before-after)!=allowed or na>nb:raise ValueError('Unexpected boundary topology')
    return vv,tt,dict(method='isolated ear triangulation with bounded curved-loop membrane prior; original boundary positions fixed',
        added_triangles=len(tt)-len(t),added_vertices=len(new_vertices),closed_boundary_edges=len(before-after),curved_membrane_loop_indices=membranes,
        skipped_non_simple_projected_loops=skipped,other_boundary_edges_unchanged=True,
        all_original_vertices_unchanged=True,global_self_intersection_free_not_certified=True)

def repair(record,output):
    root=output/record['frame_id'];root.mkdir(parents=True,exist_ok=True)
    request=dict(source_mesh=record['mesh'],source_sha256=record['mesh_sha256'],
                 settings=SETTINGS,script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('local_mesh_repair.py')))
    if (root/'complete.json').exists():
        done=read(root/'complete.json')
        if done['request']!=request or sha(root/'mesh.ply')!=done['mesh_sha256']:raise ValueError('Repair resume mismatch')
        return done
    if sha(request['source_mesh'])!=request['source_sha256']:raise ValueError('Changed original geometry')
    mesh=o3d.io.read_triangle_mesh(request['source_mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    ms=ml.MeshSet();ms.add_mesh(ml.Mesh(vertex_matrix=v,face_matrix=t))
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    vv=ms.current_mesh().vertex_matrix();tt=ms.current_mesh().face_matrix()
    if not np.array_equal(v[t],vv[tt]):raise ValueError('Topology split moved original surface')
    loops,rejected=boundary_loops(tt);selected=[];entries=[]
    for index,loop in enumerate(loops):
        points=vv[loop]
        if points[:,0].min()>SETTINGS['min_head_x'] and np.ptp(points,axis=0).max()<SETTINGS['max_loop_extent'] and len(loop)<SETTINGS['max_loop_edges']:
            selected.append(loop);entries.append(dict(loop=index,vertices=loop.tolist(),center=points.mean(0).tolist(),extent=np.ptp(points,axis=0).tolist()))
    if selected:vv,tt,operation=close_loops(vv,tt,selected)
    else:operation=dict(added_triangles=0,added_vertices=0)
    # Preserve the existing filtered head too: restoring all removed head faces
    # caused brown background-bearing spikes in the expanded-view canary.
    result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt))
    result.remove_unreferenced_vertices();result.compute_vertex_normals();o3d.io.write_triangle_mesh(str(root/'mesh.ply'),result)
    atomic_json(root/'operations.json',dict(selected=entries,operation=operation,rejected_boundary_components=len(rejected),
        original_triangle_positions_preserved=True,preserve_existing_semantic_carving=True,
        geometric_completion_is_prior=True,uses_synthetic_rgb=False,uses_heldout_rgb=False))
    done=dict(request=request,mesh_sha256=sha(root/'mesh.ply'),operations_sha256=sha(root/'operations.json'),added_triangles=operation['added_triangles'])
    atomic_json(root/'complete.json',done);print(f'repair={record["frame_id"]} loops={len(selected)} added={operation["added_triangles"]}',flush=True);return done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,default=Path('/mnt/data/dec5_screen_travel_dynamic_150_v2'))
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_expanded_head_repairs'));p.add_argument('--frames',nargs='+');a=p.parse_args()
    for record in read(a.parent/'request.json')['inventory']:
        if a.frames is None or record['frame_id'] in a.frames:repair(record,a.output)
