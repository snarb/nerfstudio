"""Independent final-mesh replay of strict semantics and 124 native ray checks."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from annotation_mask_domain import semantic_faces
from contrastive_forearm_depth_guard import make_guard
from diffusion_mesh_repair import scene_for
import study_forearm_plane_transfer_v3 as prior


def run(root,frame):
    folder=root/frame;request=read(folder/'request.json');result=read(folder/'geometry_result.json')
    if result['request_sha256']!=sha(folder/'request.json'):raise ValueError('Changed request')
    for name,h in request['scripts'].items():
        if sha(Path(__file__).with_name(name))!=h:raise ValueError('Changed producer helper')
    for name,h in result['hashes'].items():
        if sha(folder/name)!=h:raise ValueError('Changed output')
    if sha(request['starting_mesh'])!=request['starting_mesh_sha256']:raise ValueError('Changed starting mesh')
    old=o3d.io.read_triangle_mesh(request['starting_mesh']);mesh=o3d.io.read_triangle_mesh(str(folder/'guarded.ply'))
    v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles);nv,nt=len(old.vertices),len(old.triangles)
    if not np.array_equal(v[:nv],np.asarray(old.vertices)) or not np.array_equal(t[:nt],np.asarray(old.triangles)):raise ValueError('Starting mesh changed')
    if len(t)-nt!=result['final_added_triangles']:raise ValueError('Triangle count mismatch')
    if not np.isfinite(v).all() or (t<0).any() or t.max()>=len(v):raise ValueError('Invalid geometry')
    prior.configure();v1=prior.v2.v1;rows,depths,hashes=v1.load_real(frame)
    if hashes!=request['source_depth_sha256']:raise ValueError('Changed depth evidence')
    selected,semantic=semantic_faces(v,t[nt:],rows,v1.masks(frame),axis_extent=True)
    if not np.array_equal(selected,t[nt:]):raise ValueError('Final semantic/extent mismatch')
    guard,calls,provenance=make_guard(frame,rows,depths,.01)
    if provenance!=result['color_guard_provenance']:raise ValueError('Changed color evidence')
    scene=scene_for(v,t)
    for camera,depth in zip(rows,depths):
        for offset in [0,.5]:
            ids,count,_=guard(scene,camera,depth,rows,depths,nt,len(t),offset)
            if len(ids) or count:raise ValueError('Qualified free-space contradiction')
    if calls!=result['color_guard_calls'][-124:]:raise ValueError('Ray evidence differs from producer')
    topology=[]
    for label,m in [('starting',old),('final',mesh)]:
        _,counts,_=m.cluster_connected_triangles()
        topology.append(dict(label=label,components=len(counts),largest_triangles=max(counts),
            small_components_below_100=sum(c<100 for c in counts),
            nonmanifold_edges=len(m.get_non_manifold_edges(allow_boundary_edges=True))))
    atomic_json(folder/'independent_audit.json',dict(frame=frame,geometry_result_sha256=sha(folder/'geometry_result.json'),
        script_sha256=sha(__file__),checks=124,qualified_veto_pixels=0,starting_prefix_exact=True,
        strict_semantics_exact=True,semantic=semantic,topology=topology,production_accepted=False))
    print(frame,'audit passed',topology,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();run(a.root,a.frame)
