"""Replay bounded removal, measured solve, assembly and native ray checks."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from study_confidence_depth_prior import unproject,project_integer
from bounded_surface_replacement import removable_faces
from confidence_boundary_completion import solve_depth,grid_faces
from annotation_mask_domain import semantic_faces
from contrastive_forearm_depth_guard import make_guard
from diffusion_mesh_repair import scene_for
import study_forearm_plane_transfer_v3 as prior


def run(root,frame):
    folder=root/frame;request=read(folder/'request.json');result=read(folder/'geometry_result.json')
    if result['request_sha256']!=sha(folder/'request.json'):raise ValueError('Changed request')
    for n,h in result['hashes'].items():
        if sha(folder/n)!=h:raise ValueError('Changed output')
    for n,h in request['scripts'].items():
        if sha(Path(__file__).with_name(n))!=h:raise ValueError('Changed producer helper')
    if sha(request['source_mesh'])!=request['source_mesh_sha256']:raise ValueError('Changed source mesh')
    prior.configure();v1=prior.v2.v1;rows,depths,hashes=v1.load_real(frame)
    if hashes!=request['source_depth_sha256']:raise ValueError('Changed native depths')
    camera=request['reference_camera'];ci=next(i for i,r in enumerate(rows) if r['physical_camera']==camera['physical_camera'])
    if camera!=rows[ci]:raise ValueError('Camera mismatch')
    ev=np.load(folder/'evidence.npz');domain=ev['domain'];model=ev['model'];pins=ev['pins'];observed=depths[ci]
    diagnostic=np.load(prior.OUT/frame/'diagnostic.npz');expected_pins=domain&diagnostic[camera['physical_camera']+'_trusted']&(np.abs(observed-model)<=.012)
    if not np.array_equal(pins,expected_pins):raise ValueError('Trusted pin mismatch')
    solved,_=solve_depth(domain,model,observed,pins,.05)
    if not np.array_equal(solved,ev['solved']) or np.max(np.abs(solved[domain]-model[domain]))>.012:raise ValueError('Solve replay mismatch')
    old=o3d.io.read_triangle_mesh(request['source_mesh']);ov,ot=np.asarray(old.vertices),np.asarray(old.triangles)
    uv,z=project_integer(camera,ov);removed=removable_faces(uv,z,ot,domain,solved,.012)
    if not np.array_equal(removed,ev['removed_original_faces']):raise ValueError('Removal exceeds bounded domain')
    retained=ot[~removed];y,x=np.nonzero(domain);vertices=np.concatenate([ov,unproject(camera,x,y,solved[y,x])])
    index=np.full(domain.shape,-1,int);index[y,x]=np.arange(len(x))+len(ov)
    faces=grid_faces(domain,domain,index);faces,_=semantic_faces(vertices,faces,rows,v1.masks(frame),axis_extent=True)
    raw=o3d.io.read_triangle_mesh(str(folder/'transferred.ply'));mesh=o3d.io.read_triangle_mesh(str(folder/'guarded.ply'))
    v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    if not np.array_equal(np.asarray(raw.vertices),vertices) or not np.array_equal(np.asarray(raw.triangles),np.concatenate([retained,faces])):raise ValueError('Raw assembly replay mismatch')
    if not np.array_equal(v,vertices) or not np.array_equal(t[:len(retained)],retained):raise ValueError('Protected geometry changed')
    allowed={tuple(f) for f in faces}
    if any(tuple(f) not in allowed for f in t[len(retained):]):raise ValueError('Unproposed final face')
    guard,calls,provenance=make_guard(frame,rows,depths,.01)
    if provenance!=result['color_guard_provenance']:raise ValueError('Changed RGB evidence')
    scene=scene_for(v,t)
    for row,depth in zip(rows,depths):
        for offset in [0,.5]:
            ids,count,_=guard(scene,row,depth,rows,depths,len(retained),len(t),offset)
            if len(ids) or count:raise ValueError('Qualified free-space contradiction')
    if calls!=result['color_guard_calls'][-124:]:raise ValueError('Ray evidence mismatch')
    topology=[]
    for label,m in [('previous',old),('raw',raw),('guarded',mesh)]:
        _,counts,_=m.cluster_connected_triangles()
        topology.append(dict(label=label,components=len(counts),largest_triangles=max(counts),
            small_components_below_100=sum(c<100 for c in counts),nonmanifold_edges=len(m.get_non_manifold_edges(allow_boundary_edges=True))))
    atomic_json(folder/'independent_audit.json',dict(frame=frame,geometry_result_sha256=sha(folder/'geometry_result.json'),
        script_sha256=sha(__file__),bounded_old_face_removal_exact=True,trusted_pins_exact=True,solve_replay_exact=True,
        assembly_replay_exact=True,unaffected_old_geometry_exact=True,checks=124,qualified_veto_pixels=0,topology=topology,
        production_accepted=False,visual_review_required=True))
    print(frame,'independent audit passed',topology,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',required=True)
    a=p.parse_args();run(a.root,a.frame)
