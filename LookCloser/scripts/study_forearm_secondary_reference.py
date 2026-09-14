"""Opt-in secondary-reference tessellation of the unchanged fitted quadric.

Preserves the complete starting mesh; only strict probe candidates are used.
This is inferred local completion, not new measured anatomy or a production fix.
"""
from pathlib import Path
import argparse,time
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation
from joint_temporal_texture import read,sha,atomic_json
from study_confidence_depth_prior import raycast_integer,unproject
from confidence_boundary_completion import grid_faces
from annotation_mask_domain import semantic_faces
from diffusion_mesh_repair import scene_for
from probe_forearm_reference_coverage import BASE,OUT as PROBE
import study_forearm_plane_transfer_v3 as prior

OUT=Path('/mnt/data/dec5_forearm_secondary_reference')


def prepare(output,frame):
    prior.configure();v1=prior.v2.v1
    rows,depths,hashes=v1.load_real(frame);masks=v1.masks(frame)
    folder=output/frame;folder.mkdir(parents=True,exist_ok=True)
    source=BASE/frame/'guarded.ply';previous=read(BASE/frame/'geometry_result.json')
    if sha(source)!=previous['hashes']['guarded.ply']:raise ValueError('Changed starting mesh')
    probe=read(PROBE/frame/'result.json')
    if probe['request_sha256']!=sha(PROBE/frame/'request.json'):raise ValueError('Changed probe request')
    probe_request=read(PROBE/frame/'request.json')
    if probe_request['mesh_sha256']!=sha(source) or probe_request['depth_sha256']!=hashes:raise ValueError('Probe input mismatch')
    names=v1.NAMES[1:];paths=[PROBE/frame/(n+'.npz') for n in names]
    for n,p in zip(names,paths):
        record=next(r for r in probe['records'] if r['camera']==n)
        if sha(p)!=record['arrays_sha256']:raise ValueError('Changed probe arrays')
    helpers=[Path(__file__).name,'probe_forearm_reference_coverage.py','forearm_quadric_rays.py',
        'confidence_boundary_completion.py','annotation_mask_domain.py','contrastive_forearm_depth_guard.py',
        'contrastive_forearm_witnesses.py','rgb_qualified_forearm_depth_guard.py','forearm_rgb_witnesses.py',
        'photometric_forearm_depth_guard.py','diagnose_forearm_color_witnesses.py','guard_jaw_measured_depth.py']
    request=dict(frame=frame,starting_mesh=str(source),starting_mesh_sha256=sha(source),
        probe_result_sha256=sha(PROBE/frame/'result.json'),probe_request_sha256=sha(PROBE/frame/'request.json'),
        source_depth_sha256=hashes,reference_order=names,strict_candidates_only=True,
        sequential_hole_only=True,ring_from_starting_mesh=True,maximum_axis_extent=.002,
        free_depth_separation=.003,witness_rgb_limit=.12,witness_comparison_margin=.01,
        ray_offsets=[0,.5],maximum_pruning_rounds=8,production_changed=False,heldout_used=False,
        inferred_quadric_not_measured_anatomy=True,scripts={n:sha(Path(__file__).with_name(n)) for n in helpers})
    if (folder/'request.json').exists() and read(folder/'request.json')!=request:raise ValueError('Frozen study mismatch')
    atomic_json(folder/'request.json',request)
    if (folder/'geometry_result.json').exists():
        result=read(folder/'geometry_result.json')
        if result['request_sha256']!=sha(folder/'request.json'):raise ValueError('Changed completed request')
        for p,h in result['hashes'].items():
            if sha(folder/p)!=h:raise ValueError('Changed completed artifact')
        print(frame,'verified completed',flush=True);return
    original=o3d.io.read_triangle_mesh(str(source));ov=np.asarray(original.vertices);ot=np.asarray(original.triangles)
    vertices=ov.copy();triangles=ot.copy();assembly=[]
    for name,path in zip(names,paths):
        camera=next(r for r in rows if r['physical_camera']==name)
        observed=raycast_integer(scene_for(vertices,triangles),camera);a=np.load(path)
        x,y=a['xy'].T;selected=a['strict']&(observed[y,x]==0)
        added=np.zeros(observed.shape,float);added[y[selected],x[selected]]=a['depth'][selected]
        active=added>0;domain=binary_dilation(active)&((observed>0)|active)
        yy,xx=np.nonzero(domain);values=np.where(active,added,observed)[yy,xx]
        if not np.isfinite(values).all() or (values<=0).any():raise ValueError('Invalid grid depth')
        index=np.full(observed.shape,-1,int);index[yy,xx]=np.arange(len(xx))+len(vertices)
        vertices=np.concatenate([vertices,unproject(camera,xx,yy,values)])
        faces=grid_faces(domain,active,index)
        faces,semantic=semantic_faces(vertices,faces,rows,masks,axis_extent=True)
        triangles=np.concatenate([triangles,faces])
        assembly.append(dict(camera=name,eligible_missing_pixels=int(active.sum()),added_triangles=len(faces),semantic=semantic))
    def save(name,faces):
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces));mesh.compute_vertex_normals()
        if not o3d.io.write_triangle_mesh(str(folder/name),mesh):raise IOError('Mesh write failed')
    save('transferred.ply',triangles)
    from contrastive_forearm_depth_guard import make_guard
    veto,calls,provenance=make_guard(frame,rows,depths,.01);rounds=[]
    for iteration in range(8):
        scene=scene_for(vertices,triangles);remove=set();checks=[]
        for ci,(camera,depth) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                ids,count,raw=veto(scene,camera,depth,rows,depths,len(ot),len(triangles),offset)
                remove.update(ids.tolist());checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw))
            if (ci+1)%10==0:
                atomic_json(folder/'progress.json',dict(stage='contrastive_native_guard',iteration=iteration,cameras=ci+1,flagged=len(remove),unix_time=time.time()))
                print(frame,iteration,ci+1,len(remove),flush=True)
        rounds.append(dict(iteration=iteration,removed_triangles=len(remove),checks=checks))
        if not remove:break
        if min(remove)<len(ot):raise ValueError('Starting mesh removal attempted')
        keep=np.ones(len(triangles),bool);keep[list(remove)]=False;triangles=triangles[keep]
    save('guarded.ply',triangles)
    saved=o3d.io.read_triangle_mesh(str(folder/'guarded.ply'))
    if not np.array_equal(np.asarray(saved.vertices)[:len(ov)],ov) or not np.array_equal(np.asarray(saved.triangles)[:len(ot)],ot):raise ValueError('Starting prefix changed')
    result=dict(request_sha256=sha(folder/'request.json'),assembly=assembly,rounds=rounds,
        observed_guard_passed=not rounds[-1]['removed_triangles'],final_added_triangles=len(triangles)-len(ot),
        color_guard_calls=calls,color_guard_provenance=provenance,original_mesh_prefix_exact=True,
        visual_status='pending',production_accepted=False,
        hashes={n:sha(folder/n) for n in ['transferred.ply','guarded.ply']})
    atomic_json(folder/'geometry_result.json',result)
    print(frame,'finished',assembly,'final',result['final_added_triangles'],'passed',result['observed_guard_passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001029','001033','001037'],required=True)
    p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();prepare(a.output,a.frame)
