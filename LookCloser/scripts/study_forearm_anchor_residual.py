"""Local measured-anchor shape control on the frozen RGB-qualified forearm prior."""
from pathlib import Path
from copy import deepcopy
import argparse
import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from study_confidence_depth_prior import project_integer,unproject
from fit_forearm_anchor_residual import anchor_targets,solve_residual
from annotation_mask_domain import semantic_faces
from rgb_qualified_forearm_depth_guard import make_guard
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
import study_forearm_plane_transfer_v3 as prior

BASE=Path('/mnt/data/dec5_forearm_rgb_qualified_curve')
ANCHORS=Path('/mnt/data/dec5_forearm_multiview_anchors')


def prepare(root,frame):
    prior.configure();v1=prior.v2.v1;start=time.monotonic()
    previous=read(BASE/frame/'geometry_result.json');parent=read(BASE/frame/'request.json')
    if previous['request_sha256']!=sha(BASE/frame/'request.json'):raise ValueError('Changed prior request')
    for n,h in previous['hashes'].items():
        if sha(BASE/frame/n)!=h:raise ValueError('Changed prior geometry')
    apath=ANCHORS/frame/'anchors.npz'
    if sha(apath)!=read(ANCHORS/frame/'result.json')['anchors_sha256']:raise ValueError('Changed anchors')
    source=next(r for r in read('/mnt/data/dec5_phase30_dynamic_150/request.json')['inventory'] if r['frame_id']==frame)
    if sha(source['mesh'])!=source['mesh_sha256']:raise ValueError('Changed original mesh')
    rows,depths,depth_hashes=v1.load_real(frame)
    if depth_hashes!=previous['depth_hashes']:raise ValueError('Changed observed depths')
    ref=next(r for r in rows if r['physical_camera']==v1.NAMES[0])
    old=o3d.io.read_triangle_mesh(source['mesh']);nv,nt=len(old.vertices),len(old.triangles)
    mesh=o3d.io.read_triangle_mesh(str(BASE/frame/'transferred.ply'))
    vertices=np.asarray(mesh.vertices).copy();triangles=np.asarray(mesh.triangles).copy()
    if not np.array_equal(vertices[:nv],np.asarray(old.vertices)) or not np.array_equal(triangles[:nt],np.asarray(old.triangles)):
        raise ValueError('Original prefix mismatch')
    used=np.unique(triangles[nt:]);uv,z=project_integer(ref,vertices[used]);xy=np.rint(uv).astype(int)
    accepted=np.load(prior.OUT/frame/'plane/evidence.npz')['accepted']
    if (xy<0).any() or (xy[:,0]>=1920).any() or (xy[:,1]>=1080).any():raise ValueError('Patch outside grid')
    pinned=(used<nv)|~accepted[xy[:,1],xy[:,0]]
    mapping=np.full(len(vertices),-1,int);mapping[used]=np.arange(len(used));local=mapping[triangles[nt:]]
    edges=np.vstack([local[:,[0,1]],local[:,[1,2]],local[:,[2,0]]])
    anchors=np.load(apath);au,az,ac=anchors['reference_uv'],anchors['reference_z'],anchors['source_index']
    split=ac%2==0
    target,weight,count,assignment=anchor_targets(uv,z,au[split],az[split],ac[split])
    trial,fit=solve_residual(len(used),edges,target,weight,pinned)
    distance,node=cKDTree(uv).query(au[~split]);valid=(distance<=.75)&~pinned[node]
    if valid.sum()<30:raise ValueError('Insufficient disjoint camera validation samples')
    before=z[node[valid]]-az[~split][valid];after=z[node[valid]]+trial[node[valid]]-az[~split][valid]
    validation=dict(samples=int(valid.sum()),before_median=float(np.median(abs(before))),after_median=float(np.median(abs(after))),
        before_p90=float(np.quantile(abs(before),.9)),after_p90=float(np.quantile(abs(after),.9)),
        protocol='even-source residual fit, odd-source depth validation; common all-source base prior, not heldout RGB')
    out=root/frame;out.mkdir(parents=True,exist_ok=False)
    request=deepcopy(parent);request.update(script_sha256=sha(__file__),anchor_residual_helper_sha256=sha(Path(__file__).with_name('fit_forearm_anchor_residual.py')),
        parent_result_sha256=sha(BASE/frame/'geometry_result.json'),anchor_input_sha256=sha(apath),
        residual_parameters=dict(smoothness=.2,prior_weight=.001,max_displacement=.006,assignment_distance_px=.75,
            boundary_ring_fixed=True,camera_node_median=True,require_partition_p90_improvement=True))
    atomic_json(out/'request.json',request)
    atomic_json(out/'partition_validation.json',validation)
    if validation['after_p90']>=validation['before_p90']:
        atomic_json(out/'rejected.json',dict(reason='residual lacks disjoint camera p90 improvement',validation=validation))
        print(frame,'rejected before mesh generation',validation,flush=True);return
    target,weight,count,assignment=anchor_targets(uv,z,au,az,ac)
    delta,fit=solve_residual(len(used),edges,target,weight,pinned)
    if fit['clipped_nodes']>len(used)*.2:raise ValueError('Too many displacement bound violations')
    movable=~pinned;vertices[used[movable]]=unproject(ref,uv[movable,0],uv[movable,1],z[movable]+delta[movable])
    faces,semantic=semantic_faces(vertices,triangles[nt:],rows,v1.masks(frame),axis_extent=True)
    triangles=np.vstack([triangles[:nt],faces])
    def save(name):
        result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(triangles))
        result.compute_vertex_normals()
        if not o3d.io.write_triangle_mesh(str(out/name),result):raise IOError('Mesh write failed')
    save('transferred.ply')
    np.savez_compressed(out/'residual.npz',vertex_ids=used,initial_depth=z,residual=delta,pinned=pinned,
                        reference_uv=uv,anchor_target=target,anchor_weight=weight,camera_count=count)
    guard,calls,provenance=make_guard(frame,rows,depths,rgb_limit=.12);rounds=[]
    for iteration in range(8):
        scene=scene_for(vertices,triangles);remove=set();checks=[]
        for ci,(camera,observed) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                ids,n,raw=guard(scene,camera,observed,rows,depths,nt,len(triangles),offset)
                remove.update(ids.tolist());checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=n,raw_far_pixels=raw))
            if (ci+1)%10==0:
                atomic_json(out/'progress.json',dict(stage='rgb_depth_guard',iteration=iteration,cameras=ci+1,flagged=len(remove),unix_time=time.time()))
                print(frame,iteration,ci+1,len(remove),flush=True)
        rounds.append(dict(iteration=iteration,removed_triangles=len(remove),checks=checks))
        if not remove:break
        keep=np.ones(len(triangles),bool);keep[list(remove)]=False
        if not keep[:nt].all():raise ValueError('Original triangle removal')
        triangles=triangles[keep]
    save('guarded.ply');passed=not rounds[-1]['removed_triangles']
    if not np.array_equal(vertices[:nv],np.asarray(old.vertices)):raise ValueError('Original vertex change')
    panel=Image.new('RGB',(860,495));draw=ImageDraw.Draw(panel)
    for i,path in enumerate([BASE/frame/'guarded.ply',out/'guarded.ply']):
        m=o3d.io.read_triangle_mesh(str(path));m.compute_triangle_normals();scene=scene_for(np.asarray(m.vertices),np.asarray(m.triangles))
        d,ids,_=camera_depth(scene,source['camera']);valid=np.isfinite(d);im=np.zeros((*d.shape,3),np.uint8)
        light=np.abs(np.asarray(m.triangle_normals)@np.array([.3,.4,.866]));im[valid]=(60+170*light[ids[valid],None]).astype(np.uint8)
        im[valid&(ids>=nt)]=[240,60,50]
        panel.paste(Image.fromarray(np.rot90(im)).crop((0,1450,430,1920)),(i*430,25));draw.text((i*430+3,4),['previous','anchor residual'][i],fill='white')
    panel.save(out/'moving_forearm_clay_native.png')
    atomic_json(out/'geometry_result.json',dict(request_sha256=sha(out/'request.json'),final_added_triangles=len(triangles)-nt,
        observed_guard_passed=passed,rounds=rounds,depth_hashes=depth_hashes,color_guard_calls=calls,color_guard_provenance=provenance,
        fit=fit,assignment=assignment,semantic=semantic,partition_validation=validation,production_accepted=False,visual_status='pending',
        elapsed_seconds=time.monotonic()-start,hashes={n:sha(out/n) for n in ['transferred.ply','guarded.ply','residual.npz','moving_forearm_clay_native.png','partition_validation.json']}))
    print(frame,'completed',validation,fit,'final triangles',len(triangles)-nt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True)
    p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_forearm_anchor_residual'))
    a=p.parse_args();prepare(a.root,a.frame)
