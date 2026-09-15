"""Opt-in rendering of measured-supported deeper intersections, not inpainting.

Only pixels with no admitted source in the completed strict renderer are queried.
Search actual deeper intersections of the SAME mesh. Require coherent measured
depth support from >=2 train cameras and a measured-valid hard RGB footprint.
No synthesized pixels, new triangles, camera changes, averaging or target RGB.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,project,apply_response,display,ROOT as COLOR
from study_measured_source_visibility import ROOT as PARENT,MeasuredGate
from review_full_block_transfer import ROOT as DEPTH_ROOT
from study_confidence_depth_prior import load_real
from study_jaw_depth_footprint import train_reference_votes
from review_jaw_repair_transfer import verified_image
from diffusion_mesh_repair import scene_for
from native_texture_footprint import snap_centers,relevant_tap,sample_native
from hard_surface_texture import gather_hard_rgb
from view_consistent_source_quality import quality
from temporal_texture_view_prior import angle_weights
from render_smooth_temporal_mesh_video import verify_request,load_sources

ROOT=Path('/mnt/data/dec5_measured_depth_peeling')
VIEWS=['moving','H004_C005_1210SZ','K004_B005_1210DS']


def nearest_admitted(ray_ids,depth,admitted):
    ray_ids,depth,admitted=map(np.asarray,(ray_ids,depth,admitted))
    if ray_ids.shape!=depth.shape or ray_ids.shape!=admitted.shape or ray_ids.ndim!=1:
        raise ValueError('Expected matched flat intersection arrays')
    eligible=np.flatnonzero(admitted&np.isfinite(depth)&(depth>0))
    order=eligible[np.lexsort((depth[eligible],ray_ids[eligible]))]
    if not len(order):return order
    return order[np.r_[True,ray_ids[order][1:]!=ray_ids[order][:-1]]]


def run(view):
    start=time.monotonic();frame='000995';arm='production'
    base=PARENT/frame/arm/view;_,previous=verified_image(base,frame)
    q=deepcopy(verify_request(base));entry=q['inventory'][0];assert entry['frame_id']==frame
    assert sha(entry['mesh'])==entry['mesh_sha256']
    rows,depths,receipt=load_real(DEPTH_ROOT,frame)
    assert receipt==q['measured_source_visibility']['depth_receipt']
    q.update(depth_peeling=dict(parent=str(base),parent_request_sha256=sha(base/'request.json'),
        parent_complete_sha256=sha(base/'frames'/frame/'complete.json'),
        minimum_coherent_depth_views=2,strictly_deeper_epsilon=1e-6,
        source_visibility='measured depth; no candidate-mesh self-occlusion for deeper layers',
        source_depth_tolerance=.0015,all_nonzero_bilinear_taps_measured_valid=True,
        only_source_id_255_pixels=True,same_mesh_intersections_only=True,
        target_rgb_read=False,geometry_uses_roi=False,mesh_changed=False))
    for name in [Path(__file__).name,'study_jaw_depth_footprint.py']:
        q['script_hashes'][name]=sha(Path(__file__).with_name(name))
    out=ROOT/frame/view;out.mkdir(parents=True,exist_ok=False)
    target=out/'frames'/frame;target.mkdir(parents=True);atomic_json(out/'request.json',q)
    old=base/'frames'/frame
    before=np.asarray(Image.open(old/'prediction_native.png')).copy();rgb=before.copy()
    chosen=np.asarray(Image.open(old/'source_ids.png')).copy();original_chosen=chosen.copy()
    depth=np.load(old/'target_depth.npz')['depth'];original_depth=depth.copy()
    labels=np.load(old/'face_source_labels.npy')
    y,x=np.nonzero((chosen==255)&(depth>0));query_pixels=np.ravel_multi_index((y,x),depth.shape)
    mesh=o3d.io.read_triangle_mesh(entry['mesh']);v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles,np.uint32)
    tv=v[t];normal=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None].clip(1e-12)
    scene=scene_for(v,t);camera=entry['camera'];pose=np.asarray(camera['transform_matrix'])
    ext=np.linalg.inv(pose@np.diag([1.,-1.,-1.,1.])).astype(np.float32)
    intrinsic=np.array([[camera['fl_x'],0,camera['cx']],[0,camera['fl_y'],camera['cy']],[0,0,1]],np.float32)
    rays=scene.create_rays_pinhole(o3d.core.Tensor(intrinsic),o3d.core.Tensor(ext),camera['w'],camera['h']).numpy()[y,x]
    intersections={k:value.numpy() for k,value in scene.list_intersections(o3d.core.Tensor(rays),nthreads=2).items()}
    rid=intersections['ray_ids'].astype(int);distance=intersections['t_hit']
    farther=np.isfinite(distance)&(distance>depth[y[rid],x[rid]]+1e-6)
    rid=rid[farther];distance=distance[farther];faces=intersections['primitive_ids'][farther].astype(int)
    bary=intersections['primitive_uvs'][farther]
    weights=np.column_stack([1-bary.sum(1),bary]);points=(tv[faces]*weights[:,:,None]).sum(1)
    atomic_json(out/'progress.json',dict(stage='measured_depth_support',query_rays=len(x),deeper_intersections=len(points)))
    votes,refs=train_reference_votes(points,rows,depths)
    good=votes>=2
    manifest=q['source_rows'][0];assert Path(manifest['source_dataset']).name==frame
    images=load_sources(rows,manifest)
    profile=torch.as_tensor(np.load(COLOR/'parameters.npz')['log_gain'],device='cuda')
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    measured=MeasuredGate(torch.as_tensor(np.stack(depths),device='cuda'))
    centers=np.array([r['transform_matrix'] for r in rows],np.float32)[:,:3,3]
    angle=angle_weights(rows,camera,q['recipe']['target_angle_sigma_degrees'])[0]
    source=np.full(len(points),-1,np.int16);colors=np.zeros((len(points),3),np.uint8)
    ids=np.flatnonzero(good)
    with torch.inference_mode():
        for start_i in range(0,len(ids),30000):
            j=ids[start_i:start_i+30000];p=points[j]
            uv,z=project(p,rows);uv=snap_centers(torch.as_tensor(uv[:,None],device='cuda'));z=torch.as_tensor(z,device='cuda')
            valid=measured(uv,z)
            valid&=(uv[:,0,:,0]>2)&(uv[:,0,:,0]<1917)&(uv[:,0,:,1]>2)&(uv[:,0,:,1]<1077)
            for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
                taps=uv.floor()+uv.new_tensor([dx,dy])
                valid&=measured(taps,z)|~relevant_tap(uv,dx,dy)
            direction=centers[:,None]-p;length=np.linalg.norm(direction,axis=-1);direction/=length[...,None]
            weight=quality((direction*normal[faces[j]]).sum(-1),length,'incidence2')*angle[:,None]
            observed=apply_response(sample_native(images,uv),profile)[:,:,0].clamp_min(0)
            value,which,_=gather_hard_rgb(observed,torch.as_tensor(weight,device='cuda')*valid,
                torch.as_tensor(labels[faces[j]],device='cuda'))
            source[j]=which.cpu().numpy();colors[j]=np.rint(display(value.T.cpu().numpy(),exposure)*255).clip(0,255).astype(np.uint8)
    selected=nearest_admitted(rid,distance,good&(source>=0));sy,sx=y[rid[selected]],x[rid[selected]]
    rgb[sy,sx]=colors[selected];chosen[sy,sx]=source[selected];depth[sy,sx]=distance[selected]
    changed=np.zeros(depth.shape,bool);changed[sy,sx]=True
    assert (original_chosen[changed]==255).all()
    np.testing.assert_array_equal(rgb[~changed],before[~changed])
    np.testing.assert_array_equal(depth[~changed],original_depth[~changed])
    assert (depth[changed]>original_depth[changed]+1e-6).all()
    Image.fromarray(rgb).save(target/'prediction_native.png');Image.fromarray(np.rot90(rgb)).save(target/'frame.png')
    Image.fromarray(chosen).save(target/'source_ids.png');np.savez_compressed(target/'target_depth.npz',depth=depth)
    np.save(target/'face_source_labels.npy',labels)
    np.savez_compressed(target/'peeling_evidence.npz',query_pixels=query_pixels,rays=rays,
        ray_ids=rid,distances=distance,faces=faces,barycentric=bary,points=points,
        coherent_votes=votes,references=refs,admitted_sources=source,selected_intersections=selected,
        filled_pixels=np.ravel_multi_index((sy,sx),depth.shape))
    result=deepcopy(previous);result.update(render_sha256=sha(target/'frame.png'),
        elapsed_seconds=time.monotonic()-start,rgb_coverage=float(rgb.any(2).mean()),visual_status='pending',
        depth_peeling=dict(query_rays=len(x),deeper_intersections=len(points),
            coherent_candidates=int(good.sum()),filled_pixels=len(selected),
            unchanged_outside_filled_pixels=True,all_fills_existing_deeper_mesh=True,
            mesh_changed=False,uses_inpainting=False,measured_gate=measured.summary()))
    atomic_json(target/'result.json',result)
    atomic_json(target/'complete.json',dict(request_sha256=sha(out/'request.json'),
        hashes={p.name:sha(p) for p in target.iterdir() if p.is_file() and p.name!='complete.json'}))
    atomic_json(out/'progress.json',dict(stage='complete',filled_pixels=len(selected)))
    print(view,result['depth_peeling'],'seconds',result['elapsed_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--view',required=True,choices=VIEWS)
    torch.set_num_threads(2);run(p.parse_args().view)
