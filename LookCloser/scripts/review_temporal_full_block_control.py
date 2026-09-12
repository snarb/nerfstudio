"""Matched hard-texture renders for the real-depth block-activation experiment."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json,ROOT,cameras,project,exr,display
from diagnose_temporal_mesh_shelf import BASE,OUTPUT,BOX
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth


def transfer_mesh_gauge(vertices, source, target):
    """Same raw calibration coordinates; transform points, never scale rotations."""
    a,b=np.eye(4),np.eye(4)
    a[:3]=source['dataparser_transform'];b[:3]=target['dataparser_transform']
    xform=b@np.linalg.inv(a)
    unscaled=np.asarray(vertices)/source['dataparser_scale']
    return (unscaled@xform[:3,:3].T+xform[:3,3])*target['dataparser_scale']


def prepare(control,base,frame):
    complete=read(control/'complete.json')
    if complete['request_sha256']!=sha(control/'request.json'):raise ValueError('Control provenance mismatch')
    for p,h in complete['hashes'].items():
        if sha(control/p)!=h:raise ValueError('Control artifact changed')
    historical=Path('/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/frames')/frame/'staging_manifest.json'
    current=control/'staged63/staging_manifest.json'
    old={r['physical_camera']:r for r in read(historical)['conversion_rows']}
    fresh={r['physical_camera']:r for r in read(current)['conversion_rows']}
    if set(old)!=set(fresh):raise ValueError('Geometry ingest camera inventory changed')
    for camera,row in fresh.items():
        if old[camera]['sha256']!=row['sha256'] or sha(row['output'])!=row['sha256'] or old[camera]['exposure_gain']!=row['exposure_gain']:
            raise ValueError('Geometry ingest does not reproduce historical pixels')
    atomic_json(control/'ingest_equivalence.json',{'historical_manifest_sha256':sha(historical),'current_manifest_sha256':sha(current),
                'exact_matching_jpeg_count':len(fresh),'exact_matching_gain_count':len(fresh),'uses_eval_rgb_for_geometry':False})
    parent=read(base/'request.json');record=next(r for r in parent['inventory'] if r['frame_id']==frame)
    original_metadata=read(record['metadata'])
    for variant in ['fuse-original','fuse-full-block']:
        meta=read(control/variant/'mesh.json')
        mesh=o3d.io.read_triangle_mesh(str(control/variant/'mesh.ply'))
        vertices=np.asarray(mesh.vertices).copy();aligned=transfer_mesh_gauge(vertices,meta,original_metadata)
        # Round-trip through the two recorded gauges must recover every vertex.
        if not np.allclose(transfer_mesh_gauge(aligned,original_metadata,meta),vertices,rtol=0,atol=1e-12):
            raise ValueError('Geometry gauge transfer failed its point round-trip')
        mesh.vertices=o3d.utility.Vector3dVector(aligned)
        aligned_path=control/variant/'mesh_video_gauge.ply';o3d.io.write_triangle_mesh(str(aligned_path),mesh)
        aligned_meta=deepcopy(meta)
        aligned_meta.update(dataparser_transform=original_metadata['dataparser_transform'],dataparser_scale=original_metadata['dataparser_scale'],
                            output=str(aligned_path),output_sha256=sha(aligned_path),
                            coordinate_transfer={'input_mesh_sha256':sha(control/variant/'mesh.ply'),'original_metadata_sha256':sha(control/variant/'mesh.json'),
                            'target_metadata_sha256':record['metadata_sha256'],'max_vertex_displacement':float(np.linalg.norm(aligned-vertices,axis=1).max()),
                            'physical_shape_changed':False,'same_raw_calibration_coordinates':True})
        atomic_json(aligned_path.with_suffix('.json'),aligned_meta)
        output=control/(variant+'_render');output.mkdir(exist_ok=True);(output/'frames').mkdir(exist_ok=True)
        request=deepcopy(parent);request['ordered_frame_ids']=[frame]
        request['source_rows']=[r for r in parent['source_rows'] if Path(r['source_dataset']).name==frame]
        row=deepcopy(record);row.update(mesh=str(aligned_path),mesh_sha256=sha(aligned_path),
                                      metadata=str(aligned_path.with_suffix('.json')),metadata_sha256=sha(aligned_path.with_suffix('.json')))
        request['inventory']=[row]
        request['recipe'].update(frames=1,geometry_control_variant=variant,parent_request_sha256=sha(base/'request.json'))
        request['script_hashes'][Path(__file__).name]=sha(__file__)
        if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Render control is immutable')
        atomic_json(output/'request.json',request)


def render(control,frame):
    import render_smooth_temporal_mesh_video as renderer
    from temporal_texture_view_prior import install
    renderer.torch.set_num_threads(4);install(renderer)
    for variant in ['fuse-original','fuse-full-block']:renderer.render(control/(variant+'_render'),[frame])


def compare(control,base,frame):
    variants=[('Published',base),('Same-depth per-view blocks',control/'fuse-original_render'),
              ('Same-depth full-block union',control/'fuse-full-block_render')]
    evidence=[]
    for name,box,scale in [('detail',BOX,3),('lipstick_hand',(280,970,730,1420),1),('face',(300,660,900,1240),1)]:
        w,h=(box[2]-box[0])*scale,(box[3]-box[1])*scale
        panel=Image.new('RGB',(w*3,h+26));draw=ImageDraw.Draw(panel)
        for i,(title,root) in enumerate(variants):
            path=root/'frames'/frame/'frame.png';im=Image.open(path).convert('RGB').crop(box)
            panel.paste(im.resize((w,h),Image.Resampling.NEAREST),(i*w,26));draw.text((i*w+4,7),title,fill='white')
            evidence.append({'path':str(path),'sha256':sha(path)})
        panel.save(control/f'comparison_{name}.png')
    atomic_json(control/'comparison_manifest.json',{'frame':frame,'render_evidence':evidence,
                'image_quality_metrics_not_computed':True,'ground_truth_available_for_novel_view':False,
                'decision':'requires_actual_visual_review'})


def real_references(control,base,frame):
    """Project the same 3D query into six nearby real cameras, with no geometry fitting."""
    control.mkdir(parents=True,exist_ok=True)
    request=read(base/'request.json');record=next(r for r in request['inventory'] if r['frame_id']==frame)
    mesh=o3d.io.read_triangle_mesh(record['mesh']);v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    d,ids,b=camera_depth(scene_for(v,t),record['camera'])
    # Portrait point at the shelf center, converted back to native array coordinates.
    py,px=1067,390;ny,nx=px,1919-py
    tri=t[ids[ny,nx]];weight=np.r_[1-b[ny,nx].sum(),b[ny,nx]];point=(v[tri]*weight[:,None]).sum(0)
    rows,_,_=cameras(frame);centers=np.array([r['transform_matrix'] for r in rows])[:,:3,3]
    chosen=np.argsort(np.linalg.norm(centers-np.array(record['camera']['transform_matrix'])[:3,3],axis=1))[:6]
    profile=read(ROOT/'camera_profiles.json');gains=dict(zip(profile['physical_cameras'],profile['rgb_gain']))
    gain=read(ROOT/'exposure.json')['fixed_exposure_gain'];panel=Image.new('RGB',(960,688));draw=ImageDraw.Draw(panel);items=[]
    for i,c in enumerate(chosen):
        row=rows[c];uv,z=project(point[None],[row]);u,vv=uv[0,0];x,y=int(round(vv)),int(round(1919-u))
        box=(x-80,y-60,x+80,y+100)
        rgb=np.rint(display(exr(row['file_path'])*np.array(gains[row['physical_camera']]),gain)*255).clip(0,255).astype(np.uint8)
        im=Image.fromarray(np.rot90(rgb)).crop(box).resize((320,320),Image.Resampling.NEAREST)
        ox,oy=(i%3)*320,(i//3)*344;panel.paste(im,(ox,oy+24));draw.text((ox+3,oy+6),row['physical_camera'],fill='white')
        items.append({'physical_camera':row['physical_camera'],'source_sha256':sha(row['file_path']),'crop_portrait':box})
    panel.save(control/'real_train_reference_shelf.png')
    atomic_json(control/'real_train_reference_shelf.json',{'references':items,'no_eval_rgb':True,
                'purpose':'Real multi-view visual reference only; projected query is not a certified true surface'})


def depth_evidence(control,base,frame):
    """Independent measured depth footprints, not candidate-mesh self visibility."""
    import gzip
    from carve_patchmatch_mesh_free_space import free_space_evidence,train_frames
    from render_patchmatch_camera_path import normalize_frame
    record=next(r for r in read(base/'request.json')['inventory'] if r['frame_id']==frame)
    meta=read(record['metadata']);mesh=o3d.io.read_triangle_mesh(record['mesh'])
    v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    # All faces seen in the reviewed local area, including nearby genuine skin.
    _,ids,_=camera_depth(scene_for(v,t),record['camera']);ids=np.rot90(ids)
    x0,y0,x1,y1=BOX;faces=np.unique(ids[y0:y1,x0:x1]);faces=faces[faces<len(t)]
    points=v[t[faces]].mean(1);free=np.zeros(len(faces),np.uint16);near=np.zeros_like(free)
    data=control/'pipeline/depth_dataset';payload=read(data/'transforms.json');sources=[]
    for original in train_frames(payload):
        row=normalize_frame(original,payload,meta);uv,z=project(points,[row])
        path=data/original['depth_file_path']
        with gzip.open(path,'rb') as stream:depth=np.load(stream,allow_pickle=False)
        f,n=free_space_evidence(depth*meta['dataparser_scale'],uv[0,:,0],uv[0,:,1],z[0],minimum_gap=.005,near_gap=.0015)
        free+=f;near+=n;sources.append({'physical_camera':row['physical_camera'],'depth_sha256':sha(path)})
    np.savez_compressed(control/'local_real_depth_evidence.npz',face_ids=faces,centers=points,free_counts=free,near_counts=near)
    selected=read(OUTPUT/'shelf_polygon_v1/geometry_diagnostic.json')['selected_faces'];index={int(f):i for i,f in enumerate(faces)}
    details=[{'face_id':int(f),'free_views':int(free[index[f]]),'near_views':int(near[index[f]])} for f in selected]
    atomic_json(control/'local_real_depth_evidence.json',{'source_depths':sources,'triangle_centroid_evidence':details,
                'input_mesh_sha256':record['mesh_sha256'],'depth_evidence_sha256':sha(control/'local_real_depth_evidence.npz'),
                'free_minimum_gap':.005,'free_relative_minimum_gap':.01,'near_gap':.0015,'native_taps':25,
                'native_required_fraction':.8,'uses_rgb':False,'candidate_self_visibility_not_used':True,
                'missing_depth_is_unknown':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render','compare','references','evidence'])
    p.add_argument('--control',type=Path,default=OUTPUT/'full_block_control');p.add_argument('--base',type=Path,default=BASE);p.add_argument('--frame',default='000971')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.control,a.base,a.frame)
    elif a.action=='render':render(a.control,a.frame)
    elif a.action=='compare':compare(a.control,a.base,a.frame)
    elif a.action=='evidence':depth_evidence(a.control,a.base,a.frame)
    else:real_references(a.control,a.base,a.frame)
