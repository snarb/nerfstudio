"""Matched same-camera RGB/clay views for the001033 depth-stage experiment."""
from pathlib import Path
from copy import deepcopy
import argparse
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from review_temporal_full_block_control import transfer_mesh_gauge
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth

BASE=Path('/mnt/data/dec5_elevated_camera_dynamic_150')
CONTROL=Path('/mnt/data/dec5_forearm_depth_control_001033')
FRAME='001033'
VARIANTS=['fuse-original','fuse-full-block']


def prepare(control):
    done=read(control/'complete.json')
    if done['request_sha256']!=sha(control/'request.json'):raise ValueError('Changed reconstruction request')
    for name,digest in done['hashes'].items():
        if sha(control/name)!=digest:raise ValueError('Changed fusion output')
    parent=read(BASE/'request.json');record=next(r for r in parent['inventory'] if r['frame_id']==FRAME)
    source=next(r for r in parent['source_rows'] if Path(r['source_dataset']).name==FRAME)
    raw=read(control/'request.json')
    if raw['source_images']!={r['file_path']:r['sha256'] for r in source['source_images']}:raise ValueError('Changed historical EXR inputs')
    original_meta=read(record['metadata'])
    for variant in VARIANTS:
        root=control/variant;metadata=read(root/'mesh.json');mesh=o3d.io.read_triangle_mesh(str(root/'mesh.ply'))
        v=np.asarray(mesh.vertices).copy();aligned=transfer_mesh_gauge(v,metadata,original_meta)
        if not np.allclose(transfer_mesh_gauge(aligned,original_meta,metadata),v,atol=1e-12,rtol=0):raise ValueError('Gauge roundtrip mismatch')
        mesh.vertices=o3d.utility.Vector3dVector(aligned);mp=root/'mesh_video_gauge.ply';o3d.io.write_triangle_mesh(str(mp),mesh)
        m=deepcopy(metadata);m.update(dataparser_transform=original_meta['dataparser_transform'],dataparser_scale=original_meta['dataparser_scale'])
        atomic_json(mp.with_suffix('.json'),m)
        output=control/(variant+'_render');output.mkdir(exist_ok=True);(output/'frames').mkdir(exist_ok=True)
        request=deepcopy(parent);row=deepcopy(record)
        row.update(mesh=str(mp),mesh_sha256=sha(mp),metadata=str(mp.with_suffix('.json')),metadata_sha256=sha(mp.with_suffix('.json')))
        request.update(ordered_frame_ids=[FRAME],inventory=[row],source_rows=[source],partial_diagnostic_only=True)
        request['recipe'].update(frames=1,geometry_control_variant=variant,head_repair_applied=False)
        request['fusion_control']=dict(request_sha256=sha(control/'request.json'),parent_request_sha256=sha(BASE/'request.json'),
            new_same_depth_matched_pair=True,historical_jpeg_equivalence_not_available=True)
        request['script_hashes'][Path(__file__).name]=sha(__file__)
        if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Immutable render mismatch')
        atomic_json(output/'request.json',request)


def render(control):
    import render_smooth_temporal_mesh_video as renderer
    from temporal_texture_view_prior import install
    from wide_dynamic_camera_flight import install_source_masks
    renderer.torch.set_num_threads(4);install(renderer);install_source_masks(renderer)
    for variant in VARIANTS:renderer.render(control/(variant+'_render'),[FRAME])


def compare(control):
    panel=Image.new('RGB',(1620,984));draw=ImageDraw.Draw(panel);evidence=[]
    for i,(label,root) in enumerate([('Published',BASE)]+[(v,control/(v+'_render')) for v in VARIANTS]):
        path=root/'frames'/FRAME/'frame.png';rgb=Image.open(path)
        panel.paste(rgb.resize((540,960)),(540*i,24));draw.text((540*i+3,5),label,fill='white')
        evidence.append(dict(path=str(path),sha256=sha(path)))
    panel.save(control/'rgb_comparison.png')
    panel=Image.new('RGB',(1620,984));draw=ImageDraw.Draw(panel)
    counts={}
    for i,(label,root) in enumerate([('Published',BASE)]+[(v,control/(v+'_render')) for v in VARIANTS]):
        row=next(r for r in read(root/'request.json')['inventory'] if r['frame_id']==FRAME)
        mesh=o3d.io.read_triangle_mesh(row['mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
        d,ids,_=camera_depth(scene_for(v,t),row['camera']);hit=np.isfinite(d)
        normal=np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]]);normal/=np.linalg.norm(normal,axis=1)[:,None].clip(1e-12)
        shade=(.2+.8*np.abs(normal@np.array(row['camera']['transform_matrix'])[:3,2]))*230
        clay=np.zeros((*d.shape,3),np.uint8);clay[hit]=shade[ids[hit],None].astype(np.uint8)
        portrait=Image.fromarray(np.rot90(clay));portrait.save(control/(label+'_clay.png'))
        panel.paste(portrait.resize((540,960)),(i*540,24));draw.text((i*540+3,5),label,fill='white')
        counts[label]=dict(triangles=len(t),vertices=len(v))
    panel.save(control/'clay_comparison.png')
    atomic_json(control/'comparison.json',dict(evidence=evidence,counts=counts,ground_truth_for_novel_view=False,visual_status='pending'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render','compare']);p.add_argument('--control',type=Path,default=CONTROL)
    a=p.parse_args();globals()[a.action](a.control)
