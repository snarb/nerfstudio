"""Matched geometry-only comparison of repeated native-depth jaw controls."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json, geometry_paths
from study_jaw_boundary_notches import PARENT, PHASE
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth


def normalization_matrix(metadata):
    transform=np.eye(4)
    transform[:3,:]=np.asarray(metadata['dataparser_transform'])*metadata['dataparser_scale']
    return transform


def run(root, frame):
    control = root/'controls'/frame; completed = read(control/'complete.json')
    for p,h in completed['hashes'].items():
        if sha(control/p) != h: raise ValueError('Changed completed control')
    source = next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id'] == frame)
    phase = next(r for r in read(PHASE/'request.json')['inventory'] if r['frame_id'] == frame)
    original, metadata = geometry_paths(frame)
    base_meta = read(metadata)
    # Camera-order floating-point centering can differ by ~1e-7 before scale.
    # Apply the explicit coordinate transform rather than relax an equality gate.
    mappings={arm:normalization_matrix(base_meta)@np.linalg.inv(normalization_matrix(read(control/arm/'mesh.json')))
              for arm in ['fuse-original','fuse-full-block']}
    variants = dict(published_original=original, repeated_original=control/'fuse-original/mesh.ply',
                    repeated_full_block=control/'fuse-full-block/mesh.ply', published_boundary=Path(source['mesh']))
    spot = next(r for r in read('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')['selected_components'] if r['frame_id']==frame)
    x0,y0,x1,y1 = spot['bbox_inclusive']; crop = (x0-65,y0-65,x1+66,y1+66)
    # Same broad native region in every view, no dynamic recentering.
    face = (290,640,790,1180)
    out=root/'control_review'/frame;out.mkdir(parents=True,exist_ok=True)
    records=[];hashes={}; meshes={};scenes={}
    for name,path in variants.items():
        meshes[name]=o3d.io.read_triangle_mesh(str(path))
        if name in ['repeated_original','repeated_full_block']:
            meshes[name].transform(mappings['fuse-original' if name=='repeated_original' else 'fuse-full-block'])
        meshes[name].compute_triangle_normals()
        scenes[name]=scene_for(np.asarray(meshes[name].vertices),np.asarray(meshes[name].triangles))
    for label, camera in [('old_moving',source['camera']),('phase_moving',phase['camera'])]:
        panel=Image.new('RGB',(500*4,570));draw=ImageDraw.Draw(panel)
        spotpanel=Image.new('RGB',((crop[2]-crop[0])*4,crop[3]-crop[1]+30));sd=ImageDraw.Draw(spotpanel)
        for index,(name,path) in enumerate(variants.items()):
            depth,ids,_=camera_depth(scenes[name],camera);valid=np.isfinite(depth)
            lighting=np.abs(np.asarray(meshes[name].triangle_normals)@np.array([.3,.4,.866]))
            rgb=np.zeros((*ids.shape,3),np.uint8);rgb[valid]=(60+170*lighting[ids[valid],None]).astype(np.uint8)
            portrait=Image.fromarray(np.rot90(rgb));portrait.save(out/f'{label}_{name}.png')
            hashes[f'{label}_{name}.png']=sha(out/f'{label}_{name}.png')
            panel.paste(portrait.crop(face),(500*index,30));draw.text((500*index+5,8),name,fill='white')
            spotpanel.paste(portrait.crop(crop),((crop[2]-crop[0])*index,30));sd.text(((crop[2]-crop[0])*index+3,5),name,fill='white')
            rec=dict(camera=label,variant=name,mesh_sha256=sha(path),vertices=len(meshes[name].vertices),triangles=len(meshes[name].triangles))
            if label=='old_moving': rec['selected_spot_misses']=int((~np.rot90(valid)[y0:y1+1,x0:x1+1]).sum())
            records.append(rec)
        panel.save(out/f'{label}_face_native.png');spotpanel.save(out/f'{label}_spot_native.png')
        for p in [out/f'{label}_face_native.png',out/f'{label}_spot_native.png']: hashes[p.name]=sha(p)
    atomic_json(out/'result.json',dict(script_sha256=sha(__file__),control_complete_sha256=sha(control/'complete.json'),
        source_request_sha256=sha(PARENT/'request.json'),phase_request_sha256=sha(PHASE/'request.json'),
        records=records,output_hashes=hashes,visual_status='pending',production_changed=False,
        render_only_normalization_to_published={k:v.tolist() for k,v in mappings.items()},
        note='Clay raycast miss counts only; no full-frame quality metrics, no inferred acceptance'))
    print(frame,records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_jaw_measured_depth'))
    p.add_argument('--frame',required=True,choices=['001193','001195']);args=p.parse_args();run(args.root,args.frame)
