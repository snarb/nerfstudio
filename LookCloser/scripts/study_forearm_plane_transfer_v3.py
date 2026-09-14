"""V3: fixed renderer-footprint semantic availability; all V2 geometry rules.

Own frozen root, immutable prior roots. No source/default changes.
"""
from __future__ import annotations
import argparse
import shutil
from copy import deepcopy
from pathlib import Path
import numpy as np
import cv2
import study_forearm_plane_transfer_v2 as v2
from joint_temporal_texture import project,sha,read,atomic_json

OUT=Path('/mnt/data/dec5_forearm_plane_transfer_v3')

def footprint_domain(row,points,inside):
    # Exactly the raw renderer's source projection convention (including -.5).
    uv,z=project(points,[row]);uv=uv[0];z=z[0]
    return inside&(z>0)&(uv[:,0]>2)&(uv[:,0]<row['w']-3)&(uv[:,1]>2)&(uv[:,1]<row['h']-3)

def configure():
    v2.OUT=OUT;v2.configure();v2.semantic_domain=footprint_domain

def freeze():
    v2.freeze(dict(study_version=3,rule='V3: semantic evidence only in renderer footprint domain; all V2 geometry rules unchanged',
        semantic_available_domain='renderer project(): z>0, u>2, u<w-3, v>2, v<h-3; elsewhere unknown',
        measured_free_space_veto_still_uses_every_in_frame_pixel=True,skin_masks_widened=False,
        source_v2_protocol_sha256=sha('/mnt/data/dec5_forearm_plane_transfer_v2/protocol.json'),
        v3_script_sha256_at_freeze=sha(__file__),renderer_projection_sha256=sha(Path(__file__).with_name('joint_temporal_texture.py'))))

def safeguards(frame):
    """Distinguish excluded border evidence from actual semantic spill."""
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    out=OUT/frame;spec=read(out/'input.json');data=np.load(out/'diagnostic.npz');masks=v2.v1.masks(frame)
    mesh=o3d.io.read_triangle_mesh(str(out/'plane_clipped/mesh.ply'));scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles));records=[]
    for row in spec['frames']:
        name=row['physical_camera'];old=data[name+'_mesh'];new=v2.v1.raycast_integer(scene,row)
        changed=(new>0)&((old==0)|(new<old-.001));y,x=np.nonzero(changed)
        points=v2.v1.unproject(row,x,y,new[y,x]);available=footprint_domain(row,points,np.ones(len(x),bool))
        outside=~masks[name][y,x]
        records.append(dict(camera=name,changed_pixels=len(x),outside_mask_in_available_domain=int((outside&available).sum()),
            outside_mask_in_unknown_border=int((outside&~available).sum())))
    atomic_json(out/'safeguards.json',dict(rows=records,source_masks_not_widened=True,diagnostic_only=True))
    print(records,flush=True)

def render(frame):
    """Reuse identical V2 work only after matching every render request input."""
    import torch
    from render_smooth_temporal_mesh_video import render_one
    torch.set_num_threads(2);out=OUT/frame;spec=read(out/'input.json');base=read(v2.v1.PARENT/'request.json')
    assert sha(v2.v1.ROOT/'parameters.npz')==read(v2.v1.ROOT/'fit_result.json')['parameters_sha256']
    record=next(r for r in base['inventory'] if r['frame_id']==frame);source=next(r for r in base['source_rows'] if Path(r['source_dataset']).name==frame)
    for view,camera in v2.v1.views(frame).items():
        for variant in ['baseline','plane_clipped']:
            mesh=Path(spec['mesh']) if variant=='baseline' else out/variant/'mesh.ply';dest=out/variant/('render_'+view)
            if dest.exists():raise ValueError('Keep existing v3 render/reuse')
            dest.mkdir(parents=True);(dest/'frames').mkdir()
            row=deepcopy(record);row.update(mesh=str(mesh),mesh_sha256=sha(mesh),camera=camera)
            src=Path('/mnt/data/dec5_forearm_plane_transfer_v2')/frame/variant/('render_'+view)
            previous=read(src/'request.json');oldrow=previous['inventory'][0]
            assert previous['renderer_sha256']==sha(Path(__file__).with_name('render_smooth_temporal_mesh_video.py'))
            request=dict(previous,inventory=[row],source_rows=[source],protocol_sha256=sha(OUT/'protocol.json'))
            atomic_json(dest/'request.json',request)
            same=all(row[k]==oldrow[k] for k in row if k!='mesh') and previous['source_rows']==[source]
            same&=previous['profiles_sha256']==spec['profiles_sha256'] and previous['exposure_sha256']==spec['exposure_sha256']
            if same:
                receipt=read(src/'frames'/frame/'complete.json')
                for filename,digest in receipt['hashes'].items():assert sha(src/'frames'/frame/filename)==digest
                for path in (src/'frames').rglob('*'):
                    target=dest/'frames'/path.relative_to(src/'frames')
                    if path.is_dir():target.mkdir(exist_ok=True)
                    else:shutil.copyfile(path,target)
                atomic_json(dest/'reuse.json',dict(source=str(src),source_request_sha256=sha(src/'request.json'),
                    source_complete_sha256=sha(src/'frames'/frame/'complete.json'),all_render_input_values_equal_except_mesh_path=True,
                    mesh_bytes_equal=True,source_hashes_passed=True,new_gpu_render=False))
                print('reused identical',frame,view,variant,flush=True)
            else:
                render_one(dest,row,source);print('new render',frame,view,variant,flush=True)

def audit(frame):
    v2.v1.audit(frame)
    out=OUT/frame;receipt=read(out/'audit.json');reused=[];fresh=[]
    protocol=read(OUT/'protocol.json')
    assert sha('/mnt/data/dec5_forearm_plane_transfer_v2/protocol.json')==protocol['source_v2_protocol_sha256']
    for path,digest in protocol['v1_provenance'][frame].items():assert sha(path)==digest
    for view in v2.v1.views(frame):
        for variant in ['baseline','plane_clipped']:
            dest=out/variant/('render_'+view);request=read(dest/'request.json');row=request['inventory'][0]
            assert request['renderer_sha256']==sha(Path(__file__).with_name('render_smooth_temporal_mesh_video.py'))
            if (dest/'reuse.json').exists():
                reuse=read(dest/'reuse.json');src=Path(reuse['source']);previous=read(src/'request.json');oldrow=previous['inventory'][0]
                assert sha(src/'request.json')==reuse['source_request_sha256']
                assert sha(src/'frames'/frame/'complete.json')==reuse['source_complete_sha256']
                assert sha(row['mesh'])==sha(oldrow['mesh'])==row['mesh_sha256']
                assert all(row[k]==oldrow[k] for k in row if k!='mesh')
                assert previous['source_rows']==request['source_rows']
                assert sha(dest/'frames'/frame/'frame.png')==sha(src/'frames'/frame/'frame.png')
                reused.append(str(dest))
            else:fresh.append(read(dest/'frames'/frame/'result.json'))
    parameters=sha(v2.v1.ROOT/'parameters.npz');assert parameters==read(v2.v1.ROOT/'fit_result.json')['parameters_sha256']
    receipt.update(reused_render_count=len(reused),new_gpu_render_count=len(fresh),new_gpu_render_seconds=sum(r['elapsed_seconds'] for r in fresh),
        reused_render_requests_verified=True,parameters_sha256=parameters,parameters_match_original_fit_receipt=True,
        safeguards=read(out/'safeguards.json'))
    atomic_json(out/'audit.json',receipt)

def summarize():
    from PIL import Image,ImageDraw
    v2.summarize();panel=Image.new('RGB',(1720,1455));draw=ImageDraw.Draw(panel)
    for j,frame in enumerate(v2.FRAMES):
        out=OUT/frame;previous=Path('/mnt/data/dec5_forearm_plane_transfer_v2')/frame
        paths=[out/'rgb'/(v2.v1.NAMES[1]+'.png'),out/'baseline/render_train_H_A/frames'/frame/'frame.png',
            previous/'plane_clipped/render_train_H_A/frames'/frame/'frame.png',out/'plane_clipped/render_train_H_A/frames'/frame/'frame.png']
        for i,(path,label) in enumerate(zip(paths,['Train reference','Original TSDF','V2','V3'])):
            im=Image.open(path)
            if i==0:im=Image.fromarray(np.rot90(np.array(im)))
            panel.paste(im.crop((0,1500,430,1920)),(430*i,485*j+30));draw.text((430*i+4,485*j+8),frame+' '+label,fill='white')
    path=OUT/'v2_v3_three_time_H_A_native.png';panel.save(path)
    summary=read(OUT/'summary.json');summary.update(native_v2_comparison_sha256=sha(path),
        new_gpu_render_count=sum(r['new_gpu_render_count'] for r in summary['audits']),
        reused_render_count=sum(r['reused_render_count'] for r in summary['audits']),
        new_gpu_render_seconds=sum(r['new_gpu_render_seconds'] for r in summary['audits']),
        seam_diagnosis=read(OUT/'001037/seam_diagnosis.json'))
    atomic_json(OUT/'summary.json',summary)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','analyze','clip_plane','geometry_review','semantic_review','safeguards','render','evaluation_inputs','score','audit','summarize','seam_diagnosis'])
    parser.add_argument('--frame',choices=v2.FRAMES);parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();OUT=args.output
    cv2.setNumThreads(2);configure()
    if args.command=='freeze':freeze()
    elif args.command=='safeguards':safeguards(args.frame)
    elif args.command=='render':render(args.frame)
    elif args.command=='audit':audit(args.frame)
    elif args.command=='summarize':summarize()
    elif args.command in ['analyze','clip_plane','evaluation_inputs','seam_diagnosis']:getattr(v2,args.command)(args.frame)
    else:getattr(v2.v1,args.command)(args.frame)
