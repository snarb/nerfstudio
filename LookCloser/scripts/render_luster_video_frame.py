"""Render one trained temporal model with a sequence-wide fixed camera."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import cv2
from PIL import Image
import torch
import yaml
from blur_runtime import write,sha,state_digest
from render_luster_paths import look_at
from luster_render_guard import apply_guard
from nerfstudio.cameras.cameras import Cameras,CameraType


def camera_spec(root):
    root=Path(root);path=root/'video_cameras.json'
    if path.exists():return json.loads(path.read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    meta=json.loads((root/'frames'/manifest['frames'][0]/'data/transforms.json').read_text())
    bounds=np.array(meta['blur_aabb']);rows=meta['frames'];center=bounds.mean(0)
    ref=next(r for r in rows if r['camera_id']==150);eye=np.array(ref['transform_matrix'])[:3,3]
    heading=np.arctan2(eye[1]-center[1],eye[0]-center[0])
    wide=np.array([r['transform_matrix'] for r in rows if r['camera_id']<=48])
    radius=float(np.median(np.linalg.norm(wide[:,:2,3]-center[:2],axis=1)))
    head_lows=[];head_highs=[]
    for frame in manifest['frames']:
        points=np.load(root/'frames'/frame/'data/hull.npz')['points']
        head=points[points[:,2]>bounds[1,2]-.30*(bounds[1,2]-bounds[0,2])]
        if len(head):head_lows.append(head[:,:2].min(0));head_highs.append(head[:,:2].max(0))
    head_low=np.min(head_lows,axis=0);head_high=np.max(head_highs,axis=0)
    head_padding=(head_high-head_low)*.15
    spec={}
    for kind in ['body','detail']:
        box=bounds.copy()
        if kind=='detail':
            box[0,2]=bounds[1,2]-.42*(bounds[1,2]-bounds[0,2])
            box[0,:2]=head_low-head_padding;box[1,:2]=head_high+head_padding
        target=box.mean(0)
        position=np.array([center[0]+radius*np.cos(heading),center[1]+radius*np.sin(heading),target[2]+.12])
        pose=look_at(position,target)
        corners=np.array([[x,y,z] for x in box[:,0] for y in box[:,1] for z in box[:,2]])
        view=(corners-pose[:3,3])@pose[:3,:3]
        assert (view[:,2]<0).all()
        projected=view[:,:2]/-view[:,2,None]
        width,height=1080,1920
        focal=min(.44*width/np.abs(projected[:,0]).max(),.44*height/np.abs(projected[:,1]).max())
        spec[kind]=dict(pose=pose.tolist(),width=width,height=height,fx=float(focal),fy=float(focal),cx=width/2,cy=height/2)
    write(path,dict(protocol='Fixed virtual cameras; one fresh trained model for each real time',fps=30,cameras=spec))
    return json.loads(path.read_text())


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('frame');p.add_argument('selection',type=Path)
    args=p.parse_args();torch.set_num_threads(2);spec=camera_spec(args.root)
    selected=json.loads(args.selection.read_text());cfg=yaml.load(Path(selected['config']).read_text(),Loader=yaml.Loader)
    if Path(cfg.pipeline.datamanager.dataparser.data).parent.name!=args.frame:raise ValueError('Selected model belongs to another temporal frame')
    state=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    pipe=cfg.pipeline.setup(device='cuda');pipe.load_pipeline(state['pipeline'],state['step']);pipe.eval();del state
    raw_occupancy=pipe.model.occupancy_grid.binaries.clone()
    if selected.get('occupancy_guard'):apply_guard(pipe,selected['occupancy_guard'])
    guarded_occupancy=pipe.model.occupancy_grid.binaries.clone()
    record=dict(frame=args.frame,selection=str(args.selection),checkpoint=selected['checkpoint'],checkpoint_sha256=sha(Path(selected['checkpoint'])),
                field_parameters_sha256=state_digest(pipe.model.field),cameras_sha256=sha(args.root/'video_cameras.json'),images={},raw_images={},guard_diagnostics={})
    for kind,view in spec['cameras'].items():
        out=args.root/'video_frames'/kind/f'{args.frame}.png';out.parent.mkdir(parents=True,exist_ok=True)
        camera=Cameras(camera_to_worlds=torch.tensor(view['pose'],dtype=torch.float32)[None],
              fx=view['fx'],fy=view['fy'],cx=view['cx'],cy=view['cy'],width=view['width'],height=view['height'],camera_type=CameraType.PERSPECTIVE).to('cuda')
        rays=camera.generate_rays(0)
        pipe.model.occupancy_grid.binaries.copy_(raw_occupancy)
        raw=pipe.model.get_outputs_for_camera_ray_bundle(rays)['rgb']
        if not torch.isfinite(raw).all():raise FloatingPointError('Raw temporal RGB is nonfinite')
        raw=np.rint(raw.cpu().numpy().clip(0,1)*255).astype('uint8')
        raw_path=args.root/'video_frames'/f'raw_{kind}'/f'{args.frame}.png';raw_path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(raw).save(raw_path);record['raw_images'][kind]=dict(path=str(raw_path),sha256=sha(raw_path))
        pipe.model.occupancy_grid.binaries.copy_(guarded_occupancy)
        result=pipe.model.get_outputs_for_camera_ray_bundle(rays);rgb=result['rgb']
        if not torch.isfinite(rgb).all():raise FloatingPointError('Temporal render has nonfinite RGB')
        if 'num_samples_per_ray' in result and int(result['num_samples_per_ray'].max())>=cfg.pipeline.model.max_steps_per_ray:raise ValueError('Temporal export saturates integration cap')
        prediction=np.rint(rgb.cpu().numpy().clip(0,1)*255).astype('uint8');Image.fromarray(prediction).save(out)
        count,labels,stats,_=cv2.connectedComponentsWithStats((raw.max(-1)>16).astype('uint8'))
        main=labels==(1+stats[1:,cv2.CC_STAT_AREA].argmax()) if count>1 else np.zeros(raw.shape[:2],dtype=bool)
        changed=np.abs(raw.astype('int16')-prediction).max(-1)>8
        record['guard_diagnostics'][kind]=dict(main_colored_component_pixels=int(main.sum()),changed_gt8_pixels=int((main&changed).sum()),
                changed_fraction=float((main&changed).sum()/max(1,main.sum())),protocol='Largest connected raw RGB component above16/255; diagnostic only, not ground truth')
        record['images'][kind]=dict(path=str(out),sha256=sha(out),size=[view['width'],view['height']])
    write(args.root/'video_frames/receipts'/f'{args.frame}.json',record)


if __name__=='__main__':main()
