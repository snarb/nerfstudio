"""Render learned RGB on a full subject orbit and a face arc."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml
from blur_runtime import write,sha
from nerfstudio.cameras.cameras import Cameras,CameraType


def look_at(eye,target):
    z=eye-target;z/=np.linalg.norm(z)
    x=np.cross([0.,0.,1.],z);x/=np.linalg.norm(x)
    y=np.cross(z,x)
    return np.column_stack([x,y,z,eye])


def face_center(rows,rois):
    matrices=[];targets=[]
    for row in rows:
        name=Path(row['file_path']).name
        if row['camera_id'] not in [11,97,151]:continue
        x0,y0,x1,y1=rois[name]['face'];x=(x0+x1)/2;y=(y0+y1)/2
        pose=np.array(row['transform_matrix'])
        direction=pose[:3,:3]@np.array([(x-row['cx'])/row['fl_x'],-(y-row['cy'])/row['fl_y'],-1.])
        direction/=np.linalg.norm(direction);a=np.eye(3)-np.outer(direction,direction)
        matrices.append(a);targets.append(a@pose[:3,3])
    return np.linalg.solve(np.sum(matrices,axis=0),np.sum(targets,axis=0))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('selection',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--orbit-frames',type=int,default=48);p.add_argument('--face-frames',type=int,default=32)
    args=p.parse_args();torch.set_num_threads(2)
    selected=json.loads(args.selection.read_text());cfg=yaml.load(Path(selected['config']).read_text(),Loader=yaml.Loader)
    state=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    pipe=cfg.pipeline.setup(device='cuda');pipe.load_pipeline(state['pipeline'],state['step']);pipe.eval();del state
    if selected.get('occupancy_guard'):
        from luster_render_guard import apply_guard
        apply_guard(pipe,selected['occupancy_guard'])
    data=cfg.pipeline.datamanager.dataparser.data
    meta=json.loads((data/'transforms.json').read_text());rows=meta['frames'];rois=json.loads((data/'rois.json').read_text())
    bounds=np.array(meta['blur_aabb']);body=bounds.mean(0);face=face_center(rows,rois)
    face_spans=[]
    for row in rows:
        if row['camera_id'] in [11,97,151]:
            box=rois[Path(row['file_path']).name]['face']
            distance=np.linalg.norm(np.array(row['transform_matrix'])[:3,3]-face)
            face_spans.append((box[3]-box[1])*distance/row['fl_y'])
    face_span=float(np.median(face_spans))*2.2
    reference=next(r for r in rows if r['camera_id']==150)
    ref_eye=np.array(reference['transform_matrix'])[:3,3]
    head_radius=np.linalg.norm(ref_eye[:2]-face[:2]);heading=np.arctan2(ref_eye[1]-face[1],ref_eye[0]-face[0])
    full_poses=np.array([r['transform_matrix'] for r in rows if r['camera_id']<=48])
    body_radius=float(np.median(np.linalg.norm(full_poses[:,:2,3]-body[:2],axis=1)))
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for kind,count in [('orbit',args.orbit_frames),('face',args.face_frames)]:
        out=args.output/kind;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        target=body if kind=='orbit' else face
        angles=heading+ (np.arange(count)*2*np.pi/count if kind=='orbit' else np.linspace(-np.pi/6,np.pi/6,count))
        radius=body_radius if kind=='orbit' else head_radius
        height=960;width=704
        z=body[2]+.12 if kind=='orbit' else ref_eye[2]
        vertical_span=float((bounds[1,2]-bounds[0,2])*1.2) if kind=='orbit' else face_span
        distance=np.hypot(radius,z-target[2]);focal=height*distance/vertical_span
        poses=np.array([look_at(np.array([target[0]+radius*np.cos(a),target[1]+radius*np.sin(a),z]),target) for a in angles])
        cameras=Cameras(camera_to_worlds=torch.tensor(poses,dtype=torch.float32),fx=focal,fy=focal,cx=width/2,cy=height/2,width=width,height=height,camera_type=CameraType.PERSPECTIVE).to('cuda')
        write(out/'path.json',dict(poses=poses.tolist(),target=target.tolist(),width=width,height=height,focal=focal,
                                   checkpoint=selected['checkpoint'],checkpoint_sha256=sha(Path(selected['checkpoint'])),
                                   occupancy_guard=selected.get('occupancy_guard')))
        start=time.monotonic();thumbs=[]
        for i in range(count):
            prediction=pipe.model.get_outputs_for_camera_ray_bundle(cameras[i:i+1].generate_rays(0))['rgb']
            if not torch.isfinite(prediction).all():raise FloatingPointError(f'{kind}/{i} has nonfinite pixels')
            im=Image.fromarray(np.rint(prediction.cpu().numpy().clip(0,1)*255).astype('uint8'))
            im.save(out/f'frame_{i:03d}.png')
            thumb=im.copy();thumb.thumbnail((176,240));ImageDraw.Draw(thumb).text((5,5),str(i),fill='white');thumbs.append(thumb)
            write(args.output/'progress.json',dict(path=kind,frame=i+1,frames=count,seconds=time.monotonic()-start))
        contact=Image.new('RGB',(8*176,((count+7)//8)*240))
        for i,im in enumerate(thumbs):contact.paste(im,((i%8)*176,(i//8)*240))
        contact.save(out/'contact.jpg')
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-framerate','8','-i',str(out/'frame_%03d.png'),'-c:v','libx264','-crf','16','-threads','2','-pix_fmt','yuv420p',str(out/'learned_rgb.mp4')],check=True)
        subprocess.run([ffmpeg,'-v','error','-i',str(out/'learned_rgb.mp4'),'-f','null','-'],check=True)
        write(out/'complete.json',dict(frames=count,finite=True,decoded=True,seconds=time.monotonic()-start,video_sha256=sha(out/'learned_rgb.mp4')))
    write(args.output/'complete.json',dict(selection=selected,paths=['orbit','face']))


if __name__=='__main__':main()
