"""Render a short learned-RGB path through the three held-out camera poses."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image, ImageDraw
import torch
from nerfstudio.cameras.camera_paths import get_interpolated_camera_path
from blur_runtime import write, sha


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);start=time.monotonic()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    pipe=state['config'].pipeline.setup(device='cuda')
    pipe.load_pipeline(state['pipeline'],state['step']);pipe.eval()
    ds=pipe.datamanager.eval_dataset
    if len(ds)!=3:raise ValueError('Review protocol requires the three held-out cameras')
    cameras=get_interpolated_camera_path(ds.cameras,steps=12,order_poses=False)
    cameras.rescale_output_resolution(.5)
    cameras=cameras.to('cuda')
    write(args.output/'path.json',dict(camera_to_worlds=cameras.camera_to_worlds.cpu().tolist(),
        fx=cameras.fx.cpu().tolist(),fy=cameras.fy.cpu().tolist(),
        cx=cameras.cx.cpu().tolist(),cy=cameras.cy.cpu().tolist(),
        width=cameras.width.cpu().tolist(),height=cameras.height.cpu().tolist()))
    contact=[]
    for index in range(len(cameras)):
        rays=cameras[index:index+1].generate_rays(0)
        out=pipe.model.get_outputs_for_camera_ray_bundle(rays)
        if not torch.isfinite(out['rgb']).all():
            raise FloatingPointError(f'Nonfinite RGB at review frame {index}')
        rgb=np.rint(out['rgb'].cpu().numpy().clip(0,1)*255).astype('uint8')
        img=Image.fromarray(rgb)
        img.save(args.output/f'frame_{index:03d}.png')
        if index%6==0 or index==len(cameras)-1:
            thumb=img.copy();thumb.thumbnail((480,270))
            ImageDraw.Draw(thumb).text((5,5),f'frame {index}',fill='white',stroke_fill='black',stroke_width=1)
            contact.append(thumb)
        print(f'frame={index+1}/{len(cameras)}',flush=True)
    sheet=Image.new('RGB',(max(i.width for i in contact),sum(i.height for i in contact)))
    y=0
    for img in contact:sheet.paste(img,(0,y));y+=img.height
    sheet.save(args.output/'contact.jpg')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','6',
                    '-i',str(args.output/'frame_%03d.png'),'-c:v','libx264','-crf','18',
                    '-pix_fmt','yuv420p',str(args.output/'learned_rgb.mp4')],check=True)
    write(args.output/'complete.json',dict(checkpoint=str(args.checkpoint),
          checkpoint_sha256=sha(args.checkpoint),step=state['step'],
          seconds=time.monotonic()-start,frames=len(cameras),render='learned RGB only',
          finite_rgb_checked=True,
          note='Interpolated views are a visual stability check; no ground-truth metrics exist for them.'))


if __name__=='__main__':main()
