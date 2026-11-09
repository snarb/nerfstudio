"""Original calibrated FOV, raw field/opacity, Full HD canvas; no GT masking."""
import argparse
import copy
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from common import read,atomic,sha
from inference import load_pipeline
from runtime import display_rgb


def run(root,spec_path,checkpoint,sizes,output,views,cap=10):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    pipe,spec,step=load_pipeline(root,spec_path,checkpoint,out)
    pipe.model.config.adaptive_max_frequency_level=cap
    ready=read(Path(spec['data'])/'ready.json');dimensions=read(sizes)
    dataset=pipe.datamanager.eval_dataset;receipts=[]
    with torch.inference_mode():
        for camera_name in views:
            row=next(r for r in ready['records'] if r['camera']==camera_name)
            index=next(i for i,p in enumerate(dataset.image_filenames) if Path(p).stem==camera_name)
            crop=dataset.cameras[index:index+1].to(pipe.model.device)
            camera=copy.deepcopy(crop);x0,y0,_,_=row['native_crop_xyxy'];width,height=dimensions[camera_name]
            camera.cx+=x0;camera.cy+=y0;camera.width.fill_(width);camera.height.fill_(height)
            coordinates=torch.tensor([[.5,.5],[float(crop.height.item())-.5,float(crop.width.item())-.5]],device=pipe.model.device)
            a=crop.generate_rays(0,coords=coordinates)
            b=camera.generate_rays(0,coords=coordinates+torch.tensor([y0,x0],device=pipe.model.device))
            torch.testing.assert_close(a.origins,b.origins,rtol=0,atol=1e-7)
            torch.testing.assert_close(a.directions,b.directions,rtol=0,atol=2e-6)
            camera.rescale_output_resolution(min(1920/width,1080/height))
            result=pipe.model.get_outputs_for_camera(camera)
            rgb=display_rgb(result['rgb']) if spec.get('format')=='exr' else result['rgb']
            pixels=np.uint8(np.clip(rgb.float().cpu().numpy()*255+.5,0,255))
            accumulation=result['accumulation'].float()
            straight=result['rgb'].float()/accumulation.clamp_min(1e-8)
            if spec.get('format')=='exr':straight=display_rgb(straight)
            straight_pixels=np.uint8(np.clip(straight.cpu().numpy()*255+.5,0,255))
            alpha=np.uint8(np.clip(accumulation.cpu().numpy()[...,0]*255+.5,0,255))
            h,w=pixels.shape[:2];origin=((1920-w)//2,(1080-h)//2)
            canvas=Image.new('RGB',(1920,1080));canvas.paste(Image.fromarray(pixels),origin);canvas.save(out/(camera_name+'_raw_fullhd.png'))
            rgba=Image.new('RGBA',(1920,1080));rgba.paste(Image.fromarray(np.dstack([straight_pixels,alpha]),'RGBA'),origin);rgba.save(out/(camera_name+'_rgba_fullhd.png'))
            receipts.append(dict(camera=camera_name,source_canvas=[width,height],render_size=[w,h],letterbox_origin=list(origin),crop_uncrop_ray_parity=True,raw_field=True,ground_truth_masking=False))
            print('FULLHD',camera_name,w,h,flush=True)
    atomic(out/'render_receipt.json',dict(step=step,checkpoint_sha256=sha(checkpoint),cap=cap,output_canvas=[1920,1080],training_rgb_resized=False,rgba='straight alpha; EXR unpremultiplied before display curve',views=receipts))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--spec',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--sizes',required=True);p.add_argument('--output',required=True);p.add_argument('--views',default='CB,GC,NB');p.add_argument('--cap',type=float,default=10);a=p.parse_args()
    run(a.root,a.spec,a.checkpoint,a.sizes,a.output,a.views.split(','),a.cap)
