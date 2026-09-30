"""Compare integration rules on identical frozen checkpoint face rays."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import torch
import yaml
from blur_runtime import metrics,write


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--allocator',action='store_true');p.add_argument('--coarse',action='store_true');p.add_argument('--occupancy',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    request=json.loads((args.run/'request.json').read_text());selection=json.loads((args.run/'selection.json').read_text())
    cfg=yaml.load(Path(selection['config']).read_text(),Loader=yaml.Loader)
    state=torch.load(selection['checkpoint'],map_location='cpu',weights_only=False)
    pipe=cfg.pipeline.setup(device='cuda');pipe.load_pipeline(state['pipeline'],state['step']);pipe.eval();del state
    original_binaries=pipe.model.occupancy_grid.binaries.clone()
    args.output.mkdir(parents=True,exist_ok=True);results=[];panels=[]
    for split,ds,camids in [('eval',pipe.datamanager.eval_dataset,[95,150]),('train',pipe.datamanager.train_dataset,[97,151])]:
        for i,path in enumerate(ds.image_filenames):
            name=Path(path).name
            if int(name.split('_')[1]) not in camids:continue
            x0,y0,x1,y1=request['rois_by_image'][name]['face'];gt=ds[i]['image'][y0:y1,x0:x1].cuda()
            yy,xx=torch.meshgrid(torch.arange(y0,y1,device='cuda'),torch.arange(x0,x1,device='cuda'),indexing='ij')
            rays=ds.cameras[i:i+1].to('cuda').generate_rays(0,coords=torch.stack([yy,xx],-1).float()+.5)
            pictures=[gt];names=['GT']
            variants = [('adaptive','adaptive',256,False,1024),('fixed256','fixed',256,False,1024),('fixed1024','fixed',1024,False,1024)]
            if args.allocator:
                variants = [('legacy1024','adaptive',256,False,1024),('corrected1024','adaptive',256,True,1024),('legacy4096','adaptive',256,False,4096)]
            if args.coarse:
                variants = [('coarse00625','adaptive',256,False,1024),('coarse001','adaptive',256,False,1024),('coarse0005','adaptive',256,False,4096)]
            if args.occupancy:
                variants = [('original','adaptive',256,False,1024),('dilate1','adaptive',256,False,1024),('fixed1024','fixed',1024,False,1024)]
            for label,mode,samples,corrected,cap in variants:
                pipe.model.occupancy_grid.binaries.copy_(original_binaries);pipe.model._eval_occupancy_backup=None
                pipe.model.config.occupancy_eval_dilation_radius=int(label=='dilate1')
                pipe.model.config.ray_sampling_mode=mode;pipe.model.config.fixed_num_samples_per_ray=samples
                pipe.model.config.corrected_arm_allocator=corrected;pipe.model.config.max_steps_per_ray=cap
                if args.coarse:pipe.model.config.adaptive_coarse_step_size={'coarse00625':.00625,'coarse001':.001,'coarse0005':.0005}[label]
                output=pipe.model.get_outputs_for_camera_ray_bundle(rays)
                if not torch.isfinite(output['rgb']).all():raise FloatingPointError('Nonfinite probe')
                row=dict(split=split,image=name,mode=label,**metrics(pipe.model,output['rgb'],gt),opacity=float(output['accumulation'].mean()))
                if 'num_samples_per_ray' in output:
                    row.update(samples_mean=float(output['num_samples_per_ray'].float().mean()),samples_max=int(output['num_samples_per_ray'].max()))
                results.append(row);pictures.append(output['rgb']);names.append(label)
                print(json.dumps(row),flush=True)
            h,w=gt.shape[:2];panel=Image.new('RGB',(w*4,h+24),(35,35,35));d=ImageDraw.Draw(panel)
            for j,pic in enumerate(pictures):
                panel.paste(Image.fromarray(np.rint(pic.cpu().numpy().clip(0,1)*255).astype('uint8')),(j*w,24));d.text((j*w+3,3),f'{name[:7]} {names[j]}',fill='white')
            panel.save(args.output/f'{split}_{name}.jpg',quality=95);panels.append(panel)
    write(args.output/'metrics.json',dict(checkpoint=selection['checkpoint'],step=selection['step'],results=results))
    contact=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),(35,35,35));y=0
    for panel in panels:contact.paste(panel,(0,y));y+=panel.height
    contact.save(args.output/'contact.jpg',quality=95)


if __name__=='__main__':main()
