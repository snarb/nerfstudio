"""Compare ray integration on frozen weights; this is not a training ablation."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from blur_runtime import metrics, write, sha


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    pipe = state['config'].pipeline.setup(device='cuda')
    pipe.load_pipeline(state['pipeline'], state['step'])
    pipe.eval()
    config = deepcopy(pipe.model.config)
    modes = {'original': {}, 'corrected_allocator': {'corrected_arm_allocator': True},
             **{f'fixed{n}': {'ray_sampling_mode': 'fixed', 'fixed_num_samples_per_ray': n}
                for n in [256, 1024, 4096]}}
    results = []
    start = time.monotonic()
    for split, index in [('train', 33), ('eval', 0)]:
        ds = getattr(pipe.datamanager, split+'_dataset')
        item = ds[index]
        box = state['request']['rois_by_image'][Path(ds.image_filenames[index]).name]['face']
        x0, y0, x1, y1 = box
        yy, xx = torch.meshgrid(torch.arange(y0,y1,2,device='cuda'),
                                torch.arange(x0,x1,2,device='cuda'), indexing='ij')
        coords = torch.stack([yy,xx],-1).float()+.5
        gt = item['image'][y0:y1:2,x0:x1:2].cuda()
        panels = [gt]
        for name, changes in modes.items():
            pipe.model.config = deepcopy(config)
            for key, value in changes.items():
                if not hasattr(pipe.model.config,key): raise ValueError(key)
                setattr(pipe.model.config,key,value)
            camera = ds.cameras[index:index+1].to('cuda')
            out = pipe.model.get_outputs_for_camera_ray_bundle(camera.generate_rays(0,coords=coords))
            pred = out['rgb']
            values = metrics(pipe.model,pred,gt)
            results.append(dict(split=split,index=index,mode=name,**values,
                                mean_opacity=float(out['accumulation'].mean())))
            panels.append(pred)
            print(results[-1],flush=True)
        panel = torch.cat(panels,1).cpu().numpy()
        Image.fromarray(np.rint(panel.clip(0,1)*255).astype('uint8')).save(args.output/f'{split}_face.png')
    write(args.output/'complete.json',dict(checkpoint=str(args.checkpoint),
          checkpoint_sha256=sha(args.checkpoint),step=state['step'],stride=2,
          order=['GT',*modes],results=results,seconds=time.monotonic()-start))


if __name__ == '__main__': main()
