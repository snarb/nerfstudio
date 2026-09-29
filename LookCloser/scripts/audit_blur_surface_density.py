"""Probe learned density near a fixed mesh reference on actual training rays.

The historical mesh is an imperfect diagnostic reference, not ground truth.
No mesh values are supplied to training by this script.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from blur_runtime import write, sha


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    metadata=json.loads((args.reference/'transforms.json').read_text())
    reference=next(r for r in metadata['frames'] if Path(r['file_path']).stem=='train_0033')
    with gzip.open(args.reference/reference['depth_file_path'],'rb') as stream:depth=np.load(stream).squeeze()
    valid=np.array(Image.open(args.reference/reference['mask_path']))>0
    valid &= np.isfinite(depth)&(depth>0)
    roi=np.zeros_like(valid);roi[330:700,800:1150]=True;valid &= roi
    locations=np.argwhere(valid);locations=locations[np.linspace(0,len(locations)-1,1024).astype(int)]
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    pipe=state['config'].pipeline.setup(device='cuda');pipe.load_pipeline(state['pipeline'],state['step']);pipe.eval()
    ds=pipe.datamanager.train_dataset
    index=next(i for i,p in enumerate(ds.image_filenames) if Path(p).stem=='train_0033')
    source=json.loads((Path(state['request']['data'])/'transforms.json').read_text())
    frame=next(r for r in source['frames'] if Path(r['file_path']).stem=='train_0033')
    if frame['transform_matrix']!=reference['transform_matrix']:raise ValueError('Reference gauge differs')
    coords=torch.tensor(locations,device='cuda').float()+.5
    rays=ds.cameras[index:index+1].to('cuda').generate_rays(0,coords=coords)
    target_depth=torch.tensor(depth[locations[:,0],locations[:,1]],device='cuda')[:,None]*rays.metadata['directions_norm']
    rendered=pipe.model(rays)
    offsets=torch.linspace(-.02,.02,257,device='cuda')
    distances=target_depth+offsets[None,:]
    positions=rays.origins[:,None]+rays.directions[:,None]*distances[:,:,None]
    density=pipe.model.field.density_fn(positions.reshape(-1,3)).reshape(len(locations),-1).float()
    center=density[:,128];peak=density.max(1).values
    peak_offset=offsets[density.argmax(1)]
    write(args.output,dict(checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),step=state['step'],
         reference='Historical mesh depth on train_0033 face; approximate, not ground truth',
         median_reference_distance=float(target_depth.median()),
         median_predicted_distance=float(rendered['depth'].median()),
         median_absolute_depth_error=float((rendered['depth']-target_depth).abs().median()),
         median_density_at_reference=float(center.median()),median_peak_density=float(peak.median()),
         median_absolute_peak_offset=float(peak_offset.abs().median()),
         opacity_mean=float(rendered['accumulation'].mean()),
         offsets=offsets.cpu().tolist(),mean_density_profile=density.mean(0).cpu().tolist()))
    print(args.output,flush=True)


if __name__=='__main__':main()
