"""Verify every cached FAS cell maps to its own image coordinate and level."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from nerfstudio.lookcloser_pixel_sampler import LookCloserPixelSamplerConfig
from blur_runtime import sha,write


def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();data=args.root/'data';meta=json.loads((data/'transforms.json').read_text());files=[data/name for name in meta['train_filenames']];torch.set_num_threads(2)
 s=LookCloserPixelSamplerConfig().setup();s._initialize_buckets(SimpleNamespace(image_filenames=files))
 expected=[];offsets=[];widths=[];heights=[];offset=0
 for path in files:
  values=torch.load(data/'lookcloser_frequencies'/f'{path.stem}.pt',weights_only=True);h,w=values.shape
  expected.append(torch.round(torch.log(values/16)/np.log((8192/16)**(1/15))).long().flatten());offsets.append(offset);offset+=h*w;widths.append(w);heights.append(h)
 expected=torch.cat(expected);offsets=torch.tensor(offsets);widths=torch.tensor(widths);heights=torch.tensor(heights);linear=[]
 for level,cells in s.buckets.items():
  ids,y,x=cells.long().T
  assert ((y>=0)&(y<heights[ids])&(x>=0)&(x<widths[ids])).all(), 'Frequency cells lie outside their image map'
  indices=offsets[ids]+y*widths[ids]+x
  assert (expected[indices]==level).all(), 'Frequency labels moved to different pixels'
  linear.append(indices)
 counts=torch.bincount(torch.cat(linear),minlength=len(expected));assert (counts==1).all(),'Missing or duplicated map cells'
 result=dict(train_images=len(files),map_cells=len(expected),all_cells_once=True,all_levels_match=True,
             sampler_sha256=sha(Path(__file__).resolve().parents[2]/'nerfstudio/lookcloser_pixel_sampler.py'),
             image_shapes={str(shape):list(s.image_shapes.values()).count(shape) for shape in set(s.image_shapes.values())})
 write(data/'sampling_audit.json',result);print(json.dumps(result))

if __name__=='__main__':main()
