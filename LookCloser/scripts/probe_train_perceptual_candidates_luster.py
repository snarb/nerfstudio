"""CPU TRAIN-only pilot: baseline/detail/full RGB on geometric grid patches."""
import argparse
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from train_perceptual_utility_luster_hybrid import load_metric
from luster_hybrid_common import read,write,source_path,sha,native_rows,assert_source
from supported_luster_hybrid import geometric_support
from risk_single_luster_hybrid import masked_low

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--target',type=int,required=True)
p.add_argument('--source',type=int,required=True)
a=p.parse_args();r=a.root;target=a.target;source=a.source
rows=native_rows();assert_source(rows,target,source)
config=read(r/'candidates/single_photometric_depth25/config.json')
dependency=r/'witness_native_restoration'/f'cam{source:03d}'/'receipt.json'
results=[];inputs={str(dependency):sha(dependency)};skipped=[]
with tempfile.TemporaryDirectory(prefix='luster_perceptual_pilot_') as tmp:
    metric,mr=load_metric(Path(tmp),'cpu')
    for witness in read(dependency)['witnesses']:
        assert_source(rows,target,witness);assert witness not in [7,12,95,150]
        d=r/f'variants/cubic75/warps/cam{witness:03d}/src{source:03d}'
        sr=read(d/'receipt.json');x0,y0,x1,y1=sr['box']
        arrays={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in sr['arrays']}
        f=r/f'variants/target_witness_confidence60/frozen_render_projections/cam{witness:03d}/render{target:03d}'
        fr=read(f/'receipt.json');assert fr['box']==sr['box']
        support=geometric_support(arrays,config)&np.load(f/'support.npy',mmap_mode='r')
        eligible=[]
        for y in range(0,support.shape[0]-255,256):
            for x in range(0,support.shape[1]-255,256):
                if support[y:y+256,x:x+256].mean()>=.8:eligible.append((x,y))
        if not eligible:
            skipped.append(dict(witness=witness,reason='no geometrically supported 256px grid patch'))
            continue
        b=np.load(f/'rgb.npy',mmap_mode='r');s=np.asarray(arrays['rgb'])
        sl,_=masked_low(s,support,6.);bl,_=masked_low(b,support,6.)
        h=np.clip(b+(s-sl)-(b-bl),0,1)
        native=source_path(witness)
        assert sha(native)==read(r/'preflight/manifest.json')['source_hashes'][str(native)]
        w=np.asarray(Image.open(native).convert('RGB'),np.float32)[y0:y1,x0:x1]/255
        indices=np.unique(np.linspace(0,len(eligible)-1,min(8,len(eligible)),dtype=int))
        for i in indices:
            x,y=eligible[i];sl=np.s_[y:y+256,x:x+256];mask=support[sl]
            def tensor(rgb):
                value=np.where(mask[...,None],np.clip(rgb[sl],0,1),.5).astype(np.float32)
                return torch.from_numpy(value).permute(2,0,1)[None]*2-1
            with torch.inference_mode():
                truth=tensor(w)
                scores={key:float(metric(tensor(value),truth)[0,0].numpy()[mask].mean())
                        for key,value in [('baseline',b),('detail',h),('full',s)]}
            results.append(dict(witness=witness,box=[x+x0,y+y0,x+x0+256,y+y0+256],support=float(mask.mean()),**scores))
        inputs.update({str(p):sha(p) for p in [d/'receipt.json',f/'receipt.json',native]})
    output=dict(usage='TRAIN-only exploratory comparison; no target photograph or evaluation mask',
        target=target,source=source,selection='up to eight evenly spaced eligible 256px grid tiles per witness; geometric support >=.8',
        rows=results,skipped=skipped,inputs=inputs,metric=read(mr),code_sha256=sha(Path(__file__)),
        limitation='Single source/target pilot; native-witness detail scale and crop LPIPS differ from final target evaluation.')
    write(r/'train_perceptual_utility'/f'pilot_cam{target:03d}_src{source:03d}.json',output)
    print(output['rows'],flush=True)
