"""Train-only full-image frequency maps with per-image receipts and heartbeat."""
import argparse
import json
from pathlib import Path
import sys
import time
import os

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import numpy as np
from PIL import Image
from nerfstudio.scripts.lookcloser_preprocess import train_progressive_and_estimate_frequency_map, save_frequency_metadata
from blur_runtime import write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('--per-level',type=int,default=1000)
    a=p.parse_args();out=a.data/'lookcloser_frequencies';out.mkdir(exist_ok=True)
    meta=json.loads((a.data/'transforms.json').read_text());names=set(meta['train_filenames'])
    rows=[r for r in meta['frames'] if r['file_path'] in names]
    torch.set_num_threads(2);start=time.time()
    for index,row in enumerate(rows):
        path=a.data/row['file_path'];stem=path.stem;receipt=out/f'{stem}.receipt.json'
        request=dict(rgb_sha256=sha(path),seed=42,steps_per_level=a.per_level,mask=None)
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['request']!=request or old['sha256']!=sha(out/f'{stem}.pt'):raise ValueError('Stale frequency cache')
            continue
        torch.manual_seed(42);np.random.seed(42)
        image=torch.from_numpy(np.array(Image.open(path).convert('RGB'))).cuda().float()/255
        fitted,freq,levels,_=train_progressive_and_estimate_frequency_map(image,steps=None,
            train_steps_per_level=a.per_level,batch_size=8192,lr=.01,ssim_threshold=.95,patch_size=8,
            eval_patch_batch_size=1024,n_levels=16,n_features=2,min_res=16,max_res=8192,
            log2_hashmap_size=19,ssim_window_size=7)
        torch.save(freq.cpu(),out/f'{stem}.pt')
        save_frequency_metadata(out/f'{stem}.json',stem,image.shape[:2],None,8,8,16,8192,16,2,19)
        write(receipt,dict(request=request,sha256=sha(out/f'{stem}.pt'),histogram=torch.bincount(levels.flatten().long(),minlength=16).tolist()))
        progress=dict(images=index+1,total=len(rows),seconds=time.time()-start,pid=os.getpid(),gpu_GiB=torch.cuda.memory_allocated()/2**30)
        write(a.data/'frequency_progress.json',progress);print(json.dumps(progress),flush=True)
        del fitted,freq,levels,image;torch.cuda.empty_cache()
    write(a.data/'frequency_complete.json',dict(seconds=time.time()-start,images=len(rows),steps_per_level=a.per_level))


if __name__=='__main__':main()
