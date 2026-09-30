"""Parallel per-image preprocessing; all workers share the frozen train split."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def fit_one(data, row):
    import numpy as np
    from PIL import Image
    import torch
    from blur_runtime import write, sha
    from nerfstudio.scripts.lookcloser_preprocess import train_progressive_and_estimate_frequency_map, save_frequency_metadata
    torch.set_num_threads(2)
    path=Path(data)/row['file_path'];out=Path(data)/'lookcloser_frequencies';stem=path.stem
    request=dict(rgb_sha256=sha(path),seed=42,steps_per_level=1000,mask=None)
    receipt=out/f'{stem}.receipt.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['request']!=request or old['sha256']!=sha(out/f'{stem}.pt'):raise ValueError('Stale frequency cache')
        return stem
    torch.manual_seed(42);np.random.seed(42)
    image=torch.from_numpy(np.array(Image.open(path).convert('RGB'))).cuda().float()/255
    start=time.monotonic()
    fitted,freq,levels,_=train_progressive_and_estimate_frequency_map(image,steps=None,
        train_steps_per_level=1000,batch_size=8192,lr=.01,ssim_threshold=.95,patch_size=8,
        eval_patch_batch_size=1024,n_levels=16,n_features=2,min_res=16,max_res=8192,
        log2_hashmap_size=19,ssim_window_size=7)
    torch.save(freq.cpu(),out/f'{stem}.pt')
    save_frequency_metadata(out/f'{stem}.json',stem,image.shape[:2],None,8,8,16,8192,16,2,19)
    write(receipt,dict(request=request,sha256=sha(out/f'{stem}.pt'),seconds=time.monotonic()-start,
                       histogram=torch.bincount(levels.flatten().long(),minlength=16).tolist()))
    del fitted,freq,levels,image;torch.cuda.empty_cache()
    return stem


def main():
    from blur_runtime import write
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    meta=json.loads((args.data/'transforms.json').read_text());names=set(meta['train_filenames'])
    rows=[r for r in meta['frames'] if r['file_path'] in names]
    if len(rows)!=162 or names & set(meta['val_filenames']):raise ValueError('Unexpected train/eval split')
    (args.data/'lookcloser_frequencies').mkdir(exist_ok=True)
    start=time.monotonic()
    write(args.data/'frequency_progress.json',dict(images=0,total=len(rows),seconds=0,pid=os.getpid()))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        futures=[executor.submit(fit_one,str(args.data),row) for row in rows]
        for count,future in enumerate(as_completed(futures),1):
            name=future.result()
            progress=dict(images=count,total=len(rows),image=name,seconds=time.monotonic()-start,pid=os.getpid(),workers=args.workers)
            write(args.data/'frequency_progress.json',progress);print(json.dumps(progress),flush=True)
    write(args.data/'frequency_complete.json',dict(seconds=time.monotonic()-start,images=len(rows),steps_per_level=1000,workers=args.workers))


if __name__=='__main__':main()
