"""Frozen progressive fitter, collecting two first-crossing thresholds in one fit."""
import argparse
import inspect
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from common import atomic,read,sha


def fit(data,source_code,only=None):
    import sys
    sys.path[:0]=[source_code]
    import nerfstudio.scripts.lookcloser_preprocess as prep
    root=Path(data);tf=read(root/'transforms.json');train={Path(p).stem for p in tf['train_filenames']}
    if only:train&=set(only)
    original=prep.train_progressive_and_estimate_frequency_map
    text=inspect.getsource(original)
    text=text.replace('level_map = torch.full((h_steps, w_steps), -1, dtype=torch.int16, device=device)','level_map = torch.full((h_steps, w_steps), -1, dtype=torch.int16, device=device)\n    low_map = level_map.clone()')
    text=text.replace('ok = scores >= float(ssim_threshold)','low_ok = (scores >= .90) & (low_map[idxs[:, 0], idxs[:, 1]] < 0)\n                low_idxs = idxs[low_ok]\n                low_map[low_idxs[:, 0], low_idxs[:, 1]] = int(level)\n                ok = scores >= float(ssim_threshold)')
    text=text.replace('return model, freq_map, level_map, debug_bundle','return model, freq_map, level_map, debug_bundle, low_map')
    namespace=dict(prep.__dict__);exec(compile(text,inspect.getfile(original),'exec'),namespace)
    dual=namespace[original.__name__]
    for c in sorted(train):
        target=root/'learned'/'ssim095'/(c+'.pt')
        if target.exists() and (root/'learned'/'ssim090'/(c+'.pt')).exists():continue
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
        image=torch.from_numpy(np.array(Image.open(root/'images'/(c+'.png')),dtype=np.float32)/255).to('cuda')
        model,freq,levels,_,low=dual(image_tensor=image,steps=None,train_steps_per_level=1000,batch_size=16384,lr=.01,ssim_threshold=.95,patch_size=8,eval_patch_batch_size=1024,n_levels=16,n_features=2,min_res=16,max_res=8192,log2_hashmap_size=23,ssim_window_size=7)
        low[low<0]=15
        low_freq=torch.empty_like(freq)
        for level in range(16):low_freq[low==level]=model.get_resolution_at_level(level)
        for policy,values in [('ssim095',freq),('ssim090',low_freq)]:
            path=root/'learned'/policy/(c+'.pt');path.parent.mkdir(exist_ok=True,parents=True)
            tmp=path.with_suffix('.tmp');torch.save(values.detach().cpu().float(),tmp);tmp.replace(path)
            atomic(path.with_suffix('.json'),dict(policy=policy,source_image_sha256=sha(root/'images'/(c+'.png')),fitter_source_sha256=sha(inspect.getfile(original)),collector_sha256=sha(__file__),seed=42,train_steps_per_level=1000,patch_size=8,stride=8,threshold=.95 if policy=='ssim095' else .9,frequency_map_sha256=sha(path),unresolved_policy='max15; original fitter may stop once every patch resolves at.95; .90 collected without changes to fitting or RNG'))
        print('FREQUENCY_DONE',c,flush=True);del model,image,freq,levels,low,low_freq;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--source-code',required=True);p.add_argument('--only',nargs='+');a=p.parse_args();fit(a.data,a.source_code,a.only)
