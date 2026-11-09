#!/usr/bin/env python3
"""One recoverable DEC5 frame: S3 copy, native preparation, train-only maps, fit."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from run_dec5_sequence import read,write


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def run(root,frame):
    manifest=read(root/'manifest.json');recipe=read(root/'recipe.json')
    directory=root/'frames'/frame;cache=Path(manifest['cache'])/frame
    directory.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    def progress(phase,**kw):write(directory/'progress.json',dict(phase=phase,time=time.time(),**kw))
    if not (directory/'frequency_ready.json').exists():
        progress('copy')
        rig=read(root/'rig.json');names=sorted(rig['cameras'])
        files=directory/'copy_files.txt'
        files.write_text(''.join(f'{frame}/{name}_{frame}.tif\n' for name in names))
        # rclone itself retries interrupted transfers. Existing identical files
        # are retained; a stage receipt is published only after native decoding.
        subprocess.run(['rclone','copy',manifest['source'],manifest['cache'],
            '--files-from',str(files),'--transfers','4','--checkers','8',
            '--retries','8','--low-level-retries','20','--contimeout','30s',
            '--timeout','5m','--stats','60s','--stats-one-line','--log-level','NOTICE'],check=True)
        progress('prepare')
        from prepare_dec5_sequence import prepare
        prepare(root,frame,recipe)
        progress('semantics')
        assets=Path(recipe['sapiens_repo']).parent;assets.mkdir(parents=True,exist_ok=True)
        with (assets/'bootstrap.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not Path(recipe['sapiens_checkpoint']).exists() or not Path(recipe['sapiens_repo']).exists():
                subprocess.run(['rclone','copy',manifest['assets_source'],str(assets),'--s3-region','us-west-2',
                    '--retries','8','--low-level-retries','20'],check=True)
                if not Path(recipe['sapiens_repo']).exists():
                    subprocess.run(['tar','-xzf',str(assets/'sapiens2_code.tar.gz'),'-C',str(assets)],check=True)
                if sha(recipe['sapiens_checkpoint'])!='b85fdb50b7d6123a967d5ee4a505e222baff8d2f7ad6bbf353578c1a61dfbac9':
                    raise ValueError('Semantic checkpoint checksum mismatch')
        from prepare_dec5_sequence import semantics
        semantics(root,frame,recipe)
        progress('frequency')
        import torch
        from fast_freqmap import process_image,save_metadata
        from nerfstudio.scripts.lookcloser_preprocess import load_image_as_tensor
        torch.set_num_threads(2);torch.manual_seed(42)
        data=directory/'data';meta=read(data/'transforms.json');output=data/'lookcloser_frequencies';output.mkdir(exist_ok=True)
        for index,name in enumerate(meta['train_filenames']):
            source=data/name;target=output/(source.stem+'.pt');receipt=target.with_suffix('.json')
            digest=sha(source);existing=read(receipt,{})
            if target.exists() and existing.get('source_sha256')==digest and existing.get('fit_steps_per_level')==1000 and existing.get('map_sha256')==sha(target):
                values=torch.load(target,map_location='cpu',weights_only=True)
                if torch.isfinite(values).all() and values.shape==(meta['h']//8,meta['w']//8):continue
            image=load_image_as_tensor(source,torch.device('cuda'));torch.manual_seed(42+index)
            values=process_image(image,1000,8192,2048,8192,ps=8,ssim_thr=.95,win=7,amp=False)
            temporary=target.with_suffix('.pt.tmp');torch.save(values,temporary);temporary.replace(target)
            save_metadata(receipt,source.name,image.shape[:2],8,8192)
            detail=read(receipt);detail.update(source_sha256=digest,fit_steps_per_level=1000,
                map_sha256=sha(target),implementation_sha256=sha(Path(__file__).with_name('fast_freqmap.py')))
            write(receipt,detail);progress('frequency',completed=index+1,total=len(meta['train_filenames']))
            del image,values;torch.cuda.empty_cache()
        expected={Path(n).stem for n in meta['train_filenames']}
        if {p.stem for p in output.glob('*.pt')}!=expected:raise ValueError('Train-only frequency inventory mismatch')
        write(directory/'frequency_ready.json',dict(data_sha256=sha(data/'transforms.json'),maps=len(expected),revision=recipe['revision']))
        progress('frequency_ready');return
    progress('train')
    from train_dec5_sequence import train
    train(root,frame,recipe)
    progress('complete',revision=recipe['revision'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--frame',required=True)
    args=parser.parse_args()
    try:run(args.root,args.frame)
    except BaseException as exc:
        directory=args.root/'frames'/args.frame;failures=read(directory/'failures.json',[])
        failures.append(dict(time=time.time(),error=repr(exc),phase=read(directory/'progress.json',{}).get('phase')))
        write(directory/'failures.json',failures)
        write(directory/'progress.json',dict(phase='retry',time=time.time(),error=repr(exc)))
        raise


if __name__=='__main__':main()
