#!/usr/bin/env python3
"""One source-bound TRAIN AA frequency pilot; explicit GPU release required."""
import argparse, hashlib, inspect, json, math, os, socket, sys, time
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def validate(spec_path):
    spec=json.loads(spec_path.read_text());release=json.loads(Path(spec['release']).read_text())
    if release.get('frequency_fit_authorized') is not True or release.get('reviewer')!='root' or release.get('spec_sha256')!=sha(spec_path):raise ValueError('Explicit source-bound root GPU release required')
    expected={'camera':'AA','split':'TRAIN','shape':[2024,3840],'ssim_threshold':.95,'patch_size':8,'stride':8,'n_levels':16,'min_res':16,'max_res':8192,'log2_hashmap_size':23,'train_steps_per_level':1000,'train_batch_size':16384,'eval_patch_batch_size':1024,'seed':42}
    if spec['recipe']!=expected:raise ValueError('Unreviewed frequency pilot recipe')
    for path,h in spec['inputs'].items():
        if sha(path)!=h:raise ValueError('Bound input changed: '+path)
    if sha(__file__)!=spec['worker_sha256']:raise ValueError('Frequency worker changed')
    complete=json.loads(Path(spec['source_complete']).read_text())
    if complete.get('status')!='TRAIN63_sources_complete_maps_unqualified' or complete.get('replays_exact') is not True or complete.get('eval_nine_files_unchanged') is not True:raise ValueError('Unqualified TRAIN source export')
    if release.get('source_complete_sha256')!=sha(spec['source_complete']):raise ValueError('Release/source export mismatch')
    return spec


def add_counts(counts,level,total,passed):
    if not 0<=level<16 or not 0<=passed<=total:raise ValueError('Invalid SSIM counts')
    row=counts.setdefault(str(level),{'evaluated':0,'passed':0})
    row['evaluated']+=total;row['passed']+=passed


def check_counts(counts,total_patches):
    remaining=total_patches
    if not counts or sorted(map(int,counts))!=list(range(max(map(int,counts))+1)):raise ValueError('Missing progressive level coverage')
    for level in sorted(map(int,counts)):
        row=counts[str(level)]
        if row['evaluated']!=remaining:raise ValueError('Incomplete progressive SSIM coverage')
        remaining-=row['passed']
    if remaining and (not counts or max(map(int,counts))!=15):raise ValueError('Censored patches require the final level evaluation')
    return remaining


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);a=parser.parse_args();spec=validate(a.spec)
    # CUDA imports/allocations happen only after explicit release validation.
    import numpy as np
    import torch
    import nerfstudio.scripts.lookcloser_preprocess as prep
    if sha(inspect.getfile(prep))!=spec['preprocess_sha256']:raise ValueError('Frozen preprocessing source changed')
    if torch.cuda.device_count()!=1:raise ValueError('Controller must pin exactly one GPU through its shared lock')
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=True)
    config=prep.LookCloserPreprocessConfig(max_res=8192,train_steps_per_level=1000,ssim_threshold=.95,eval_patch_batch_size=1024,patch_size=8,n_levels=16,min_res=16,log2_hashmap_size=23,train_batch_size=16384)
    source=Path(spec['image']);image=prep.load_image_as_tensor(source,device=torch.device('cuda'))
    if list(image.shape)!=[2024,3840,3]:raise ValueError('Wrong true source raster')
    counts={};censored=np.zeros((253,480),bool);original=prep.compute_ssim;start=time.time();torch.cuda.reset_peak_memory_stats();last_write=[-1,0.]
    def observed_ssim(*args,**kwargs):
        result=original(*args,**kwargs);frame=sys._getframe(1)
        if frame.f_code.co_name!='eval_and_assign':raise ValueError('Unexpected SSIM caller; do not infer censoring')
        level=int(frame.f_locals['level']);add_counts(counts,level,result.numel(),int((result>=.95).sum().item()))
        if level==15:
            indices=frame.f_locals['idxs'].detach().cpu().numpy();failed=(result<.95).detach().cpu().numpy()
            censored[indices[failed,0],indices[failed,1]]=True
        if level!=last_write[0] or time.time()-last_write[1]>=30:
            atomic(output/'progress.json',{'time':time.time(),'status':'fitting_frequency_AA','level':level,'counts':counts,'elapsed_seconds':time.time()-start,'gpu_peak_bytes':torch.cuda.max_memory_allocated()})
            last_write[:]=[level,time.time()]
        return result
    prep.compute_ssim=observed_ssim
    maps=output/'maps';maps.mkdir(exist_ok=True);destination=maps/'train_AA.pt'
    try:config._process_single_image(img_path=source,image_tensor=image,save_freq_path=destination,debug_out_dir=None,scene_size=2.)
    finally:prep.compute_ssim=original
    unresolved=check_counts(counts,253*480)
    if int(censored.sum())!=unresolved:raise ValueError('Censor mask/count mismatch')
    frequency=torch.load(destination,map_location='cpu',weights_only=True)
    if tuple(frequency.shape)!=(253,480) or not torch.isfinite(frequency).all():raise ValueError('Invalid output map')
    levels=np.rint(np.log(frequency.numpy()/16)/np.log(8192/16)*15).clip(0,15).astype(int)
    histogram=np.bincount(levels.ravel(),minlength=16).tolist()
    for level in range(16):
        expected=counts.get(str(level),{}).get('passed',0)+(unresolved if level==15 else 0)
        if histogram[level]!=expected:raise ValueError('Saved map and observed SSIM assignments disagree')
    metadata=json.loads(destination.with_suffix('.json').read_text());metadata.update(source_image_sha256=sha(source),fit_script_sha256=sha(__file__),preprocess_source_sha256=sha(inspect.getfile(prep)),ssim_threshold=.95,train_steps_per_level=1000,eval_patch_batch_size=1024,fit_seed=42,frequency_map_sha256=sha(destination),exact_unresolved_after_final_SSIM=unresolved)
    atomic(destination.with_suffix('.json'),metadata)
    from PIL import Image
    Image.fromarray(censored.astype(np.uint8)*255).save(output/'unresolved_patch_mask.png')
    heat=prep.colorize_level_map(torch.from_numpy(levels),16).resize((3840,2024),Image.Resampling.NEAREST)
    heat.save(output/'frequency_levels.png')
    image_cpu=Image.open(source).convert('RGB')
    for actor,box in spec['actor_crops'].items():
        actual=image_cpu.crop(box);colored=heat.crop(box);sheet=Image.new('RGB',(actual.width*2,actual.height))
        sheet.paste(actual,(0,0));sheet.paste(Image.blend(actual,colored,.45),(actual.width,0));sheet.save(output/('frequency_'+actor+'.png'))
    result={'technical_frequency_pilot':True,'camera':'AA','split':'TRAIN','source_shape':[2024,3840],'recipe':spec['recipe'],'source_image_sha256':sha(source),'source_complete_sha256':sha(spec['source_complete']),'map_sha256':sha(destination),'metadata_sha256':sha(destination.with_suffix('.json')),'counts_by_level':counts,'exact_unresolved_patches':unresolved,'unresolved_fraction':unresolved/(253*480),'max_level_fraction':float((levels==15).mean()),'level_histogram':histogram,'elapsed_seconds':time.time()-start,'gpu_peak_memory_bytes':torch.cuda.max_memory_allocated(),'host':socket.gethostname(),'gpu':torch.cuda.get_device_name(),'torch_version':torch.__version__,'instrumentation':'Read-only SSIM counters; no math, RNG, optimizer or model changes','whole_dataset_maps_qualified':False,'training_authorized':False}
    result['censor_mask_sha256']=sha(output/'unresolved_patch_mask.png');result['review_artifacts']={p.name:sha(p) for p in output.glob('frequency_*.png')}
    atomic(output/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
