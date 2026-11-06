"""Resolve immutable DAG inputs, then run one cold configuration to saturation."""
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
from prepare_marshawn_tuning import write

def score(result):
    selected=result['selection']
    return float(selected['masked_lpips']),-float(selected['psnr']),int(selected['step'])

def winner(results):
    valid=[r for r in results if r and not r.get('skipped')]
    if not valid:raise ValueError('No completed candidate')
    best=min(float(r['selection']['masked_lpips']) for r in valid)
    return min((r for r in valid if r['selection']['masked_lpips']<=best+.001),key=lambda r:(-r['selection']['psnr'],r['selection']['step']))

def resolve(task,deps,data,output):
    base=dict(data=str(data),output=str(output),objective='full',sampling='baseline',width=64,radius=0,coarse_step=.00625,distortion_final=.01,seed=42,frequency_threshold=.95)
    if task.get('inherit'):
        choices=list(deps.values())
        if task['inherit']=='actor':choices=[r for r in choices if r and r.get('config',{}).get('objective')=='actor']
        if task['inherit']=='control':choices=[r for r in choices if r and r.get('config',{}).get('objective')=='full' and r['config']['sampling']=='baseline' and r['config']['radius']==0]
        parent=winner(choices);base.update(parent['config']);base.update(data=str(data),output=str(output))
    base.update(task.get('overrides',{}));return base

def fit_maps(source,output,threshold):
    import shutil,torch,random,numpy as np,inspect,socket
    import nerfstudio.scripts.lookcloser_preprocess as preprocessing
    from nerfstudio.scripts.lookcloser_preprocess import LookCloserPreprocessConfig,load_image_as_tensor
    from prepare_marshawn_tuning import sha,verify_inventory,refresh_inventory
    source=Path(source);output=Path(output)
    source_ready=verify_inventory(source)
    source_ready_hash=sha(source/'ready.json')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'ready.json').exists():
        pending=json.loads((output/'ready.json').read_text())
        if pending.get('frequency_parent_ready_sha256')!=source_ready_hash:
            raise ValueError('Frequency parent dataset changed')
        if 'inventory' in pending:
            ready=verify_inventory(output)
            if ready['inventory']['frequency_recipe']['ssim_threshold']!=threshold:
                raise ValueError('Sealed frequency threshold changed')
            return output
    else:
        for name in ['images','masks','semantics']:
            if not (output/name).exists():(output/name).symlink_to(source/name,target_is_directory=True)
        for name in ['transforms.json','box.json']:shutil.copy2(source/name,output/name)
        pending=dict(source_ready)
        pending.pop('inventory',None)
        pending['frequency_parent_ready_sha256']=source_ready_hash
        pending['frequency_policy']=f'INCOMPLETE new train-only SSIM{threshold} fits; not ready for training'
        write(output/'ready.json',pending)
    maps=output/'lookcloser_frequencies';maps.mkdir(exist_ok=True)
    config=LookCloserPreprocessConfig(max_res=8192,train_steps_per_level=1000,ssim_threshold=threshold,
             eval_patch_batch_size=1024,patch_size=8,ssim_window_size=7,n_levels=16,min_res=16)
    recipe=dict(ssim_threshold=threshold,train_steps_per_level=1000,eval_patch_batch_size=1024,
                fit_script_sha256=sha(__file__),preprocess_source_sha256=sha(inspect.getfile(preprocessing)),
                ssim_window_size=7,fit_seed=42)
    meta=json.loads((source/'transforms.json').read_text())
    for rel in meta['train_filenames']:
        image=source/rel;dest=maps/(image.stem+'.pt');image_hash=sha(image)
        if dest.exists() and dest.with_suffix('.json').exists():
            info=json.loads(dest.with_suffix('.json').read_text())
            if info.get('source_image_sha256')!=image_hash or any(info.get(k)!=v for k,v in recipe.items()) or info.get('frequency_map_sha256')!=sha(dest):
                raise ValueError('Stale frequency map/source/recipe: '+image.stem)
            continue
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        tensor=load_image_as_tensor(image,device=torch.device('cuda'));tmp=maps/(image.stem+'.tmp.pt')
        config._process_single_image(img_path=image,image_tensor=tensor,save_freq_path=tmp,debug_out_dir=None,scene_size=2.)
        info=json.loads(tmp.with_suffix('.json').read_text());info.update(recipe)
        info.update(source_image_sha256=image_hash,frequency_map_sha256=sha(tmp),host=socket.gethostname(),
            torch_version=torch.__version__,cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(),
            repeatability='GPU fitting is stochastic; fixed seed does not establish bitwise repeatability',
            censoring='Maximum level includes unresolved patches; per-patch convergence status not emitted by legacy fitter')
        write(dest.with_suffix('.json'),info);tmp.replace(dest)
        if tmp.with_suffix('.json').exists():tmp.with_suffix('.json').unlink()
        del tensor;torch.cuda.empty_cache()
    refresh_inventory(output)
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    task=json.loads(a.task);deps=json.loads(os.environ.get('TUNING_DEPENDENCIES_JSON','{}'));a.output.mkdir(parents=True,exist_ok=True)
    spec=resolve(task,deps,a.data,a.output)
    if task.get('select_only'):
        write(a.output/'complete.json',winner(list(deps.values())));return
    if task.get('conditional')=='geometry':
        parent=winner(list(deps.values()))
        if not parent.get('geometry_trigger',False):write(a.output/'complete.json',dict(skipped=True,reason='No declared geometry trigger',config=spec));return
    if task.get('conditional')=='frequency_gain':
        base=deps[task['frequency_control']];best=winner([r for k,r in deps.items() if k!=task['frequency_control']])
        if best['selection']['masked_lpips']>base['selection']['masked_lpips']*.97:
            write(a.output/'complete.json',dict(skipped=True,reason='Frequency improvement below3%',config=spec));return
        spec['frequency_threshold']=round(best['config']['frequency_threshold']+task['threshold_offset'],2)
    if spec['frequency_threshold']!=.95:
        spec['data']=str(fit_maps(a.data,a.output/'frequency_data',spec['frequency_threshold']))
    path=a.output/'spec.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Immutable resolved specification changed')
    write(path,spec)
    subprocess.run([sys.executable,str(Path(__file__).with_name('train_marshawn_tuning.py')),'--spec',str(path)],check=True)
    from marshawn_tuning_metrics import actor_regression_guard
    complete=a.output/'complete.json';result=json.loads(complete.read_text())
    controls=[r for r in deps.values() if r and not r.get('skipped') and r.get('config',{}).get('objective')=='full' and r['config'].get('sampling')=='baseline' and r['config'].get('radius')==0 and r['config'].get('seed')==spec['seed']]
    result['actor_regression_guard']=actor_regression_guard(result,controls[0] if controls else None)
    write(complete,result)
if __name__=='__main__':main()
