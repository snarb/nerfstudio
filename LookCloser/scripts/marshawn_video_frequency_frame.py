"""Fit every ALL66 source-bound map using the canonical per-image worker.

Each map has a separate immutable spec and lock; no process shares fitter
weights or optimizer state across cameras or frames. Existing maps are reused
only after the per-image worker validates their exact RGB SHA and recipe.
"""
from pathlib import Path
import argparse, json, os, subprocess, sys, time
from marshawn_video_train import read,sha,atomic


def specs(campaign,frame):
    campaign=Path(campaign).resolve();data=campaign/'frames'/('%06d'%frame)/'data'
    ready=read(data/'ready.json');tf=read(data/'transforms.json')
    if ready['frame']!=frame or ready['train']!=66 or len(tf['train_filenames'])!=66:raise ValueError('ALL66 frame required')
    rows={r['file_path']:r for r in tf['frames']};out=data/'frequency_specs';out.mkdir(exist_ok=True)
    code=campaign/'code';source=code/'nerfstudio/scripts/lookcloser_preprocess.py';paths=[]
    for filename in tf['train_filenames']:
        camera=rows[filename]['physical_camera'];image=data/filename;stem=Path(filename).stem
        spec=dict(frame=frame,camera=camera,image=str(image),image_sha256=sha(image),
            output=str(data/'lookcloser_frequencies'/(stem+'.pt')),source_code=str(code),
            preprocess_source_sha256=sha(source),allocator_GiB=4)
        path=out/(camera+'.json')
        if path.exists() and read(path)!=spec:raise ValueError('Frequency spec changed: '+camera)
        if not path.exists():atomic(path,spec)
        paths.append(path)
    return paths


def run(campaign,frame):
    campaign=Path(campaign).resolve();paths=specs(campaign,frame)
    worker=campaign/'code/LookCloser/scripts/marshawn_video_frequency.py'
    lease_fd=os.environ.get('LOOKCLOSER_GPU_LEASE_FD')
    inherited=() if lease_fd is None else (int(lease_fd),)
    if inherited:os.fstat(inherited[0])
    pending=list(paths)
    while pending:
        retry=[]
        for path in pending:
            result=subprocess.run([sys.executable,str(worker),'--spec',str(path)],pass_fds=inherited)
            if result.returncode==75:retry.append(path)
            elif result.returncode:raise SystemExit(result.returncode)
        pending=retry
        if pending:time.sleep(10)
    # All workers completed validation before ready becomes the immutable cold
    # training input. Admission must be built after this final inventory seal.
    from prepare_marshawn_tuning import refresh_inventory,verify_inventory
    data=campaign/'frames'/('%06d'%frame)/'data'
    prepared=read(data/'ready.json')
    if 'recipe_source' not in prepared:
        prepared['recipe_source']=dict(path='recipe.json',sha256=sha(campaign/'recipe.json'),
            derivation='Campaign-sealed exact selected Marshawn hyperparameters; canonical source in factory_proof.json')
        atomic(data/'ready.json',prepared)
    ready=refresh_inventory(data);verify_inventory(data)
    atomic(data/'frequency_ready.json',dict(frame=frame,train=66,ready_sha256=sha(data/'ready.json'),
        inventory_files=len(ready['inventory']['files']),map_count=66))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--frame',type=int,required=True);p.add_argument('--specs-only',action='store_true');a=p.parse_args()
    if a.specs_only:print(json.dumps([str(p) for p in specs(a.campaign,a.frame)]))
    else:run(a.campaign,a.frame)
