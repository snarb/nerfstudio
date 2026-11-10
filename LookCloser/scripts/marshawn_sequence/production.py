"""One independent final frame, including native preparation and durable training."""
import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path
from common import atomic,read,sha


def wait_for_existing_worker(output):
    """A crashed orchestration parent must rejoin its surviving neural child."""
    path=Path(output)/'worker.lock'
    if path.exists():
        with path.open('a+') as lease:fcntl.flock(lease,fcntl.LOCK_EX)


def run(root,frame):
    root=Path(root);settings=read(root/'settings.json');recipe=read(root/'selected_recipe.json')
    if settings.get('production_torch_extensions_dir'):os.environ['TORCH_EXTENSIONS_DIR']=settings['production_torch_extensions_dir']
    tools=Path(__file__).parent;local=root/'frames'/f'{frame:06d}';local.mkdir(parents=True,exist_ok=True)
    frame_lease=(local/'frame.lock').open('a+')
    fcntl.flock(frame_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    native=local/'native';dataset=local/'data';output=local/'train'
    def finish(completion):
        from archive import relocate
        if settings.get('archive_to'):
            completion=relocate(root,frame,completion,settings['archive_to'])
        atomic(local/'frame_complete.json',completion)
        # Only release resumable optimizer states after a durable final receipt.
        for old in (output/'checkpoints').glob('*.ckpt'):old.unlink()
        for p in (native/'linear').glob('*.npy'):p.unlink()
    if (local/'frame_complete.json').exists():return
    if (local/'archive_receipt.json').exists():
        atomic(local/'frame_complete.json',read(local/'archive_receipt.json'));return
    if (local/'export_complete.json').exists():
        finish(read(local/'export_complete.json'));return
    wait_for_existing_worker(output)
    def command(args):
        subprocess.run([str(x) for x in args],check=True)
    if settings.get('source_owners'):
        from native_transfer import ensure_native
        ensure_native(root,frame)
    elif not (native/'native_ready.json').exists():
        command([settings['python'],tools/'prepare.py','--frame',frame,'--output',native,'--box',root/'box.json','--split',root/'split.json'])
    sam_env=dict(os.environ,PYTHONPATH=settings.get('sam_dependency_path',settings['dependency_path']))
    sam_command=[settings['sam_python'],str(tools/'parts.py'),'--data',str(native)]
    if settings.get('sam_checkpoint'):sam_command+=['--checkpoint',settings['sam_checkpoint']]
    subprocess.run(sam_command,check=True,env=sam_env)
    seeds=native/'points3D.txt'
    if not seeds.exists():seeds=Path(settings.get('seed_root',str(root/'seed_points')))/f'{frame:06d}/sparse/text/points3D.txt'
    command([settings['python'],tools/'qualify_parts.py','--data',native,'--seeds',seeds])
    if recipe['parameters']['frequency_source_policy']!='uniform16':
        # Final training includes former heldout cameras: generate their maps too.
        tf=read(native/'transforms.json');tf['train_filenames']=[f['file_path'] for f in tf['frames']];atomic(native/'transforms.json',tf)
        command([settings['python'],tools/'frequency.py','--data',native,'--source-code',settings['source_code']])
    from assets import assemble,admission,exr_variant,scale_variant,resolution_variant,confidence_variant,historical_confidence_variant
    sys.path[:0]=[settings['source_code'],str(Path(settings['source_code'])/'LookCloser/scripts')]
    assemble(native,dataset,recipe['parameters']['frequency_source_policy'],all_train=True);admission(dataset)
    if recipe['parameters'].get('confidence_profile')=='historical_power':
        historical_confidence_variant(dataset,local/'data_confidence',settings['source_code'],recipe['parameters'].get('confidence_width',64));dataset=local/'data_confidence'
    elif recipe['parameters'].get('confidence_width')==32:
        confidence_variant(dataset,local/'data_confidence',32);dataset=local/'data_confidence'
    units=recipe['parameters'].get('model_units_scale',settings.get('model_units_scale',1.))
    if units!=1:
        scale_variant(dataset,local/'data_model',units);dataset=local/'data_model'
    if recipe['parameters'].get('max_res',8192)!=8192:
        resolution_variant(dataset,local/'data_resolution',recipe['parameters']['max_res']);dataset=local/'data_resolution'
    if recipe['parameters'].get('format')=='exr':
        exr_variant(dataset,local/'data_exr');dataset=local/'data_exr'
    else:
        # Linear masters can be recreated from immutable PNG16 sources; bounded staging footprint.
        for p in (native/'linear').glob('*.npy'):p.unlink()
    from campaign import make_spec
    spec_path=make_spec(root,recipe['parameters'],dataset,output,recipe['steps'],allow_plateau=False)
    bound=read(spec_path)
    if bound.get('nerfacc_extension_sha256') and sha(bound['nerfacc_extension'])!=bound['nerfacc_extension_sha256']:raise ValueError('Production runtime extension changed before worker launch')
    worker='worker_background.py' if recipe['parameters'].get('background_opacity_prior') else 'worker.py'
    from agent import completed
    if not completed(dict(completion=str(output/'stage_complete.json'),target=recipe['steps'],max_optimizer_attempts=150000)):
        command([settings['python'],tools/worker,'--spec',spec_path])
    result=read(output/'stage_complete.json')
    rows=[r for r in read(output/'metrics.json') if r['cap']==recipe.get('selected_inference_cap',10)]
    if not rows:raise ValueError('Frozen inference cap has no committed evaluation')
    best=min(rows,key=lambda r:(r['metrics']['lpips'],-r['metrics']['psnr']))
    ckpt=output/'checkpoints'/('step-%09d.ckpt'%best['step'])
    if not ckpt.exists():raise ValueError('Selected scored checkpoint is missing')
    from archive import export
    model=export(local,ckpt,best)
    completion=dict(frame=frame,recipe_sha256=sha(root/'selected_recipe.json'),aabb_sha256=sha(root/'box.json'),train_cameras=72,in_sample_metrics=True,completed_iterations=result['completed_iterations'],selected=best,checkpoint=str(model),checkpoint_sha256=sha(model),cold_provenance_sha256=sha(output/'cold_provenance.json'),inference_only_export=True)
    atomic(local/'export_complete.json',completion)
    if spec_path.exists():
        spec=read(spec_path)
        if spec.get('sampling_memmap'):__import__('shutil').rmtree(spec['sampling_memmap'],ignore_errors=True)
    finish(completion)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frame',type=int,required=True);a=p.parse_args();run(a.root,a.frame)
