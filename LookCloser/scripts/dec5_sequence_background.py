"""Restartable per-frame frozen background preparation for an explicit recipe."""
from pathlib import Path
import subprocess,sys,time
from run_dec5_sequence import read,write,identity
from dec5_sequence_worker import sha


def ensure(frame,root,specification):
    if specification!=dict(kind='bounded_ngp_v1',steps=20000,start_step=15189):
        raise ValueError('Only the measured bounded20k/15189 background recipe is supported')
    frame=Path(frame);root=Path(root);out=frame/'neural_background';out.mkdir(exist_ok=True)
    source=frame/'semantic_data/transforms.json';digest=sha(source)
    complete=read(out/'complete.json')
    if complete is None:
        write(frame/'progress.json',dict(phase='background_volume',time=time.time(),background_steps=20000))
        command=[sys.executable,str(Path(__file__).with_name('train_dec5_neural_background.py')),
            '--root',str(root),'--frame-root',str(frame),'--output',str(out),'--production-stage']
        log_path=out/'worker_stdout.log'
        with log_path.open('a') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            write(out/'child.json',dict(pid=child.pid,starttime=identity(child.pid),started=time.time(),command=command))
            code=child.wait()
        if code:
            with log_path.open('rb') as log:
                log.seek(0,2);log.seek(max(0,log.tell()-65536));tail=log.read().decode(errors='replace')
            reason='out of memory' if 'out of memory' in tail.lower() else 'failed'
            raise RuntimeError('Background volume '+reason+'; exit '+str(code)+'; '+tail[-500:])
        complete=read(out/'complete.json')
    request=read(out/'request.json',{})
    if not complete or complete.get('smoke') or complete.get('steps')!=20000:
        raise ValueError('Full20000-step background component required')
    if (request.get('source_data_sha256')!=digest or request.get('uses_validation_rgb_for_fitting') is not False
            or request.get('uses_test_rgb') is not False or len(request.get('records',[]))!=64):
        raise ValueError('Background component source provenance differs')
    expected=dict(scene_scale=1.5,empty_subject_half_width=.18,rays_per_batch=1024,max_res=1024,levels=12,hash_log2=19)
    if any(request.get(key)!=value for key,value in expected.items()):raise ValueError('Background component architecture differs')
    for record in request['records']:
        if sha(frame/'semantics'/(record['camera']+'.npz'))!=record['semantic_sha256']:
            raise ValueError('Background semantic evidence changed')
    if 'source_rgb_sha256' in request:
        meta=read(source)
        for f in meta['frames']:
            if f['file_path'] in meta['train_filenames'] and sha(frame/'semantic_data'/f['file_path'])!=request['source_rgb_sha256'].get(f['physical_camera']):
                raise ValueError('Background training RGB changed')
    checkpoint=Path(complete['checkpoint']['path']);config=Path(complete['configuration'])
    checkpoint_sha=sha(checkpoint)
    if checkpoint_sha!=complete['checkpoint']['sha256']:raise ValueError('Frozen background component checksum differs')
    dependency=dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,config=str(config),config_sha256=sha(config),
        source_metadata=str(source),source_metadata_sha256=digest,frozen=True,empty_subject_half_width=.18,start_step=15189)
    old=read(out/'dependency.json')
    if old is not None and old!=dependency:raise ValueError('Frozen per-frame background dependency changed on resume')
    write(out/'dependency.json',dependency)
    return dependency
