#!/usr/bin/env python3
"""Actual20k component fit surviving a terminated factory parent; research only."""
from pathlib import Path
import argparse,subprocess,sys,time,os,signal
from run_dec5_sequence import read,write,alive
from dec5_sequence_worker import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame-root',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();frame=a.frame_root;out=frame/'neural_background'
    if out.exists():raise FileExistsError('Crash test requires a fresh component directory')
    code='from pathlib import Path; from dec5_sequence_background import ensure; import sys; ensure(Path(sys.argv[1]),Path(sys.argv[2]),dict(kind="bounded_ngp_v1",steps=20000,start_step=15189))'
    command=[sys.executable,'-c',code,str(frame),str(a.root)]
    def start(name):
        with (frame/(name+'.log')).open('w') as log:
            return subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    first=start('factory_first');deadline=time.monotonic()+300
    while read(out/'progress.json',{}).get('step',-1)<500:
        if first.poll() is not None:raise RuntimeError('Initial factory exited before crash gate')
        if time.monotonic()>deadline:raise TimeoutError('Initial fitting observation deadline; inspect live PIDs before retry')
        time.sleep(2)
    worker=read(out/'worker.json');assert alive(worker)
    initial_config=read(out/'latest.json');config=next((out/'training').rglob('config.yml'));config_sha=sha(config)
    os.kill(first.pid,signal.SIGTERM);first.wait(timeout=20)
    if not alive(worker):raise RuntimeError('Fitting child did not survive factory termination')
    second=start('factory_recovery')
    write(frame/'autonomy_progress.json',dict(phase='parent_terminated_child_continues',terminated_factory_pid=first.pid,
        surviving_fitter=worker,recovery_factory_pid=second.pid,time=time.time()))
    result=second.wait()
    if result:raise RuntimeError('Recovery factory failed; inspect recovery log')
    complete=read(out/'complete.json');request=read(out/'request.json')
    if read(out/'worker.json')['pid']!=worker['pid']:raise ValueError('A second fitting worker rewrote the model')
    if complete['checkpoint']['step']!=19999 or complete['smoke']:raise ValueError('Full fitting did not finish')
    if sha(config)!=config_sha:raise ValueError('Cache adoption rewrote the sealed configuration')
    metadata=read(frame/'semantic_data/transforms.json');excluded={f['physical_camera'] for f in metadata['frames'] if f['file_path'] in metadata['val_filenames']}
    if excluded&set(request['source_rgb_sha256']):raise ValueError('Held-out RGB entered fitting provenance')
    write(frame/'autonomy_complete.json',dict(actual_background_updates=20000,source_frame=metadata['frame_id'],
        fitting_child_survived_factory_sigterm=True,fitting_worker_pid_unchanged=True,
        recovery_factory_returncode=result,sealed_configuration_unchanged=True,
        checkpoint_sha256=complete['checkpoint']['sha256'],source_rgb_cameras=len(request['source_rgb_sha256']),
        excluded_cameras=sorted(excluded),production50_changed=False,finished=time.time()))


if __name__=='__main__':main()
