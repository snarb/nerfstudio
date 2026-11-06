"""Hash-bound, idle-exclusive small LC ray probe; never trains or changes a source artifact."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def processes(gpu):
    value=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    return [dict(pid=int(line.split(',')[0]),mib=int(line.split(',')[1])) for line in value.splitlines() if line.strip()]


def verify(plan):
    for path,digest in plan['bound_files'].items():
        if sha(path)!=digest:raise ValueError('Bound source changed: '+path)
    for source in plan['source_trees']:
        manifest=json.loads(Path(source['manifest']).read_text())
        for relative,digest in manifest[source['key']].items():
            if sha(Path(source['root'])/relative)!=digest:raise ValueError('Frozen code changed: '+relative)


def parity(trace,model):
    import numpy as np
    from PIL import Image
    audit=json.loads((trace/'rays/audit.json').read_text())
    if audit['checkpoint_sha256']!=model['checkpoint_sha256'] or audit['step']!=model['step'] or not audit['frozen_state_unchanged']:
        raise ValueError('Wrong or mutated traced checkpoint')
    chain=json.loads((trace/'coordinate_chain.json').read_text())
    if chain['feature_reweighting_strength']!=model['feature_reweighting_strength']:
        raise ValueError('Wrong feature reweighting regime')
    if audit['runtime'].get('feature_reweighting_strength',model['feature_reweighting_strength'])!=model['feature_reweighting_strength']:
        raise ValueError('Checkpoint runtime overrides the declared feature regime')
    rows=[]
    for row in audit['rows']:
        native=row['profiles']['native'];profile=trace/'rays'/native['trace']
        with np.load(profile) as arrays:
            weights=arrays['actual_weights'].astype(float)
            foreground=weights@arrays['rgb'].reshape(-1,3).astype(float)
        rgb=np.asarray(native['model_rgb'],float)
        alpha=float(native['model_accumulation'])
        item=dict(pixel=row['pixel'],native_alpha=alpha,actual_weight_sum=float(weights.sum()),
                  alpha_error=abs(alpha-float(weights.sum())),foreground_RGB_error=float(np.abs(foreground-rgb).max()))
        if item['alpha_error']>2e-6 or item['foreground_RGB_error']>2e-6:
            raise ValueError('Original native weight RGB/alpha parity failed')
        if audit['split']=='eval':
            image=Image.open(model['reference_PNGs'][row['pixel']['camera_name']])
            if image.mode!='RGB':raise ValueError('Expected archived raw RGB PNG')
            target=np.asarray(image)[row['pixel']['y'],row['pixel']['x']].astype(float)
            item['PNG_byte_float_error']=float(np.abs(rgb*255-target).max())
            if item['PNG_byte_float_error']>1.01:raise ValueError('Native ray differs from archived PNG by more than one byte')
        rows.append(item)
    result=dict(rows=rows,archived_PNGs_have_alpha=False,
                alpha_reference='Original renderer accumulation, not RGB-only PNG',
                query_count=len(rows),no_fit=True)
    write(trace/'native_parity.json',result)


def main(root):
    plan=json.loads((root/'plan.json').read_text());release=json.loads((root/'release.json').read_text())
    if release.get('approved_readonly_probe')is not True or release['plan_sha256']!=sha(root/'plan.json'):
        raise ValueError('Exact read-only probe release required')
    if plan['total_unique_rays']>64 or plan['optimizer_updates']!=0:raise ValueError('Unreviewed probe scope')
    verify(plan)
    if (root/'complete.json').exists() or (root/'failed.json').exists():raise ValueError('Preserve existing generation')
    gpu=plan['gpu'];child=None;started=time.monotonic();owned=set();checks=[]
    lock=Path('/tmp/lookcloser_tuning_gpu_%d_%d.lock'%(os.getuid(),gpu)).open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if processes(gpu):lock.close();raise RuntimeError('GPU must be idle before read-only probe')
    def interrupted(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def check(phase):
        current=processes(gpu)
        if any(row['pid']not in owned for row in current):raise RuntimeError('Unexpected process on exclusive GPU')
        if sum(row['mib'] for row in current)>plan['max_process_mib']:raise RuntimeError('Probe process VRAM budget exceeded')
        if time.monotonic()-started>plan['max_seconds']:raise TimeoutError('Bounded probe time exhausted')
        row=dict(time=time.time(),pid=os.getpid(),phase=phase,gpu=gpu,processes=current)
        checks.append(row)
        with(root/'health.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n')
        write(root/'status.json',row)
    try:
        for model in plan['models']:
            for query in plan['queries']:
                label=model['label']+'_'+query['split'];output=root/'traces'/label
                output.parent.mkdir(exist_ok=True)
                command=[plan['python'],str(root/'helpers/audit_marshawn_semantic_checkpoint.py'),
                    '--data',plan['data'],'--output',str(output),'--queries',query['path'],
                    '--receipt',model['receipt'],'--checkpoint',model['checkpoint'],'--config',model['config'],
                    '--code-root',model['code_root'],'--gpu',str(gpu),'--memory-cap-gib',str(plan['allocator_cap_gib'])]
                environment=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',
                    PYTHONPATH=str(root/'helpers')+':'+model['code_root']+':'+str(Path(model['code_root'])/'LookCloser/scripts'))
                check(label+'_before')
                with(root/(label+'.log')).open('w')as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
                    owned.add(child.pid)
                    while child.poll()is None:check(label);time.sleep(5)
                    rc=child.returncode;child=None
                if rc:raise RuntimeError('Read-only trace failed: '+label)
                check(label+'_cache_exit')
                if processes(gpu):raise RuntimeError('Trace GPU resources remain after exit')
                parity(output,model)
                with (root/(label+'_partition.log')).open('w') as partition_log:
                    subprocess.run([plan['python'],str(root/'helpers/partition_marshawn_trace_heights.py'),
                        '--audit',str(output/'rays/audit.json'),'--ready',str(Path(plan['data'])/'ready.json'),
                        '--output',str(output/'height_bins')],env=dict(environment,CUDA_VISIBLE_DEVICES=''),check=True,
                        stdout=partition_log)
        verify(plan)
        write(root/'complete.json',dict(passed=True,plan_sha256=sha(root/'plan.json'),optimizer_updates=0,
            source_mutation=False,physical_geometry_certified=False,bounded_crop_release=False,
            peak_process_mib=max((sum(p['mib']for p in row['processes'])for row in checks),default=0),
            elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll()is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        write(root/'failed.json',dict(error=repr(error),source_mutation=False));raise
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
