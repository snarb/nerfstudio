"""Exclusive, hash-bound, time-bounded controller for one read-only floor selector diagnostic."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
from run_marshawn_floor_trace_probe import verify,processes,write,sha


def main(root):
    plan=json.loads((root/'plan.json').read_text());release=json.loads((root/'release.json').read_text())
    assert release['approved_readonly_probe']is True and release['plan_sha256']==sha(root/'plan.json')
    assert plan['cut']==-.02 and plan['total_TRAIN_controls']==109 and plan['max_seconds']==900
    assert sum((r[2]-r[0])*(r[3]-r[1])for r in plan['regions'].values())<=300000
    if(root/'complete.json').exists()or(root/'failed.json').exists():raise ValueError('Existing generation must remain intact')
    verify(plan);gpu=plan['gpu'];lock=Path('/tmp/lookcloser_tuning_gpu_%d_%d.lock'%(os.getuid(),gpu)).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not processes(gpu),'GPU must be idle';started=time.monotonic();child=None;peak=0
    def interrupt(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    model=plan['model'];output=root/'gpu_trace'
    command=[plan['python'],str(root/'helpers/audit_marshawn_semantic_checkpoint.py'),'--data',plan['data'],'--output',str(output),'--queries',plan['queries'],
             '--receipt',model['receipt'],'--checkpoint',model['checkpoint'],'--config',model['config'],'--code-root',model['code_root'],'--gpu',str(gpu),'--memory-cap-gib','8']
    environment=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',FLOOR_POINT_GATE_PLAN=str(root/'plan.json'),
        PYTHONPATH=str(root/'helpers')+':'+model['code_root']+':'+str(Path(model['code_root'])/'LookCloser/scripts'))
    try:
        with(root/'worker.log').open('w')as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
            while child.poll()is None:
                current=processes(gpu);used=sum(p['mib']for p in current);peak=max(peak,used)
                if any(p['pid']!=child.pid for p in current):raise RuntimeError('Unexpected GPU process')
                if used>10000 or time.monotonic()-started>900:raise RuntimeError('Resource/time bound exceeded')
                record=dict(time=time.time(),controller_pid=os.getpid(),worker_pid=child.pid,gpu=gpu,processes=current,elapsed=time.monotonic()-started)
                write(root/'status.json',record)
                with(root/'health.jsonl').open('a')as f:f.write(json.dumps(record)+'\n')
                time.sleep(5)
            if child.returncode:raise RuntimeError('Worker failed, see worker.log')
        assert not processes(gpu)
        imported=json.loads((output/'imported_helpers.json').read_text())
        expected={str((root/'helpers'/name).resolve()):sha(root/'helpers'/name)for name in['probe_marshawn_occupancy.py','audit_marshawn_tuning_checkpoint.py','audit_marshawn_semantic_checkpoint.py']}
        assert imported==expected,'Actual helper binding mismatch'
        result=json.loads((output/'rays/diagnostic.json').read_text())
        assert result['passed']and result['checkpoint_sha256']==model['checkpoint_sha256']and result['actual_worker_sha256']==sha(root/'helpers/marshawn_floor_point_gate_worker.py')
        verify(plan)
        write(root/'complete.json',dict(passed=True,plan_sha256=sha(root/'plan.json'),source_mutation=False,optimizer_updates=0,empirical_prior_not_emptyspace_certificate=True,
            diagnostic_sha256=sha(output/'rays/diagnostic.json'),peak_process_mib=peak,elapsed_seconds=time.monotonic()-started))
    except BaseException as e:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll()is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        write(root/'failed.json',dict(error=repr(e),source_artifacts_readonly=True));raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
