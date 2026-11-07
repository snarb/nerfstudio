"""Idle-exclusive, UUID-bound controller for a reviewed native-only probe."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
from audit_marshawn_artifact_native_rays import sha


def main(root):
    p=json.loads((root/'plan.json').read_text());release=json.loads((root/'release.json').read_text())
    assert release['approved_readonly_probe'] and release['plan_sha256']==sha(root/'plan.json')
    assert len(p['pixels'])<=12 and p['max_seconds']<=900 and p['optimizer_updates']==0
    lock=Path('/tmp/lookcloser_tuning_gpu_%d_%d.lock'%(os.getuid(),p['gpu'])).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def processes():
        raw=subprocess.check_output(['nvidia-smi','--id='+p['gpu_uuid'],'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
        return [dict(pid=int(s.split(',')[0]),mib=int(s.split(',')[1])) for s in raw.splitlines() if s.strip()]
    assert not processes()
    actual=subprocess.check_output(['nvidia-smi','--id='+str(p['gpu']),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert actual==p['gpu_uuid']
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    start=time.monotonic();child=None
    def interrupt(*_):raise RuntimeError('Interrupted')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        with (root/'worker.log').open('w') as log:
            child=subprocess.Popen([p['python'],str(root/'audit_marshawn_artifact_native_rays.py'),'--plan',str(root/'plan.json')],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            peak=0
            while child.poll() is None:
                current=processes();used=sum(x['mib'] for x in current);peak=max(peak,used)
                assert all(x['pid']==child.pid for x in current),'Unexpected GPU process'
                assert used<=10000 and time.monotonic()-start<=900
                with (root/'health.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),controller=os.getpid(),worker=child.pid,gpu_uuid=actual,processes=current))+'\n')
                time.sleep(3)
            assert child.returncode==0,'Worker failed; see worker.log'
        assert not processes()
        (root/'complete.json').write_text(json.dumps(dict(passed=True,elapsed=time.monotonic()-start,peak_mib=peak,plan_sha256=sha(root/'plan.json'),audit_sha256=sha(root/'trace/audit.json')))+'\n')
    except BaseException as e:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        (root/'failed.json').write_text(json.dumps(dict(error=repr(e)))+'\n');raise
    finally:lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
