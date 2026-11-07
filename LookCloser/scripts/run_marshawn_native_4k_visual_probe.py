"""One reviewed native4K resource render after paired production and cache exit."""
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(p,a):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(a,indent=2)+'\n');os.replace(q,p)


def memory_available():
    return int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024


def processes():
    out=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    return [dict(pid=int(r.split(',')[0]),mib=int(r.split(',')[1]))for r in out.splitlines()if r.strip()]


def main(release_path):
    release=json.loads(release_path.read_text());root=release_path.parent;plan=json.loads((root/'plan.json').read_text())
    if release.get('approved_resource_probe')is not True or release['plan_sha256']!=sha(root/'plan.json'):raise ValueError('Exact resource-probe release required')
    if not(Path(plan['production'])/'complete.json').exists():raise ValueError('Native pair must finish before thisprobe')
    pid=plan['production_controller_pid']
    command_path=Path('/proc/%d/cmdline'%pid)
    if command_path.exists()and b'run_marshawn_native_resolution_pair.py'in command_path.read_bytes():
        raise ValueError('Historical pair controller stillalive; require explicit cache/process exit')
    for process in Path('/proc').glob('[0-9]*/cmdline'):
        try:command=process.read_bytes()
        except (FileNotFoundError,PermissionError):continue
        if any(worker in command for worker in [b'run_marshawn_native_resolution_chunk.py',b'evaluate_marshawn_native_reference.py']):
            raise ValueError('Native worker/evaluator cache may still be live')
    if(root/'complete.json').exists()or(root/'failure.json').exists():raise ValueError('Preserve existing generation')
    def verify():
        for path,h in plan['bound_files'].items():
            if sha(path)!=h:raise ValueError('Bound source changed: '+path)
    verify();lock=Path(plan['GPU_lock']).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not processes(),'GPU must be idle';assert memory_available()>=20*2**30;assert shutil.disk_usage(root).free>=20*2**30
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONPATH=':'.join([str(root/'code')]+plan['python_path']))
    version=subprocess.check_output([plan['python'],'-c','import torch,json; print(json.dumps([torch.__version__,torch.version.cuda]))'],env=environment,text=True)
    assert json.loads(version)==[plan['runtime']['torch'],plan['runtime']['cuda']]
    command=[plan['python'],plan['renderer'],'--config',plan['config'],'--checkpoint',plan['source']['checkpoint'],'--checkpoint-sha256',plan['source']['checkpoint_sha256'],
             '--box',plan['box'],'--output',plan['output'],'--camera','HC','--chunk','512','--diagnostic',plan['diagnostic']]
    for path in plan['python_path']:command+=['--python-path',path]
    child=None;start=time.monotonic();peak=0;minimum=memory_available()
    def interrupt(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        with(root/'render_stdout.log').open('w')as log:
            child=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll()is None:
                current=processes();used=sum(r['mib']for r in current);available=memory_available();peak=max(peak,used);minimum=min(minimum,available)
                if any(r['pid']!=child.pid for r in current):raise RuntimeError('Unexpected GPU process')
                if used>plan['maximum_GPU_MiB']or available<4*2**30 or time.monotonic()-start>1200:raise RuntimeError('Resource bound exceeded')
                row=dict(time=time.time(),controller_pid=os.getpid(),worker_pid=child.pid,processes=current,available_RAM_bytes=available,elapsed_seconds=time.monotonic()-start)
                write(root/'status.json',row)
                with(root/'resources.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
                time.sleep(1)
            if child.returncode:raise RuntimeError('Renderer failed; preserve original log/evidence')
        assert not processes();verify();receipt=Path(plan['output'])/'receipt.json';a=json.loads(receipt.read_text())
        assert a['checkpoint_sha256']==plan['source']['checkpoint_sha256']and a['updates']==0 and a['field_normalization_and_grid_unchanged']
        assert a['resolution_scale']==2 and a['camera_runtime']['w']==3840 and a['camera_runtime']['h']==2024
        write(root/'complete.json',dict(passed=True,receipt_sha256=sha(receipt),elapsed_seconds=time.monotonic()-start,peak_GPU_MiB=peak,minimum_available_RAM_bytes=minimum,
            remaining_eight_renders_authorized=False,requires_resource_and_visual_review=True,primary_FHD_score_unchanged=True))
    except BaseException as e:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll()is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        write(root/'failure.json',dict(error=repr(e),source_artifacts_readonly=True));raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True);main(p.parse_args().release)
