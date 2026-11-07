"""Release-gated phase launcher with bounded guards for its own leased workers only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    temp = Path(str(path)+'.tmp'); temp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(temp,path)


def main(root, phase):
    policy = read(root/'phase_policy.json')
    for path,digest in policy['bindings_sha256'].items():
        if sha(path) != digest: raise ValueError('Prepared phase artifact changed: '+path)
    for arm in policy['arms']:
        spec_path = root/'specs'/(phase+'_'+arm+'.json'); spec=read(spec_path); release=read(spec['release'])
        auth = 'diagnostic_authorized' if phase=='diagnostic' else 'technical_authorized'
        if release.get('reviewer')!='root' or release.get(auth) is not True or release.get('spec_sha256')!=sha(spec_path):
            raise ValueError('Exact phase/spec release required')
    if phase!='zero':
        previous=root/('zero_verification.json' if phase=='technical' else 'technical_verification.json')
        if not read(previous).get('passed'): raise ValueError('Previous proof gate missing')
    guard=root/(phase+'_guard');guard.mkdir(exist_ok=True)
    lock=(guard/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (guard/'complete.json').exists():raise ValueError('Completed phase cannot silently relaunch')
    write(guard/'launch.json',dict(pid=os.getpid(),time=time.time(),phase=phase))
    controller_root=root/(phase+'_controller'); plan=read(root/(phase+'_plan.json'))
    subprocess.run([sys.executable,policy['controller'],'launch','--root',str(controller_root),'--manifest',str(root/(phase+'_plan.json'))],check=True)
    violations=[];term_sent={};last=0
    while True:
        apps=[]
        for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
            cols=[x.strip()for x in line.split(',')]
            if len(cols)==3:apps.append(cols)
        observations=[]
        for job in plan['jobs']:
            p=root/'remote_queue/jobs'/job['name']/'receipt.json'
            if not p.exists():continue
            receipt=read(p);worker=receipt.get('worker');row=dict(name=job['name'],status=receipt['status'])
            if worker:
                proc=Path('/proc')/str(worker['pid'])
                if proc.exists():
                    try:
                        fields=(proc/'stat').read_text().split(') ',1)[1].split();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                    except FileNotFoundError:continue
                    identity=fields[19]==str(worker['start']) and boot==worker['boot']
                    own=[r for r in apps if r[1]==str(worker['pid'])];memory=sum(int(r[2])for r in own);age=time.time()-receipt['started']
                    row.update(identity=identity,process_state=fields[0],memory_MiB=memory,seconds=age)
                    limit=policy['max_seconds_per_worker'][phase]
                    bad=memory>16384 or any(r[0]!=receipt['gpu_uuid']for r in own) or age>limit
                    if identity and fields[0]!='Z' and bad:
                        if 'train_marshawn_focused_clothing_geometry.py' not in command or str(root/'specs'/(phase+'_')) not in command:
                            raise RuntimeError('Refuse to signal unrelated process')
                        if worker['pid'] not in term_sent:
                            os.killpg(worker['pid'],signal.SIGTERM);term_sent[worker['pid']]=time.time();violations.append(row.copy())
                        elif time.time()-term_sent[worker['pid']]>25:
                            os.killpg(worker['pid'],signal.SIGKILL)
                        row['own_worker_guard_violation']=True
            observations.append(row)
        status=dict(time=time.time(),jobs=observations,violations=violations);write(guard/'status.json',status)
        if time.time()-last>30:
            with (guard/'checks.jsonl').open('a')as stream:stream.write(json.dumps(status)+'\n')
            last=time.time()
        state_path=controller_root/'state.json'
        if state_path.exists():
            state=read(state_path)
            if all(j['status']in('complete','failed','blocked','skipped')for j in state['jobs'].values()):
                passed=not violations and all(j['status']=='complete'for j in state['jobs'].values())
                write(guard/'complete.json',dict(passed=passed,time=time.time(),jobs={n:j['status']for n,j in state['jobs'].items()},violations=violations))
                if not passed:raise RuntimeError('Phase did not pass; preserve all evidence')
                return
        time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['zero','technical','diagnostic'],required=True);a=p.parse_args();main(a.root,a.phase)
