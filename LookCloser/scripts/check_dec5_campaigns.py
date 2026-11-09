#!/usr/bin/env python3
"""One compact root-supervision call for production and the two full HD pilots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
from pathlib import Path

SILVER = r'''
from pathlib import Path
import sys,json,time
root=Path('/home/ubuntu/lookcloser/campaigns/dec5_sequence50_20261028')
sys.path.insert(0,str(root/'revisions/v6/code/LookCloser/scripts'))
from run_dec5_sequence import snapshot,read
s=snapshot(root,read(root/'manifest.json'));s['check_type']='root hourly supervision'
if RECORD:
    with (root/'supervision.jsonl').open('a') as log:log.write(json.dumps(s)+'\n')
active=[r for r in s['frames'] if r['worker_alive']]
print(json.dumps(dict(time=s['time'],counts=s['counts'],controller_alive=s['controller_alive'],
    recipe_revision=read(root/'recipe.json',{}).get('revision'),background_seed=read(root/'recipe.json',{}).get('background_seed'),
    worker_count=len(active),failures=sum(r['failure_count'] for r in s['frames']),
    oom=any(r['oom_evidence'] for r in s['frames']),free_gib=round(s['persistent_free_bytes']/2**30,1),
    gpus=[{k:g[k] for k in ['index','used_mib','utilization']} for g in s['gpus']],
    active=[{k:r[k] for k in ['frame','phase','completed','total','step'] if k in r} for r in active],
    postflight_complete=(root/'campaign_complete.json').exists())))
'''

CLEVER = r'''
from pathlib import Path
import json,time,subprocess
root=Path('/home/ubuntu/lookcloser_transfer/recipe_pilot_20261028')
def read(path,default=None):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return default
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        args=(p/'cmdline').read_bytes().split(b'\0')
        state=(p/'stat').read_text().rsplit(')',1)[1].split()
    except (OSError,ProcessLookupError):continue
    if state[0]!='Z' and args and b'python' in args[0]:
        processes.append((int(p.name),b'\0'.join(args),state[19]))
checks=[]
for name,revision,script in [('stable_full_v6','hd_stable_v6','run_dec5_stable_recipe_pilot.py'),
                             ('hair_full_v7','hd_hair_v7','run_dec5_hair_recipe_pilot.py')]:
    campaign=root/name;frame=campaign/'frames/000901';training=frame/('training_'+revision)
    workers=[dict(pid=pid,starttime=start) for pid,args,start in processes
             if str(campaign).encode() in args and script.encode() in args]
    log=root/(name+'.log')
    with log.open('rb') as f:
        f.seek(0,2);f.seek(max(0,f.tell()-131072));tail=f.read().lower()
    history=read(training/'history.json',[])
    checks.append(dict(name=name,alive=bool(workers),workers=workers,
        complete=(campaign/'complete.json').exists(),progress=read(frame/'progress.json'),
        latest=read(training/'latest.json'),sealed_gates=[r['step'] for r in history],
        oom=b'out of memory' in tail))
native=root.parent/'native_full_pair_20261029'
for arm in ['control','warm']:
    campaign=native/('full_'+arm)
    if not campaign.exists():continue
    frame=campaign/'frames/000901';training=frame/('training_native_full_'+arm)
    workers=[dict(pid=pid,starttime=start) for pid,args,start in processes
             if str(native).encode() in args and b'run_dec5_native_full_pair.py' in args
             and (b'--arm\x00'+arm.encode()) in args]
    log=native/('full_'+arm+'.log')
    with log.open('rb') as f:
        f.seek(0,2);f.seek(max(0,f.tell()-65536));tail=f.read().lower()
    checks.append(dict(name='native_full_'+arm,alive=bool(workers),workers=workers,
        complete=(campaign/'complete.json').exists(),progress=read(frame/'progress.json'),
        latest=read(training/'latest.json'),sealed_gates=[r['step'] for r in read(training/'history.json',[])],
        oom=b'out of memory' in tail))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                            '--format=csv,noheader,nounits'],text=True,timeout=20).strip()
path_root=root.parent/'background_path_20261029'
path_worker=read(path_root/'worker.json',{})
path_alive=any(pid==path_worker.get('pid') and start==str(path_worker.get('starttime')) for pid,args,start in processes)
s=dict(time=time.time(),check_type='root hourly supervision',checks=checks,gpu=gpu,
    background_path16=dict(alive=path_alive,complete=(path_root/'complete.json').exists(),progress=read(path_root/'progress.json')))
native_path=root.parent/'native_background_path96_20261029'
native_worker=read(native_path/'worker.json',{})
s['native_background_path96']=dict(alive=any(pid==native_worker.get('pid') and start==str(native_worker.get('starttime')) for pid,args,start in processes),
    complete=(native_path/'complete.json').exists(),progress=read(native_path/'progress.json'),migration=read(native_path/'migration.json'))
if RECORD:
    with (root/'supervision.jsonl').open('a') as log:log.write(json.dumps(s)+'\n')
    if native.exists():
        with (native/'supervision.jsonl').open('a') as log:log.write(json.dumps(s)+'\n')
print(json.dumps(s))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record',action='store_true');p.add_argument('--brief',action='store_true');a=p.parse_args()
    def check(item):
        host,source=item
        source='RECORD='+repr(a.record)+'\n'+source
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
                'ubuntu@'+host,'python3','-'],input=source,text=True,capture_output=True,
                check=True,timeout=55)
            return host,json.loads(result.stdout)
        except (subprocess.SubprocessError,json.JSONDecodeError) as e:
            return host,dict(supervision_interrupted=True,error=str(e))
    with ThreadPoolExecutor(max_workers=2) as pool:
        result=dict(pool.map(check,[('silver-sable',SILVER),('clever-shadow',CLEVER)]))
    local_root=Path('/fsx/tmp/lookcloser_hair_001083_20261027_1330/local_20261028/sequence901_probe')
    local_gpus=None
    for name in ['neural_background','foreground_background_control','foreground_background_volume','full_background_volume',
                 'native_background_control','native_background_volume',
                 'native_boundary_control','native_boundary_uniform','native_boundary_frequency','native_warm_background','native96_bundle']:
        local=local_root/name
        if not (local/'worker.json').exists():continue
        from run_dec5_sequence import read,alive,gpu_status
        if local_gpus is None:local_gpus=gpu_status()
        progress=local/'progress.json';training=local
        if (local/'frames/000901').exists():
            progress=local/'frames/000901/progress.json'
            training=local/'frames/000901'/('training_'+read(local/'recipe.json')['revision'])
        state=dict(worker_alive=alive(read(local/'worker.json')),worker=read(local/'worker.json'),
            progress=read(progress),checkpoint=read(training/'latest.json'),
            complete=(local/'complete.json').exists(),gpus=local_gpus)
        log=local.parent/(name+'.log')
        with log.open('rb') as handle:
            handle.seek(0,2);handle.seek(max(0,handle.tell()-65536));tail=handle.read().lower()
        state['oom']=b'out of memory' in tail
        if a.record:
            import time
            state['time']=time.time()
            with (local/'supervision.jsonl').open('a') as handle:handle.write(json.dumps(state)+'\n')
        result['local-'+name]=state
    if a.brief:
        for key,state in list(result.items()):
            if key.startswith('local-'):
                result[key]=dict(alive=state['worker_alive'],complete=state['complete'],oom=state['oom'],
                    step=(state['progress'] or {}).get('step'),checkpoint_step=(state['checkpoint'] or {}).get('step'))
        result['local_gpus']=local_gpus
        clever=result.get('clever-shadow',{})
        if 'checks' in clever:
            clever['checks']=[dict(name=c['name'],alive=c['alive'],complete=c['complete'],oom=c['oom'],
                step=(c['latest'] or {}).get('step'),eval_gates=len(c['sealed_gates'])) for c in clever['checks']]
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
