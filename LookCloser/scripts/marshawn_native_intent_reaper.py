"""Retire only source-bound completed native GPU demand, independent of owners.

No GPU process is signalled. Pending/unknown/unproved work remains reserved.
Scientific job completion and the exact immutable plan decide retirement;
CPU source review may continue after native inference has finished.
"""
from pathlib import Path
import argparse,json,sys,time,hashlib,subprocess
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107')
sys.path[:0]=[str(V/'operations/sam3_assist/code'),str(V/'code/LookCloser/scripts')]
import marshawn_sam3_assist as io
import marshawn_tuning_controller as fleet


def completion_proof(request):
    path=request.get('source_bound_plan')
    if not path or not request.get('source_bound_plan_sha256'):return None
    path=Path(path)
    if not path.resolve().is_relative_to((V/'operations').resolve())or io.sha(path)!=request['source_bound_plan_sha256']:return None
    plan=io.read(path);state_path=Path(request.get('actual_state',str(path.parent/'controller/state.json')))
    if not state_path.resolve().is_relative_to((V/'operations').resolve())or not state_path.exists():return None
    state=io.read(state_path)
    proof=dict(source_plan=str(path),source_plan_sha256=io.sha(path),actual_state=str(state_path),actual_state_sha256=io.sha(state_path))
    if 'jobs'in plan:
        expected={j['name']for j in plan['jobs']}
        if not expected or state.get('manifest_sha256')!=fleet.digest(plan):return None
        if not all(state.get('jobs',{}).get(n,{}).get('status')=='complete'for n in expected):return None
        proof.update(kind='EXACT_GENERIC_NATIVE_PLAN_ALL_JOBS_COMPLETE',jobs=sorted(expected))
    elif 'records'in plan:
        expected={r['camera']for r in plan['records']};done=state_path.parent/'all66_review_ready.json'
        if len(expected)!=66 or not done.exists():return None
        marker=io.read(done)
        if marker.get('configuration_sha256')!=io.sha(path)or marker.get('frame')!=plan['frame']or set(marker.get('cameras',{}))!=expected:return None
        for name in expected:
            row=state.get('cameras',{}).get(name,{})
            if not row.get('collected')or state.get('jobs',{}).get(row.get('name'),{}).get('status')!='complete':return None
        proof.update(kind='EXACT_NATIVE_ALL66_COLLECTED',completion=str(done),completion_sha256=io.sha(done),cameras=66)
    else:return None
    return proof


def cycle():
    root=V/'operations/native10_requests';retired=[]
    for path in root.glob('*.json'):
        request=io.read(path)
        if request.get('status')=='complete':continue
        proof=completion_proof(request)
        if not proof:continue
        before=io.sha(path);again=io.read(path)
        if io.sha(path)!=before:continue
        result=dict(again,status='complete',no_active_GPU_requirement=True,outcome='source_bound_native_GPU_jobs_complete_CPU_review_may_continue',completion_reaper=proof,completion_reaper_code_sha256=io.sha(__file__),retired_at=time.time())
        io.write(path,result);retired.append(str(path))
        with(root/'completion_reaper_checks.jsonl').open('a')as f:f.write(json.dumps(dict(time=time.time(),request=str(path),old_request_sha256=before,new_request_sha256=io.sha(path),proof=proof))+'\n')
    return retired


def run():
    root=V/'operations/native10_requests';held=fleet.lock(root/'completion_reaper.lock')
    if held is None:return
    initial=io.sha(__file__);io.write(root/'completion_reaper_identity.json',dict(process=fleet.identity(),code_sha256=initial))
    while True:
        if io.sha(__file__)!=initial:raise ValueError('Released completion reaper code changed')
        cycle();time.sleep(5)


def watchdog():
    root=V/'operations/native10_requests';held=fleet.lock(root/'completion_reaper_watchdog.lock')
    if held is None:return
    initial=io.sha(__file__);io.write(root/'completion_reaper_watchdog_identity.json',dict(process=fleet.identity(),code_sha256=initial))
    while True:
        if io.sha(__file__)!=initial:raise ValueError('Released completion reaper code changed')
        path=root/'completion_reaper_identity.json';old=io.read(path).get('process')if path.exists()else None
        if not fleet.alive(old):
            with(root/'completion_reaper.log').open('ab')as log:
                subprocess.Popen([sys.executable,__file__,'run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','check','watchdog']);a=p.parse_args()
    if a.action=='run':run()
    elif a.action=='watchdog':watchdog()
    else:print(cycle())
