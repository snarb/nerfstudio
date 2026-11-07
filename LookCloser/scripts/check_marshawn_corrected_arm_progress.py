"""Compact campaign health snapshot; no model/checkpoint imports or GPU work."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


REMOTE = r'''
import json,subprocess
from pathlib import Path
request=json.loads(REQUEST);root=Path(request['remote_root']);phase=request['phase'];arms={}
for arm in ('legacy1024','corrected1024'):
 run=root/'runs'/(phase+'_'+arm);progress=run/'progress.json';s=json.loads(progress.read_text())if progress.exists()else{}
 log=root/'remote_queue/jobs'/('correctedARM126316_'+phase+'_'+arm)/'worker.log';text=log.read_text()if log.exists()else''
 latest=run/'latest.json';proof=run/'resume_proof_126316.json'
 arms[arm]=dict(step=s.get('step'),attempts=s.get('attempts'),elapsed_seconds=s.get('elapsed_seconds'),
  source_resume_proof_exists=proof.exists(),latest=json.loads(latest.read_text())if latest.exists()else None,
  render_steps=sorted(p.name for p in(run/'renders').glob('step-*')),complete=(run/'complete.json').exists(),
  OOM='out of memory'in text.lower(),traceback='Traceback'in text)
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
print(json.dumps(dict(arms=arms,actual_GPU_apps=raw.strip().splitlines())))
'''


def snapshot(root, phase):
    read = lambda p: json.loads(Path(p).read_text())
    guard = root / 'external_guard_corrected_ARM_v1'
    policy = read(guard / 'policy.json')
    state = read(guard / phase / 'local_state.json')
    controller = read(guard / (phase + '_controller') / 'state.json')
    request = json.dumps(dict(remote_root=policy['remote_root'], phase=phase))
    script = REMOTE.replace('REQUEST', repr(request))
    remote = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             policy['address'], 'python3', '-'], input=script,
                            capture_output=True, text=True, check=True, timeout=30)
    result = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), phase=phase,
        controller_pid=state.get('controller', {}).get('pid'),
        controller_present=Path('/proc', str(state.get('controller', {}).get('pid'))).exists(),
        remote_guardian_alive=state.get('remote', {}).get('guardian_alive'),
        guardian_failed=state['failed'] or state.get('remote', {}).get('state', {}).get('failed', False),
        controller_jobs={k: v['status'] for k, v in controller['jobs'].items()},
        observed_workers=[dict(name=v['name'], pid=v['worker']['pid'], memory_MiB=v['memory_MiB'],
                               age=v['age']) for v in state.get('remote', {}).get('state', {}).get('live', [])],
        **json.loads(remote.stdout))
    with (root / ('manual_' + phase + '_supervision.jsonl')).open('a') as stream:
        stream.write(json.dumps(result) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', default='production', choices=['production'])
    args = parser.parse_args()
    snapshot(args.root, args.phase)
